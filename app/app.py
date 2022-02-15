''' Vanilla app for getting MP concentration prediction using pretrained MAR model. '''

import io

import numpy as np
import pandas as pd
import streamlit as st
import plotly_express as px

from utils import (load_coeff_matrix, run_simulation, calc_statistics, 
                   prepare_atlas_with_concentrations)

def upload_new_data(caption):
    file_path = st.file_uploader(caption, type=['.csv'])
    if file_path is not None:
        data = np.genfromtxt(file_path, delimiter=',')
        return data
    
def visualize_concentration_lineplot(input_vec, output_vec): 
    # some labels are missing in the atlas - create custom index
    region_labels = np.concatenate((np.arange(1, 35), np.arange(37, 81), np.arange(83, 171)), axis=None)
    chart_df = pd.DataFrame({
        't0': input_vec,
        't1 pred': output_vec
    }, index=region_labels) 
    
    fig = px.line(chart_df, markers=True, height=500, width=1000)
    fig.update_layout(
        xaxis_title='ROI (index of brain region based on AAL atlas)',
        yaxis_title='concentration of MP',
        title='MP concentration in each brain region')

    return fig

def visualize_concentration_image(brain, sliced_axis=0):  
    brain = np.rot90(brain)
    fig = px.imshow(brain[:, :, :].T, animation_frame=sliced_axis, 
                    color_continuous_scale='Reds',
                    zmin=0, width=1000)
    return fig

@st.cache
def convert_array(array):
    return pd.DataFrame(array).to_csv().encode('utf-8')

def main():
    st.title('BrainSpread')
    st.header('Misfolded proteins spreading')
    
    st.subheader('Data loading')
    t0_concentration = upload_new_data(caption='Upload CSV file with regions concentration (shape: 1x116)')
    
    if t0_concentration is not None:
        t1_concentration_pred = run_simulation(t0_concentration)
        stats, total_sum, top_brain_regions = calc_statistics(t1_concentration_pred)
        region_concentrations = prepare_atlas_with_concentrations(t1_concentration_pred)
    
        st.subheader('Results')        
        st.text('Statistics of t1 predictions')
        st.code(stats)
        
        st.text('Total sum of concentration')
        st.code(total_sum)
        
        st.text('Brain region index with the highest MP concentration (top 5)')
        st.code(top_brain_regions)
        
        st.text('Visualization')
        fig1 = visualize_concentration_lineplot(t0_concentration, t1_concentration_pred)
        st.plotly_chart(fig1)

        fig_col1 = visualize_concentration_image(region_concentrations)
        st.plotly_chart(fig_col1)
        
        st.subheader('Download')
        st.download_button(
            'Download a CSV file with predictions',
            convert_array(t1_concentration_pred),
            't1_pred.csv',
            'text/csv',
            key='download-csv'
            )
        
        buffer = io.StringIO()
        fig1.write_html(buffer, include_plotlyjs='cdn')
        html_bytes = buffer.getvalue().encode()

        st.download_button(
            label='Download figure as HTML',
            data=html_bytes,
            file_name='figure.html',
            mime='text/html'
        )

if __name__ == '__main__':
    main()
        