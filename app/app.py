import numpy as np
import pandas as pd
import streamlit as st
import plotly_express as px

def upload_new_data(caption):
    file_path = st.file_uploader(caption, type=['.csv'])
    if file_path is not None:
        data = np.genfromtxt(file_path, delimiter=",")
        return data
    
def load_coeff_matrix():
    coeff_path = '../results/MAR/coeff_matrix.csv'
    data = np.genfromtxt(coeff_path, delimiter=",")
    return data

def visualize_concentration(input_vec, output_vec): 
    chart_df = pd.DataFrame({
        't0': input_vec,
        't1 pred': output_vec
    }) 
    
    fig = px.line(chart_df, markers=True, height=500, width=1000)
    fig.update_layout(
        xaxis_title='ROI (index of brain region based on AAL atlas)',
        yaxis_title='concentration of MP',
        title='MP concentration in each brain region')

    st.plotly_chart(fig)

def run_simulation(t0_concentration):
    coeff_matrix = load_coeff_matrix()
    t1_concentration_pred = coeff_matrix @ t0_concentration
    return t1_concentration_pred

def calc_statistics(concentration):
    stats = pd.DataFrame({'concentration': concentration}).describe()
    total_sum = np.sum(concentration)
    top_brain_regions = np.argsort(concentration)[::-1][:5]
    return stats, total_sum, top_brain_regions

@st.cache
def convert_array(array):
    return pd.DataFrame(array).to_csv().encode('utf-8')

if __name__ == '__main__':
    st.title('BrainSpread')
    st.header('Misfolded proteins spreading')
    
    st.subheader('Data loading')
    t0_concentration = upload_new_data(caption="Upload CSV file with regions concentration (shape: 1x116)")
    
    if t0_concentration is not None:
        t1_concentration_pred = run_simulation(t0_concentration)
    
        st.subheader('Results')
        
        stats, total_sum, top_brain_regions = calc_statistics(t1_concentration_pred)
        
        st.text('Statistics')
        st.code(stats)
        
        st.text('Total sum of concentration')
        st.code(total_sum)
        
        st.text('Brain region index with the highest MP concentration (top 5)')
        st.code(top_brain_regions)
        
        st.text('Visualization')
        visualize_concentration(t0_concentration, t1_concentration_pred)
        
        st.subheader('Download')
        st.text('Download CSV file with t1 predictions')

        st.download_button(
            "Press to Download",
            convert_array(t1_concentration_pred),
            "t1_pred.csv",
            "text/csv",
            key='download-csv'
            )
        
        