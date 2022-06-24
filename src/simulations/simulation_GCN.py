'''
    SYNOPSIS: python3 simulation_GCN.py <category> <matrix> <epochs>

'''

import logging
logging.getLogger('tensorflow').disabled = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from keras import metrics
from keras import backend as K

import json
from utils_vis import *
from utils import *

import numpy as np
from scipy.stats import pearsonr

from spektral.data import BatchLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from spektral.transforms import AdjToSpTensor, LayerPreprocess
from spektral.layers.convolutional.gcn_conv import GCNConv
from spektral.layers import ARMAConv
from spektral.data import Dataset, Graph

import matplotlib.pyplot as plt

import yaml
import sys
from datetime import datetime
from time import time
from prettytable import PrettyTable
import csv

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.get_logger().setLevel(logging.ERROR) # 3: filter INFO messages (leaves only WARNING and ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

date = str(datetime.now().strftime('%y-%m-%d_%H:%M:%S'))
logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO, force=True, filename = f"trace_GCN_{date}.log")
digits = 4
mode = ''

##### MODELS DEFINITION #####
class GraphNet(tf.keras.Model):
    def __init__(self, 
                 n_output_nodes,
                 l2_reg=1e-3,
                 dropout_rate=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.conv1 = GCNConv(32, activation='relu', kernel_regularizer=l2(l2_reg))
        self.conv2 = GCNConv(32, activation='relu', kernel_regularizer=l2(l2_reg))
        self.batchnorm = BatchNormalization()
        self.flatten = Flatten()
        self.drop1 = Dropout(dropout_rate)
        self.fc1 = Dense(32, activation='tanh')  
        self.fc2 = Dense(n_output_nodes, activation=self.custom_activation)
        
    @staticmethod
    def custom_activation(x):
        # try different activations
        return tf.keras.activations.linear(x)

    def call(self, inputs):
        ''' Inputs: 
        x: feature matrix with shape [batch_size, n_nodes, n_features]
        a: adjacency matrix with shape [batch_size, n_nodes, n_nodes] '''
        x, a = inputs
        x = self.conv1([x, a])
        x = self.batchnorm(x) 
        x = self.drop1(x)
        x = self.conv2([x, a])
        output = self.flatten(x)
        output = self.drop1(output)
        output = self.fc1(output)
        output = self.fc2(output)
        return output    
    
class ARMANet(tf.keras.Model):
    def __init__(self,
                 n_output_nodes,
                 droupout_skip=0.75,
                 dropout_rate=0.5,
                 l2_reg=5e-5,
                 **kwargs):
        super().__init__(**kwargs)
        self.gc_1 = ARMAConv(16, iterations=1, order=2, share_weights=True, 
                             dropout_rate=droupout_skip, activation='elu',
                             gcn_activation='elu', kernel_regularizer=l2(l2_reg))
        self.dropout = Dropout(dropout_rate)
        self.gc_2 = ARMAConv(16, iterations=1, order=2, share_weights=True, 
                             dropout_rate=droupout_skip, activation='relu',
                             gcn_activation='elu', kernel_regularizer=l2(l2_reg))
        self.flatten = Flatten()
        self.fc1 = Dense(n_output_nodes, activation='relu')
    
    def call(self, inputs):
        x, a = inputs
        x = self.gc_1([x, a])
        # x = self.dropout(x)
        x = self.gc_2([x, a])
        x = self.flatten(x)
        output = self.fc1(x)
        return output

#### DATASET DEFINITION #####
class ConnectomeDataset(Dataset):
    def __init__(self, file, dataset_folder, **kwargs):
        self.dataset_path_file = file
        self.max_rois = 166
        self.save_dir = dataset_folder
        self.paths = {}
        self.load_original_data()
        super().__init__(**kwargs)

    def load_original_data(self):
        with open(self.dataset_path_file, 'r') as f:
            self.paths = json.load(f)        
            
    def load_csv(self, path):
        data = np.genfromtxt(os.path.join('../..', path), delimiter=",")
        return data
    
    '''
    def drop_data_in_connect_matrix(self, cm, missing_labels=[35, 36, 81, 82]):
        index_to_remove = [(label - 1) for label in missing_labels]
        cm = np.delete(cm, index_to_remove, axis=0)
        cm = np.delete(cm, index_to_remove, axis=1) 
        return cm
    '''
    
    def save(self):
        # Create the directory
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # Write the data to file
        for subj_path, data in self.paths.items():
            subject = subj_path.split('/')[-2]
            node_init_vals = self.load_csv(data['baseline'])
            node_terminal_vals = self.load_csv(data['followup'])
            
            adjacency_matrix = np.identity(node_init_vals.shape[0])

            filename = os.path.join(self.save_dir, f'graph_{subject}')
            np.savez(filename, 
                     x=node_init_vals, 
                     a=adjacency_matrix, 
                     y=node_terminal_vals,
                     subject=subject)

    def generate_A(self, cm, t0_concentration):
        ''' Generate the matrix A from which starting the optimization (this is **A** in our paper). '''
        if matrix == 0: # CM (assuming it's diagonal has been set to zero by normalization)
            return np.copy(cm) + np.identity(cm.shape[0])
        elif matrix == 1: # Random
            return np.random.rand(cm.shape[0], cm.shape[1])
        elif matrix == 2: # diagonal
            return np.diag(t0_concentration)
        elif matrix == 3: # identity
            return np.identity(cm.shape[0])
    
    def read(self):
        ''' Return a list of Graph objects. 
        From documentation: 
        In general, node attributes should have shape (n_nodes, n_node_features) 
        and the adjacency matrix should have shape (n_nodes, n_nodes). 
        Graph-level labels can be either scalars or 1-dimensional arrays of shape [n_labels, ].
        Node-level labels can be 1-dimensional arrays of shape [n_nodes, ] 
        (representing a scalar label for each node), 
        or 2-dimensional arrays of shape [n_nodes, n_labels]. '''        
        output = []
        for subj in self.paths:
            t0_concentration = np.genfromtxt(self.paths[subj]['baseline'], delimiter=",") 
            t1_concentration = np.genfromtxt(self.paths[subj]['followup'], delimiter=",")
            A = self.generate_A(drop_data_in_connect_matrix(np.genfromtxt(self.paths[subj]['CM'], delimiter=",")), t0_concentration)
            output.append(
                Graph(
                    x=t0_concentration, 
                    a=A,  
                    y=t1_concentration, 
                    subject=subj
                )
            )

        return output

#### MODEL TRAINING #####

class NormalizeTargets:
    ''' Normalizes the target vector by scaling the values to range [0, 1]. '''

    def __call__(self, graph):
        graph.y = (graph.y - np.min(graph.y)) / (np.max(graph.y) - np.min(graph.y))
        return graph
    
class NormalizeNodes:
    ''' Normalizes the nodes attributes by scaling the values to range [0, 1]. '''
    
    def __call__(self, graph):
        graph.x = (graph.x - np.min(graph.x)) / (np.max(graph.x) - np.min(graph.x))
        return graph
    
class BinarizeMatrix:
    ''' Binarizes adjacency matrix. Required for AGNNConv layers. '''
    
    def __call__(self, graph):
        graph.a = np.where(graph.a > 0, 1, 0).astype(int)
        return graph 
        
class PearsonCoeff(metrics.Metric):
    ''' Custom metric: Pearson Correlation Coefficient. '''
    
    def __init__(self, name='pearsoncoeff', **kwargs):
        super(PearsonCoeff, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        x = y_true
        y = y_pred
        mx = K.mean(x, axis=0)
        my = K.mean(y, axis=0)
        if x == np.NaN or y == np.NaN:
            print(f'x : {x}')
            print(f'y: {y}')
        xm, ym = x - mx, y - my
        r_num = K.sum(xm * ym)
        x_square_sum = K.sum(xm * xm)
        y_square_sum = K.sum(ym * ym)
        r_den = K.sqrt(x_square_sum * y_square_sum)
        r = r_num / r_den
        self.pearson_val = K.mean(r)

    def result(self):
        return self.pearson_val    

def get_dataset(layer_type,
                dataset_folder, 
                to_sparse=False, 
                matrix_binarization=False, 
                matrix_fill=True,
                norm_nodes=False, 
                norm_targets=False):
    ''' Load and preprocess dataset. '''
    dataset = ConnectomeDataset(dataset_path, dataset_folder)
    if matrix_binarization: dataset.apply(BinarizeMatrix())
    dataset.apply(LayerPreprocess(layer_type)) # normalize the adjacency matrix 
    
    if norm_nodes: dataset.apply(NormalizeNodes()) # normalize the node attributes
    if norm_targets: dataset.apply(NormalizeTargets()) # normalize target vectors 
    if to_sparse: dataset.apply(AdjToSpTensor()) # Converts the adjacency matrix to a SparseTensor
    return dataset

def split_dataset(dataset):
    # split dataset into train, val and test sets
    idxs = np.arange(len(dataset))
    split_va, split_te = int(0.9 * len(dataset)), int(0.95 * len(dataset))
    idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
    
    dataset_tr = dataset[idx_tr]
    dataset_va = dataset[idx_va]
    dataset_te = dataset[idx_te]
          
    return dataset_tr, dataset_va, dataset_te

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def create_model(dataset, layer_type, learning_rate=1e-3):
    if layer_type == GCNConv:
        model = GraphNet(n_output_nodes=dataset[0].n_nodes)
    elif layer_type == ARMAConv:
        model = ARMANet(n_output_nodes=dataset[0].n_nodes)

    # disclaimer: loss = mse + regularization, thus mse loss != mse metric 
    model.compile(loss='mse', 
                  optimizer=Adam(learning_rate),
                  metrics=[metrics.MeanSquaredError(), PearsonCoeff()])
    return model

def train(dataset_train, model, dataset_val=None, patience=100, 
          is_summary=False, use_lr_scheduler=False):
    loader_tr = BatchLoader(dataset_train, batch_size=8, shuffle=False)
    if dataset_val is not None:
        loader_val = BatchLoader(dataset_val, batch_size=8, shuffle=False)
        validation_data = loader_val.load()
        validation_steps = loader_val.steps_per_epoch
    else:
        validation_data = None
        validation_steps = None

    if use_lr_scheduler:
        def __scheduler(epoch, lr):
            if epoch < 80:
                return lr
            else:
                return lr * np.exp(5e-3)
        
        callbacks = [LearningRateScheduler(__scheduler)]
    else:
        callbacks = None
        
    model.fit(
        loader_tr.load(),
        steps_per_epoch=loader_tr.steps_per_epoch,
        epochs=epochs,
        validation_data=validation_data,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=0
        )   
        
    if is_summary: model.summary()
    
    return model

def evaluate(dataset, model):
    loader = BatchLoader(dataset, batch_size=8)

    eval_results = model.evaluate(
            loader.load(),
            steps=loader.steps_per_epoch)

    return eval_results

def predict(dataset, model):
    loader = BatchLoader(dataset, batch_size=8, shuffle=False)

    predictions = model.predict(
        loader.load(),
        steps=loader.steps_per_epoch
    )
    
    return predictions

def calc_corr(y, y_pred):
    corr, _ = pearsonr(y, y_pred)
    return np.round(corr, 2)

def cross_validation(dataset, layer_type, show_test_pred=True, n_splits=None):
    if n_splits is None:
        # LOOCV - leave one out cross validation  
        n_splits = dataset.n_graphs
    logging.info(f'Number of splits in cross validation: {n_splits}')
    folds = KFold(n_splits, shuffle=True)
    eval_results = []
    counter = 0
    for train_index, test_index in folds.split(dataset):
        print(f"Fold: {counter}")
        logging.info(f"Fold: {counter}")
        dataset_train, dataset_test = dataset[train_index], dataset[test_index]
        model = create_model(dataset_train, layer_type)
        model = train(dataset_train, model)
        test_metrics = evaluate(dataset_test, model)
        eval_results.append(test_metrics)
        if show_test_pred:
            test_prediction(dataset_test, model)
        counter += 1
    return eval_results
    
def test_prediction(dataset_test, model):
    single_sample = dataset_test[:1]
    y_test_pred = predict(single_sample, model)[0] 
    y_test = single_sample[0]['y']
    x_test = single_sample[0]['x']
    subj = single_sample[0]['subject']
    pcc = calc_corr(y_test, y_test_pred) 
    mse = mean_absolute_error(y_test, y_test_pred)
    #pt_test.add_row([subj, round(mse, digits), round(pcc, digits)])
    #save_test_predictions(y_test, y_test_pred, subj)
    save_prediction_plot(x_test, y_test_pred, y_test, subj, subj +'test/GCN_test_' + date + '.png', mse, pcc)
    reg_err = np.abs(y_test - y_test_pred)
    save_coeff_matrix(subj + 'test/GCN_test_regions_' + date + '.csv', reg_err)
    total_mse.append(mse)
    total_pcc.append(pcc)
    total_reg_err.append(reg_err)
    test_scores[subj] = [mse, pcc]
    #visualize_test_sample(x_test, y_test, y_test_pred, subj, mse, corr)

'''
def visualize_test_sample(x_test, y_test, y_pred, subj_name, mse=None, corr_coef=None):
    plt.figure(figsize=(15, 7))
    plt.plot(x_test, marker='.', label='test baseline')
    plt.plot(y_test, marker='.', label='test followup')
    plt.plot(y_pred, marker='.', label='pred followup')
    plt.xlabel('ROI')
    plt.ylabel('normalized MP concentration')
    if (corr_coef is not None) and (mse is not None): 
        plt.title(f'GCNConv model predictions for {subj_name}\nMSE: {mse:.2f}\nCorr coef: {corr_coef}')
    else:
        plt.title(f'GCNConv model predictions for {subj_name}')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(config.plot_save_folder, f'{subj_name}.png'))
'''
'''
def save_test_scores(mse, pcc, save_file=config.scores_filename):
    with open(save_file, 'a+') as f:
        writer = csv.writer(f, escapechar=' ', quoting=csv.QUOTE_NONE)
        if os.stat(save_file).st_size == 0:
            # if file is empty, write a header
            writer.writerow(['MSE,PCC'])
        writer.writerow([mse, pcc])
'''

'''     
def save_test_predictions(y_test, y_test_pred, subj):
    # Save ground-truth and predicted followup values for each subject in order to make analysis on population. 
    # Note: Object of type 'float32' is not JSON serializable, thus convertion to 'float64' is needed. 
    
    data = {
        'true': list(y_test.astype('float64')), 
        'pred': list(y_test_pred.astype('float64'))
    }
    with open(os.path.join(subj + 'test/GCN_test_regions_' + date + '.csv', f'{subj}.json'), 'w') as outfile:
        json.dump(data, outfile, indent=4)
'''  
    
if __name__ == '__main__':
    with open('../../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.chdir(os.getcwd()+'/../../..')
    category = sys.argv[1] if len(sys.argv) > 1 else ''
    while category == '':
        try:
            category = input('Insert the category [ALL, AD, LMCI, MCI, EMCI, CN; default ALL]: ')
        except Exception as e:
            print("Using default")
            category = 'ALL'
        category = 'ALL' if category == '' else category
    
    matrix = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    while matrix < 0 or matrix > 3:
        try:
            matrix = int(input('Choose the initial A matrix [0=CM, 1=Rnd, 2=Diag, 3=Identity default 0]: '))
        except Exception as e:
            print("Using default")
            matrix = 0
            
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    while epochs < 1:
        try:
            epochs = int(input('Choose the number of epochs for training [default 10000]: '))
        except Exception as e:
            print("Using default")
            epochs = 10000
    
    dataset_path = config['paths']['dataset_dir'] +  f'datasets/dataset_{category}.json'
    output_mat = config['paths']['dataset_dir'] + f'simulations/{category}/matrices/'
    output_res = config['paths']['dataset_dir'] + f'simulations/{category}/results/'
    if not os.path.exists(output_mat):
        os.makedirs(output_mat)
    if not os.path.exists(output_res):
        os.makedirs(output_res)
        
    layer_type = GCNConv
    dataset = get_dataset(layer_type, config['paths']['dataset_dir'])
    #dataset.save()
    logging.info(f'Number of graphs in dataset: {len(dataset.paths.keys())}')
    
    ### TRAINING ###
    training_scores = {}
    test_scores = {}
    total_mse = []
    total_pcc = []
    total_reg_err = []
    
    total_time = time()
    test_results = cross_validation(dataset, layer_type)
    total_time = time() - total_time
    
    ### OUTPUT ###
    
    avg_reg_err = np.mean(total_reg_err, axis=0)
    avg_reg_err_filename = output_res+f'GCN_region_{date}.png'
    save_avg_regional_errors(avg_reg_err, avg_reg_err_filename)
    np.savetxt(f"{output_mat}GCN_{category}_regions_{date}.csv", np.mean(np.array(total_reg_err), axis=0), delimiter=',')
    
    pt_avg = PrettyTable()
    pt_avg.field_names = ["Avg MSE test ", "SD MSE test", "Avg Pearson test", "SD Pearson test"]
    pt_avg.add_row([round(np.mean(total_mse), digits), round(np.std(total_mse), 2), round(np.mean(total_pcc), digits), round(np.std(total_pcc), 2)])

    pt_test = PrettyTable()
    pt_test.field_names = ["ID", "Avg MSE test", "Avg Pearson test"]
    pt_test.sortby = "ID"

    for s in test_scores:
        mse_subj = [test_scores[s][0]]
        pcc_subj = [test_scores[s][1]]
        pt_test.add_row([s, round(np.mean(mse_subj), digits), round(np.mean(pcc_subj), digits)])
    
    filename = f"{output_res}GCN_{date}.txt"
    out_file = open(filename, 'w')
    out_file.write(f"Category: {category}\n")
    out_file.write(f"Subjects: {len(dataset.paths.keys())}\n")
    out_file.write(f"Matrix: {'CM' if matrix==0 else 'R' if matrix==1 else 'D' if matrix==2 else 'I' if matrix==3 else 'unknown'}\n")
    out_file.write(f"Total time (s): {format(total_time, '.2f')}\n")
    out_file.write(pt_avg.get_string() + '\n')
    out_file.write(pt_test.get_string() + '\n')
    out_file.close()
    logging.info('***********************')
    logging.info(f"Category: {category}")
    logging.info(f"Subjects: {len(dataset.paths.keys())}")
    logging.info(f"Matrix: {'CM' if matrix==0 else 'R' if matrix==1 else 'D' if matrix==2 else 'I' if matrix==3 else 'unknown'}")
    logging.info(f"Total time (s): {format(total_time, '.2f')}")
    logging.info('***********************')
    logging.info(f"Results saved in {filename}")
    print(f"Results saved in {filename}")