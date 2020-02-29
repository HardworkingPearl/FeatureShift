from __future__ import print_function
import torch
import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence

# calculate average along the data vector
def cal_mean(data_train, data_test, step_size):
    # sample x channel x frequence x datapoint
    shape_train = data_train.shape
    shape_test = data_test.shape
    data_train_temp = np.zeros((shape_train[0],shape_train[1],shape_train[2],shape_train[3]//step_size))
    data_test_temp = np.zeros((shape_test[0],shape_test[1],shape_test[2],shape_test[3]//step_size))
    for i in range(shape_train[3]//step_size):
        data_train_temp[:,:,:,i] = np.mean(data_train[:,:,:,step_size*i:step_size*i+step_size-1])
    for i in range(shape_test[3]//step_size):
        data_test_temp[:,:,:,i] = np.mean(data_test[:,:,:,step_size*i:step_size*i+step_size-1])
    print("Data has been replaced with mean values!")
    return data_train_temp, data_test_temp
    
# Use this function to transfer the data into graph
def make_graph(data, MAX_DEGREE):
    # data:  channel x frequence x datapoint
    Adj_sample = []
    L_sample = []
    graph_sample = []
    chebyshev_sample = []
    for i in range(data.shape[0]):
            Adj = preprocess_adj(np.corrcoef(data[i,:,:]))
            L = generate_L(Adj)
            graph, chebyshev = generate_G(L, MAX_DEGREE, data[i,:,:])
            Adj_sample.append(Adj)
            L_sample.append(L)
            graph_sample.append(np.stack(graph, axis = 0))
            chebyshev_sample.append(np.stack(chebyshev, axis = 0))
    Adj_sample = torch.from_numpy(np.stack(Adj_sample, axis = 0)).float()
    L_sample = torch.from_numpy(np.stack(L_sample, axis = 0)).float()
    graph_sample = torch.from_numpy(np.stack(graph_sample, axis = 0)).float()
    chebyshev_sample = torch.from_numpy(np.stack(chebyshev_sample, axis = 0)).float()
    return Adj_sample, L_sample, graph_sample, chebyshev_sample
    
            
def normalize_adj(Adj): 
    Adj = np.where(Adj < 0, 0, Adj) # remove all the negative correlation
    Adj_other = Adj[Adj != 1]
    Min = np.min(Adj_other)
    Max = np.max(Adj_other)
    return np.where( Adj != 1, (Adj - Min)/(Max - Min), Adj)
# standard adjacent matrix
def preprocess_adj(Adj):
    Adj = np.where(Adj < 0, 0, Adj)
    Adj = np.where(Adj > 0, 1, Adj) #standard adjacent matrix
    D = np.diag(np.power(np.array(Adj.sum(1)), -0.5).flatten(), 0) # normalized degree matrix
    Adj_normalized = Adj.dot(D).transpose().dot(D)
    return Adj_normalized

def generate_L(Adj):
    L = normalized_laplacian(Adj)
    L_scaled = rescale_laplacian(L)
    return L_scaled

def generate_G(L, MAX_DEGREE, X):
    graph_feature = X 
    chebyshev_matrix = chebyshev_polynomial(L, MAX_DEGREE)
    return graph_feature, chebyshev_matrix

def normalized_laplacian(Adj):
    laplacian = np.eye(Adj.shape[0]) - Adj
    return laplacian

def rescale_laplacian(laplacian):
    try:
        #print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - np.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    #print("Calculating Chebyshev polynomials up to order {}...".format(k))
    X = np.asmatrix(X)
    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    T_k[0] = T_k[0].toarray()

    for item in range(len(T_k)):
        T_k[item] = np.asarray(T_k[item])
    return np.stack(T_k, axis = 0)


def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
    
def relative_power(data):
    # data: (channel x frequency x datapoint) type = np.array
    data_shape = data.shape
    data = np.power(data,2)
    data_list = []
    for i in range(data_shape[0]):
        power_sum = np.sum(data[i],axis = -1)
        power_sum = np.sum(power_sum, axis = -1)
        data_list.append(data[i]/power_sum)
    data = np.stack(data_list,axis = 0)
    return data

def normalize_data(data_train, data_test, type_scaler):
    # data: (sample x channel x frequency x datapoint)
    if type_scaler == 'Standard':
        scaler =  StandardScaler()
    elif type_scaler == 'MaxMin':
        scaler = MinMaxScaler()
    for i in range(data_train.shape[1]):
        for j in range(data_train.shape[2]):
            array_means = data_train[:,i,j,:].mean()
            array_std = data_train[:,i,j,:].std()
            data_train[:,i,j,:] = (data_train[:,i,j,:] - array_means)/array_std
            data_test[:,i,j,:] = (data_test[:,i,j,:] - array_means)/array_std
    print("Data has been normalized!")
    return data_train, data_test    