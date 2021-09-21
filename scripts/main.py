"""
The main file to train/test the GLAD-Meta algorithm.
Contains code to generate data, run training and the 
loss function.
"""
import numpy as np
import torch

import scripts.utils.prepare_data as prepare_data


def getGLADdata(num_nodes, sparsity, num_samples, batch_size=1):
    """Prepare true DAG adj matrices and then run a linear SEM 
    simulation to get the corresponding samples.
    
    Args:
        num_nodes (int): The number of nodes in DAG
        num_edges (int): The number of desired edges in DAG
        num_samples (int): The number of samples to simulate from DAG
        batch_size (int, optional): The number of batches
    
    Returns:
        Xb (torch.Tensor BxMxD): The sample data
        trueTheta (torch.Tensor BxDxD): The true underlying precision matrices
    """
    Xb, trueTheta = [], []
    for b in range(batch_size):
        # I - Getting the true edge connections
        edge_connections = prepare_data.generateRandomGraph(
            num_nodes, 
            sparsity
            )
        # II - Gettings samples from fitting a Gaussian distribution
        X, true_theta = prepare_data.simulateGaussianSamples(
            num_nodes,
            edge_connections,
            num_samples, 
            )
        # collect the batch data
        Xb.append(X)
        trueTheta.append(true_theta)
    # Converting the data to torch 
    Xb = prepare_data.convertToTorch(np.array(Xb), req_grad=False)
    trueTheta = prepare_data.convertToTorch(np.array(trueTheta), req_grad=False)

    print(f'TrueTheta:\n {trueTheta, trueTheta.shape}')
    print(f'Samples:\n {Xb, Xb.shape}')
    return Xb, trueTheta


def lossGLADmeta(theta, S):
    """The objective function of the graphical lasso which is 
    the loss function for the meta learning of glad
    loss-meta = 

    Args:
        theta (tensor 3D): precision matrix BxDxD
        S (tensor 3D): covariance matrix BxDxD (dim=D)
    
    Returns:
        loss (tensor 1D): the loss value of the obj function
    """
    B, D, _ = S.shape
    t1 = -1*torch.logdet(theta)
    # Batch Matrix multiplication: torch.bmm
    t21 = torch.einsum("bij, bjk -> bik", S, theta)
    # getting the trace (batch mode)
    t2 = torch.einsum('jii->j', t21) 
    meta_loss = torch.sum(t1+t2)/B # sum over the batch
    return meta_loss