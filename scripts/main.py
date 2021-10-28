"""
The main file to train/test the mGLAD algorithm.
Contains code to generate data, run training and the 
loss function.
"""
import numpy as np
import torch

# Helper functions for mGLAD
from scripts.glad.glad_params import glad_params
from scripts.glad import glad
from scripts.utils.metrics import reportMetrics

import scripts.utils.prepare_data as prepare_data


#################### Functions to generate data ####################
def get_data(num_nodes, sparsity, num_samples, batch_size=1):
    """Prepare true DAG adj matrices and then run a linear SEM 
    simulation to get the corresponding samples.
    
    Args:
        num_nodes (int): The number of nodes in DAG
        num_edges (int): The number of desired edges in DAG
        num_samples (int): The number of samples to simulate from DAG
        batch_size (int, optional): The number of batches
    
    Returns:
        Xb (torch.Tensor BxMxD): The sample data
        trueTheta (torch.Tensor BxDxD): The true precision matrices
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
    return np.array(Xb), np.array(trueTheta)
######################################################################


#################### Functions to prepare model ######################
def init_mGLAD(theta_init_offset=1.0, nF=3, H=3):
    """Initialize the GLAD model parameters and the optimizer
    to be used.

    Args:
        theta_init_offset (float): Initialization diagonal offset 
            for the pred theta (adjust eigenvalue)
        nF (int): #input features for the entrywise thresholding
        H (int): The hidden layer size to be used for the NNs
    
    Returns:
        model: class object
        optimizer: class object
    """
    model = glad_params(
        theta_init_offset=theta_init_offset,
        nF=nF, 
        H=H
        )
    optimizer = glad.get_optimizers(model)
    return model, optimizer


def forward_mGLAD(Sb, model_glad):
    """Run the input through the meta-GLAD algorithm.
    It executes the following steps in batch mode
    1. Run the GLAD model to get initial good regularization
    2. Calculate the meta-loss
    
    Args:
        Sb (torch.Tensor BxDxD): The input covariance matrix
        metaGLADmodel (dict): Contains the learnable params
    
    Returns:
        predTheta (torch.Tensor BxDxD): The predicted theta
        loss (torch.scalar): The meta loss 
    """
    # 1. Running the GLAD model 
    predTheta = glad.glad(Sb, model_glad)
    # 2. Calculate the meta-loss
    loss = loss_mGLAD(predTheta, Sb)
    return predTheta, loss


def loss_mGLAD(theta, S):
    """The objective function of the graphical lasso which is 
    the loss function for the meta learning of glad
    loss-meta = 1/B(-log|theta| + <S, theta>)

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
    # print(t1, torch.det(theta), t2) 
    # regularization term 
    # tr = 1e-02 * torch.sum(torch.abs(theta))
    meta_loss = torch.sum(t1+t2)/B # sum over the batch
    return meta_loss 


def run_mGLAD(Xb, trueTheta=None, EPOCHS=250, VERBOSE=True):
    """Running the meta-GLAD algorithm.
    
    Args:
        Xb (torch.Tensor BxMxD): The input sample matrix
        trueTheta (torch.Tensor BxDxD): The corresponding 
            true graphs for reporting metrics.
        EPOCHS (int): The number of training epochs
        VERBOSE (bool): if True, prints to sys.out

    Returns:
        predTheta (torch.Tensor BxDxD): Predicted graphs
    """
    # Calculating the batch covariance
    Sb = prepare_data.getCovariance(Xb) # BxDxD
    # Converting the data to torch 
    Xb = prepare_data.convertToTorch(Xb, req_grad=False)
    Sb = prepare_data.convertToTorch(Sb, req_grad=False)
    trueTheta = prepare_data.convertToTorch(
        trueTheta,
        req_grad=False
        )
    B, M, D = Xb.shape
    # model and optimizer for mGLAD
    model_glad, optimizer_glad = init_mGLAD(
        theta_init_offset=1.0,
        nF=3,
        H=3
        )
    # Optimizing for the meta loss
    for e in range(EPOCHS):      
        # reset the grads to zero
        optimizer_glad.zero_grad()
        # calculate the loss
        predTheta, loss = forward_mGLAD(Sb, model_glad)
        # calculate the backward gradients
        loss.backward()
        if not e%25: print(f'epoch:{e}/{EPOCHS} loss:{loss.detach().numpy()}')
        # updating the optimizer params with the grads
        optimizer_glad.step()
#         print('theta_init_offset: ', model_glad.theta_init_offset)
        # reporting the metrics if true thetas provided
        if trueTheta is not None and (e+1)%EPOCHS == 0 and VERBOSE:
            for b in range(B):
                compare_theta = reportMetrics(
                    trueTheta[b].detach().numpy(), 
                    predTheta[b].detach().numpy()
                )
                print(f'Batch:{b} - {compare_theta}')
    return predTheta
######################################################################
