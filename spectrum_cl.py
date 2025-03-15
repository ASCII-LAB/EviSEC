import numpy as np
import torch

def compute_laplacian(A):
    device = A.device
    D = torch.diag(torch.sum(A, axis=1))
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.diag(D))).to(device)
    L = torch.eye(A.shape[0], device=device) - torch.matmul(torch.matmul(D_inv_sqrt, A), D_inv_sqrt)
    
    return L

def filter_laplacian_by_ratio(A, ratio, partial=0):

    A = A.float()
    L = compute_laplacian(A)
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    
    k = int(ratio * len(eigenvalues))  
    V_low = eigenvectors[:, :k]
    Lambda_low = torch.diag(eigenvalues[:k])
    # print(f"Lambda_low:{Lambda_low}")    

    V_high = eigenvectors[:, k:]
    Lambda_high = torch.diag(eigenvalues[k:])
    # print(f"Lambda_high:{Lambda_high}")

    # V_mixed_low = torch.cat((V_low, V_high[:]))
    A_low = torch.matmul(V_low, torch.matmul(Lambda_low, V_low.T)) 
    A_high = torch.matmul(V_high, torch.matmul(Lambda_high, V_high.T)) 
    
    return A_low, A_high
