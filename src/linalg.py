import torch

def normalize_tensor(mat: torch.Tensor, dim: int):
    norm = torch.sqrt(torch.sum(mat * mat, dim=dim, keepdim=True))
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    return mat / norm

def cosine_sim(A: torch.Tensor, B: torch.Tensor):
    A_hat = normalize_tensor(A, dim=1)   # (n, d)
    B_hat = normalize_tensor(B, dim=1)   # (m, d)
    return A_hat @ B_hat.T               # (n, m)
