import torch
from typing import List, Tuple
from .linalg import cosine_sim

def topk_indices(scores: torch.Tensor, k: int):
    return torch.topk(scores, k=k).indices

def recommend_recipes(
    pantry_vec: torch.Tensor,         # (1, D)
    recipe_mat: torch.Tensor,         # (N, D)
    k: int = 5,
    difficulty_penalty: torch.Tensor = None  # (N,) optional
) -> Tuple[torch.Tensor, torch.Tensor]:
    sims = cosine_sim(pantry_vec, recipe_mat).squeeze(0)  # (N,)
    if difficulty_penalty is not None:
        sims = sims - 0.1 * difficulty_penalty
    idx = topk_indices(sims, k)
    return idx, sims[idx]

def missing_ingredients(pantry_set: set, recipe_ings: List[str]) -> List[str]:
    return [w for w in recipe_ings if w not in pantry_set]
