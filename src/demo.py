import json
import pandas as pd
from pathlib import Path
from typing import List
import torch

from src.vectorize import build_vocab, vectorize         
from src.recommend import recommend_recipes, missing_ingredients

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def split_ing(s: str) -> List[str]:
    return [x.strip() for x in s.split(";") if x.strip()]

def main():
    recipes = pd.read_csv(DATA_DIR / "recipes.csv")
    recipes["ing_list"] = recipes["ingredients"].apply(split_ing)

    with open(DATA_DIR / "pantry.json", "r") as f:
        pantry_dict = json.load(f)
    pantry_list = list(pantry_dict.keys())

    vocab = build_vocab(recipes["ing_list"].tolist() + [pantry_list])
    R = vectorize(recipes["ing_list"].tolist(), vocab)        
    U = vectorize([pantry_list], vocab, weights=pantry_dict)    

    # 추천
    k = 5
    idx, scores = recommend_recipes(U, R, k=k)

    # 결과 출력
    pantry_set = set(pantry_list)
    print("== 내 재료 ==")
    print(sorted(pantry_list))
    print("\n== 추천 레시피 Top-{} ==".format(k))
    for rank, (i, sc) in enumerate(zip(idx.tolist(), scores.tolist()), start=1):
        row = recipes.iloc[i]
        miss = missing_ingredients(pantry_set, row["ing_list"])
        print(f"[{rank}] {row['title']}  (score: {sc:.4f})")
        print(f"    필요한 재료: {', '.join(row['ing_list'])}")
        print(f"    부족한 재료: {', '.join(miss) if miss else '없음'}")

if __name__ == "__main__":
    main()
