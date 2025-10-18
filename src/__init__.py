from pathlib import Path
import json
from typing import List
import pandas as pd
import streamlit as st
import torch

from src.vectorize import build_vocab, vectorize
from src.recommend import recommend_recipes, missing_ingredients

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

st.set_page_config(page_title="Fridge2Dish", page_icon="🥗", layout="centered")
st.title("🥗 자취생의 냉장고 파먹기 ! ")

# 데이터 로드
@st.cache_data
def load_data():
    recipes = pd.read_csv(DATA_DIR / "recipes.csv")
    recipes["ing_list"] = recipes["ingredients"].apply(lambda s: [x.strip() for x in s.split(";") if x.strip()])
    with open(DATA_DIR / "pantry.json", "r") as f:
        pantry_dict = json.load(f)
    return recipes, pantry_dict

recipes, pantry_dict = load_data()

# 전체 재료 후보
all_ings = sorted({w for lst in recipes["ing_list"] for w in lst} | set(pantry_dict.keys()))

st.subheader("내 재료")
st.caption("사이드바에서 재료 선택 및 가중치(수량/선호) 조절이 가능합니다.")

# ===== 사이드바 입력 =====
st.sidebar.header("재료 선택 & 가중치")
default_ings = list(pantry_dict.keys())
selected_ings = st.sidebar.multiselect("재료를 골라주세요", all_ings, default=default_ings)

weights = {}
for ing in selected_ings:
    weights[ing] = st.sidebar.slider(f"{ing}", 0.0, 5.0, float(pantry_dict.get(ing, 1.0)), 0.1)

topk = st.sidebar.slider("추천 개수 (Top-K)", 1, 10, 5)
use_missing_penalty = st.sidebar.checkbox("부족 재료 페널티 적용(현실적인 추천)", value=True)
penalty_strength = st.sidebar.slider("페널티 강도", 0.0, 0.5, 0.1, 0.05)

# 벡터화
vocab = build_vocab(recipes["ing_list"].tolist() + [selected_ings])
R = vectorize(recipes["ing_list"].tolist(), vocab)                    # (N, D)
U = vectorize([selected_ings], vocab, weights=weights or None)        # (1, D)

# 부족 재료 페널티
penalty_tensor = None
pantry_set = set(selected_ings)
if use_missing_penalty:
    penalty = []
    for lst in recipes["ing_list"]:
        miss = missing_ingredients(pantry_set, lst)
        penalty.append(len(miss))
    penalty_tensor = torch.tensor(penalty, dtype=torch.float32)

# 추천 
idx, scores = recommend_recipes(U, R, k=topk, difficulty_penalty=(penalty_strength * penalty_tensor) if penalty_tensor is not None else None)

st.subheader(f"추천 메뉴 Top-{topk}")
if len(selected_ings) == 0:
    st.warning("재료를 하나 이상 선택해주세요.")
else:
    for rank, (i, sc) in enumerate(zip(idx.tolist(), scores.tolist()), start=1):
        row = recipes.iloc[i]
        miss = missing_ingredients(pantry_set, row["ing_list"])
        with st.container():
            st.markdown(f"**[{rank}] {row['title']}** — score: `{sc:.4f}`")
            st.caption("필요한 재료: " + ", ".join(row["ing_list"]))
            if miss:
                st.error("부족한 재료: " + ", ".join(miss))
            else:
                st.success("부족한 재료: 없음")
            st.divider()

# 하단 도움말 
with st.expander("설명"):
    st.markdown(
        """
        - 텍스트 재료를 단어장으로 만들어 **벡터화**하고, **코사인 유사도**로 유사한 레시피를 추천합니다.  
        - **부족 재료 페널티**를 적용하면, 만들기 쉬운 메뉴가 위로 올라옵니다.  
        - 가중치 슬라이더(수량/선호)를 조절하면 랭킹이 즉시 변합니다.
        """
    )
