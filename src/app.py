import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

st.set_page_config(page_title="자취생의 냉장고 파먹기", page_icon="🥗", layout="centered")
st.title("자취생의 냉장고 파먹기")

# =========================
# 1) 내장 데이터 (메뉴/재료)
# =========================
CSV_TEXT = """title,ingredients
김치볶음밥,밥;김치;대파;계란;참기름
계란장조림,계란;간장;설탕;물;고추
참치마요덮밥,밥;참치캔;마요네즈;간장;대파
된장찌개,된장;두부;양파;감자;애호박;대파
오일파스타,스파게티면;올리브오일;마늘;소금;후추
삼겹살숙주볶음,삼겹살;숙주;간장;굴소스;후추
떡볶이(엽떡스타일),떡;고추장;고춧가루;설탕;어묵;대파
편의점조합(삼각김밥+컵라면),삼각김밥;컵라면

간장계란밥,밥;계란;간장;참기름
라면치즈계란,라면사리;라면스프;계란;슬라이스치즈;대파
떡라면,라면사리;라면스프;떡;대파
치즈계란토스트,식빵;계란;슬라이스치즈;버터;케첩
참치마요주먹밥,밥;참치캔;마요네즈;김;소금
비빔국수,소면;고추장;식초;설탕;참기름;오이;계란
토마토계란볶음,토마토;계란;대파;소금;후추;식용유
김치전,김치;부침가루;물;식용유
부침개(야채),부침가루;양파;당근;대파;물;식용유
어묵볶음,어묵;양파;당근;간장;설탕;물엿;마늘;식용유
스팸구이정식,스팸;밥;계란;김;김치
소시지야채볶음,소시지;양파;당근;피망;간장;설탕;후추;식용유
스팸마요덮밥,밥;스팸;마요네즈;간장;대파
김치참치볶음밥,밥;김치;참치캔;대파;참기름
핫도그(자취버전),핫도그빵;소시지;케첩;머스터드
콘치즈,옥수수콘;마요네즈;설탕;모짜렐라치즈;버터

제육볶음,돼지고기앞다리;양파;대파;고추장;간장;설탕;고춧가루;마늘;후추;식용유
감자계란볶음,감자;계란;양파;소금;후추;식용유
카레라이스,밥;카레가루;감자;양파;당근;식용유;물
부대찌개,소시지;스팸;김치;두부;양파;고추장;고춧가루;마늘;라면사리;대파;국물용멸치;물
순두부찌개,순두부;돼지고기다짐;양파;대파;고춧가루;간장;마늘;계란;물
국물떡볶이,떡;어묵;대파;고추장;간장;설탕;다시다;물
로제떡볶이,떡;어묵;우유;생크림;고추장;설탕;후추;버터

토마토파스타,스파게티면;토마토파스타소스;양파;마늘;올리브오일;소금;후추
크림베이컨파스타,스파게티면;베이컨;생크림;우유;양파;마늘;버터;소금;후추
페페론치노,스파게티면;올리브오일;마늘;페페론치노;소금

편의점(샐러드+닭가슴살),샐러드;훈제닭가슴살
편의점(햇반+참치+마요),즉석밥;참치캔;마요네즈;간장
편의점(컵국+즉석밥),컵국;즉석밥
컵라면+치즈+김,컵라면;슬라이스치즈;김
"""

df = pd.read_csv(StringIO(CSV_TEXT))
df["ing_list"] = df["ingredients"].apply(lambda s: [x.strip() for x in str(s).split(";") if x and x.strip()])

# 기본 재료 가중치(슬라이더 기본값에 사용)
DEFAULT_PANTRY = {
    "밥": 1.0, "즉석밥": 1.0, "소면": 1.0, "스파게티면": 1.0, "식빵": 1.0, "라면사리": 1.0,

    "계란": 1.0, "두부": 1.0, "순두부": 1.0, "삼겹살": 1.0, "돼지고기앞다리": 1.0,
    "소시지": 1.0, "스팸": 1.0, "베이컨": 1.0, "참치캔": 1.0, "훈제닭가슴살": 1.0,
  
    "양파": 1.0, "대파": 1.0, "감자": 1.0, "당근": 1.0, "애호박": 1.0, "피망": 1.0, "오이": 1.0,
    "김치": 1.0, "김": 1.0, "고추": 1.0, "샐러드": 1.0, "토마토": 1.0, "숙주": 1.0,

    "떡": 1.0, "어묵": 1.0,
 
    "우유": 1.0, "생크림": 1.0, "슬라이스치즈": 1.0, "모짜렐라치즈": 1.0, "버터": 1.0,
  
    "옥수수콘": 1.0, "핫도그빵": 1.0,

    "간장": 1.0, "고추장": 1.0, "고춧가루": 1.0, "된장": 1.0, "설탕": 1.0, "식초": 1.0,
    "참기름": 1.0, "굴소스": 1.0, "마요네즈": 1.0, "케첩": 1.0, "머스터드": 1.0,
    "후추": 1.0, "소금": 1.0, "마늘": 1.0, "다진마늘": 1.0, "올리브오일": 1.0, "식용유": 1.0,
    "라면스프": 1.0, "물엿": 1.0, "맛술": 1.0, "다시다": 1.0,

    "부침가루": 1.0, "밀가루": 1.0, "카레가루": 1.0,

    "토마토파스타소스": 1.0, "페페론치노": 1.0,

    "삼각김밥": 1.0, "컵라면": 1.0, "컵국": 1.0,

    "물": 1.0
}


# =========================
# 2) 유틸(벡터화/유사도/추천)
# =========================
def build_vocab(lists_of_tokens):
    vocab = {}
    for lst in lists_of_tokens:
        for tok in lst:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab  # {token: index}

def vectorize(list_of_lists, vocab, weights=None):
    M, D = len(list_of_lists), len(vocab)
    X = np.zeros((M, D), dtype=np.float32)
    for i, tokens in enumerate(list_of_lists):
        for t in tokens:
            j = vocab.get(t)
            if j is None:
                continue
            w = 1.0 if weights is None else float(weights.get(t, 1.0))
            X[i, j] += w
    return X

def cosine_sim(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return An @ Bn.T

def missing_ingredients(pantry_set, need_list):
    return [x for x in need_list if x not in pantry_set]

def recommend(U, R, k=5, penalty=None, penalty_strength=0.1):
    S = cosine_sim(U, R).reshape(-1)
    if penalty is not None:
        S = S - penalty_strength * penalty  # 부족 재료 개수만큼 점수 깎기
    idx = np.argsort(-S)[:k]
    return idx, S[idx]

# ==========================================
# 3) UI — 빈 상태 시작 + 실시간 재료 추가/해제
# ==========================================
st.subheader("내 재료 선택")

# 레시피에서 등장하는 재료 + 기본창고 재료 → 후보 풀
all_ings_base = sorted({w for lst in df["ing_list"] for w in lst} | set(DEFAULT_PANTRY.keys()))

# 세션 상태 초기화 (빈 상태 시작)
if "my_ings" not in st.session_state:
    st.session_state["my_ings"] = []           # 현재 선택된 나의 재료(빈 목록으로 시작)
if "all_ings" not in st.session_state:
    st.session_state["all_ings"] = list(all_ings_base)  # 후보 재료 목록

# ===================  빠른 입력/초기화 UI ===================
# 폼을 쓰면 입력칸 + 버튼이 같은 줄에서 보기 좋게 정렬,
# Enter 키로도 '추가'가 동작합니다.
with st.form("quick_add_form", clear_on_submit=False):
    c1, c2, c3 = st.columns([6, 2, 2], vertical_alignment="center")

    with c1:
        # 라벨 감추고 placeholder만 보이게 → 높이/정렬 깔끔하게...
        new_ing_text = st.text_input(
            label="새 재료 추가",
            placeholder="예) 베이컨, 치즈",
            label_visibility="collapsed"
        )

    with c2:
        add_clicked = st.form_submit_button("추가", use_container_width=True)

    with c3:
        clear_clicked = st.form_submit_button("모두 해제", use_container_width=True)

# 어떤 버튼을 눌렀는지에 따라 처리
if add_clicked and new_ing_text.strip():
    to_add = [x.strip() for x in new_ing_text.split(",") if x.strip()]
    st.session_state["all_ings"] = sorted(set(st.session_state["all_ings"]) | set(to_add))
    st.session_state["my_ings"] = sorted(set(st.session_state["my_ings"]) | set(to_add))
    st.rerun()

if clear_clicked:
    st.session_state["my_ings"] = []
    st.rerun()

# 멀티셀렉트(세션 상태와 연결)
selected_ings = st.multiselect(
    "보유 재료를 골라주세요",
    options=st.session_state["all_ings"],
    default=st.session_state["my_ings"],
    key="my_ings"
)

st.caption("가중치(수량/선호) 조절")
weights = {}
cols = st.columns(3)
for i, ing in enumerate(st.session_state["my_ings"]):
    with cols[i % 3]:
        weights[ing] = st.slider(ing, 0.0, 5.0, float(DEFAULT_PANTRY.get(ing, 1.0)), 0.1)

topk = st.slider("추천 개수 (Top-K)", 1, 10, 5)
use_penalty = st.checkbox("부족 재료 페널티 적용", value=True)
penalty_strength = st.slider("페널티 강도", 0.0, 0.5, 0.1, 0.05)

# =========================
# 4) 추천 계산 및 출력
# =========================
if len(st.session_state["my_ings"]) == 0:
    st.warning("재료를 하나 이상 선택하거나, 위 입력창에 새 재료를 추가해 주세요.")
else:
    vocab = build_vocab(df["ing_list"].tolist() + [st.session_state["my_ings"]])
    R = vectorize(df["ing_list"].tolist(), vocab)                               # (N, D)
    U = vectorize([st.session_state["my_ings"]], vocab, weights=weights)        # (1, D)

    pantry_set = set(st.session_state["my_ings"])
    penalty_vec = None
    if use_penalty:
        penalty_vec = np.array(
            [len(missing_ingredients(pantry_set, lst)) for lst in df["ing_list"]],
            dtype=np.float32
        )

    idx, scores = recommend(U, R, k=topk, penalty=penalty_vec, penalty_strength=penalty_strength)

    st.subheader(f"추천 레시피 Top-{topk}")
    for rank, (i, sc) in enumerate(zip(idx.tolist(), scores.tolist()), start=1):
        row = df.iloc[i]
        miss = missing_ingredients(pantry_set, row["ing_list"])
        st.markdown(f"**[{rank}] {row['title']}** — score: `{sc:.4f}`")
        st.caption("필요한 재료: " + ", ".join(row["ing_list"]))
        if miss:
            st.error("부족한 재료: " + ", ".join(miss))
        else:
            st.success("부족한 재료: 없음")
        st.divider()

# (선택) 디버그: 현재 후보/선택 재료 보기
with st.expander("재료 확인하기 !"):
    st.write("후보 재료:", ", ".join(st.session_state["all_ings"]))
    st.write("선택 재료:", ", ".join(st.session_state["my_ings"]))

# 설명
with st.expander("설명"):
    st.write(
        "- 텍스트 재료를 단어장으로 벡터화하고 코사인 유사도로 추천합니다.\n"
        "- 부족 재료 페널티로 만들기 쉬운 메뉴를 위로 올립니다.\n"
        "- 재료 가중치(수량/선호) 슬라이더로 랭킹이 즉시 반영됩니다.\n"
        "- 이 버전은 페이지(세션) 내에서 재료를 실시간 추가할 수 있습니다. (단,새로고침하면 초기화됩니다.)"
    )
