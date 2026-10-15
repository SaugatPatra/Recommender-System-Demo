"""
app.py -- Streamlit demo.   Run:  streamlit run app.py
Pick a user (or pick songs you like), press Recommend, switch between the three models,
and read the evaluation table. The sidebar slider re-trains the distilled target model live
with a different distillation weight (alpha).
"""
import numpy as np, pandas as pd, streamlit as st
from train import load_artifacts, train_target, ALPHA
from metrics import evaluate
from data import GENRES

st.set_page_config(page_title="Cross-domain distillation demo", layout="wide")


@st.cache_resource
def _load():
    return load_artifacts()


world, models, soft_lab, extra = _load()
n_songs = int(world["n_songs"]); names = world["song_names"]; genre = world["song_genre"]
PLAIN, DIST, SRC = list(models.keys())

# ---------- sidebar ----------
st.sidebar.header("Settings")
alpha = st.sidebar.slider("Distillation weight α", 0.0, 1.0, float(ALPHA), 0.1,
    help="0 = learn only from real labels, 1 = learn only from the source model's soft labels")
if st.sidebar.button("Re-train distilled target model with this α"):
    with st.spinner("Training compact target model..."):
        m = train_target(world, soft_lab, extra, alpha)
    models[DIST] = m
    st.session_state["alpha_used"] = alpha
    st.session_state.pop("metrics", None)
st.sidebar.caption(f"Distilled model currently uses α = {st.session_state.get('alpha_used', ALPHA)}")
mode = st.sidebar.radio("Who are we recommending for?", ["Existing user", "Pick songs I like"])
view = st.sidebar.radio("Model", [PLAIN, DIST, SRC, "Compare all three side by side"])

st.title("🎵 Music recommendation with cross-domain knowledge distillation")
st.caption("Source domain: big video-platform log (300k interactions). Target domain: small music app (4k interactions).")


def top10(model, user_scores_fn, exclude):
    s = user_scores_fn(model).copy(); s[list(exclude)] = -np.inf
    return np.argsort(-s)[:10]


def table(idx, uid=None):
    rows = []
    for r, i in enumerate(idx, 1):
        row = {"#": r, "Song": names[i], "Genre": GENRES[genre[i]]}
        if uid is not None:                                  # compare with held-out real labels
            hit = np.where(world["test_items"][uid] == i)[0]
            row["Held-out result"] = "—" if len(hit) == 0 else ("✅ liked" if world["test_y"][uid][hit[0]] else "❌ not clicked")
        rows.append(row)
    return pd.DataFrame(rows).set_index("#")


# ---------- input ----------
if mode == "Existing user":
    uid = st.number_input("User ID", 0, int(world["n_users"]) - 1, 7)
    taste = world["user_taste"][uid]; fav = [GENRES[k] for k in np.argsort(-taste)[:2]]
    seen_mask = world["tgt_u"] == uid
    seen = set(world["tgt_i"][seen_mask].tolist())
    liked = world["tgt_i"][seen_mask & (world["tgt_y"] == 1)]
    st.write(f"**Hidden true taste (for the demo only):** {', '.join(fav)}.  "
             f"**Songs this user clicked in training data:** {', '.join(names[liked]) or 'none'}")
    scorer = lambda m: m.scores_for_user(uid, n_songs)
    uid_for_table = uid
else:
    options = list(range(n_songs))
    picks = st.multiselect("Pick 3 or more songs you like", options, format_func=lambda i: f"{names[i]} ({GENRES[genre[i]]})")
    seen = set(picks); uid_for_table = None
    scorer = lambda m: m.fold_in_user(picks, n_songs)

# ---------- recommend ----------
if st.button("Recommend", type="primary"):
    if mode != "Existing user" and len(picks) < 3:
        st.warning("Pick at least 3 songs first.")
    else:
        chosen = [PLAIN, DIST, SRC] if view.startswith("Compare") else [view]
        cols = st.columns(len(chosen))
        for col, name in zip(cols, chosen):
            with col:
                st.subheader(name)
                st.table(table(top10(models[name], scorer, seen), uid_for_table))

# ---------- evaluation ----------
st.divider()
st.header("Evaluation on the same held-out test set")
if "metrics" not in st.session_state:
    st.session_state["metrics"] = {n: evaluate(m.predict, world) for n, m in models.items()}
mdf = pd.DataFrame(st.session_state["metrics"]).T.round(3)
st.dataframe(mdf)
st.bar_chart(mdf[["AUC", "Precision@10", "Recall@10"]])
st.caption("AUC: how well clicked songs are scored above non-clicked ones (0.5 = random). "
           "Precision@10: share of the top-10 that were clicked. Recall@10: share of all clicked songs found in the top-10.")
