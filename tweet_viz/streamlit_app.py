import os
from pathlib import Path
import nltk


def _ensure_nlp_data():
    data_dir = Path("./nltk_data")
    os.environ["NLTK_DATA"] = str(data_dir.resolve())
    data_dir.mkdir(exist_ok=True)
    if str(data_dir) not in nltk.data.path:
        nltk.data.path.append(str(data_dir))

    # Install both old/new NLTK tagger names + common corpora TextBlob/NRCLex need
    required = [
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/brown", "brown"),
        ("corpora/words", "words"),
        ("corpora/movie_reviews", "movie_reviews"),
        ("corpora/subjectivity", "subjectivity"),
    ]
    for key, pkg in required:
        try:
            nltk.data.find(key)
        except LookupError:
            nltk.download(pkg, download_dir=str(data_dir), quiet=True)

    try:
        from textblob import download_corpora as _tb_dl
        _tb_dl.download_lite()  # or _tb_dl.download_all() if you prefer the full set
    except Exception:
        pass

_ensure_nlp_data()
import streamlit as st
import pandas as pd
from classification2_streamlit import get_profiles, best_emoji, nrc_emotions

# ---------- Config ----------
st.set_page_config(page_title="Emoji Suggester", layout="centered")
DATA_DIR = "archive"
TEXT_COL = "Text"

# ---------- Session State ----------
if "profiles" not in st.session_state:
    st.session_state.profiles = None
if "active_n_rows" not in st.session_state:
    st.session_state.active_n_rows = None
if "pending_n_rows" not in st.session_state:
    st.session_state.pending_n_rows = 1000

# ---------- Sidebar Controls ----------
st.sidebar.header("Setup")
st.sidebar.number_input(
    "Rows to learn",
    min_value=1,
    max_value=1_000_000,
    step=100,
    key="pending_n_rows",
)
build = st.sidebar.button("Build / Update profiles", type="primary", use_container_width=True)
top_k = st.sidebar.slider("How many emojis to return", min_value=1, max_value=10, value=1)
st.sidebar.caption(f"Active rows: {st.session_state.active_n_rows or '—'}")


# ---------- Cached Loader ----------
@st.cache_resource(show_spinner=True)
def load_or_build_profiles(_n_rows: int, _cache_path: Path):
    return get_profiles(Path(DATA_DIR), n_rows=_n_rows, text_col=TEXT_COL, cache_path=_cache_path)


def diagnose_data_dir(dir_path: Path) -> str:
    dir_path = Path(dir_path)
    lines = [f"path: {dir_path.resolve()}", f"exists: {dir_path.exists()}"]
    if dir_path.exists():
        files = sorted([p.name for p in dir_path.glob('*.csv')])
        lines.append(f"csv_count: {len(files)}")
        preview = files[:15]
        lines.extend([f"- {name}" for name in preview])
        if len(files) > len(preview):
            lines.append(f"... (+{len(files) - len(preview)} more)")
    return "\n".join(lines)


# ---------- Build Trigger ----------
if build:
    try:
        load_or_build_profiles.clear()
        cache_path = Path(f".cache/emoji_profiles_{int(st.session_state.pending_n_rows)}.pkl")
        st.session_state.profiles = load_or_build_profiles(int(st.session_state.pending_n_rows), cache_path)
        st.session_state.active_n_rows = int(st.session_state.pending_n_rows)
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Build failed: {e}")
        st.sidebar.markdown("**Data diagnostics**")
        st.sidebar.code(diagnose_data_dir(Path('archive')))

# ---------- Main UI ----------
st.title("Emoji Suggester")
st.caption("Input text → get the most fitting emoji")

user_text = st.text_area("Text", height=140, placeholder="Type or paste text...")
run = st.button("Suggest Emoji", type="primary", use_container_width=True)

profiles = st.session_state.profiles

if run:
    if profiles is None:
        st.error("Build profiles in the sidebar first.")
        st.stop()
    try:
        results = best_emoji(user_text, profiles, top_k=top_k, return_details=True)

        nrc = nrc_emotions(user_text)
        with st.expander("Emotions for input text"):
            df = pd.DataFrame(
                {
                    "emotion": list(nrc["freq"].keys()),
                    "frequency": list(nrc["freq"].values()),
                    "count": [nrc["raw"][k] for k in nrc["freq"].keys()],
                }
            ).sort_values("frequency", ascending=False, kind="stable")
            st.bar_chart(df.set_index("emotion")["frequency"])

        # Display top choices
        if isinstance(results, list) and results and isinstance(results[0], dict):
            emojis = " ".join([r["emoji"] for r in results])
        else:
            emojis = " ".join([emj for _, emj in results])
        st.markdown(f"### {emojis}")

        # Math display (cosine similarity and distance)
        st.markdown("Cosine distance:")
        st.latex(r"\cos(\theta)=\frac{\mathbf{x}\cdot\mathbf{y}}{\|\mathbf{x}\|\,\|\mathbf{y}\|}")

        if isinstance(results, list) and results and isinstance(results[0], dict):
            for r in results:
                st.subheader(r["emoji"])
                st.write(f"cos(θ) = {r.get('cos_sim', float('nan')):.6f}")
        else:
            st.info("Update classification2.best_emoji(..., return_details=True) to expose cosine components.")

    except Exception as e:
        st.error(f"Inference failed: {e}")
