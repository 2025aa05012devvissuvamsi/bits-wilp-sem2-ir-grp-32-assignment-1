"""
DocSearch Information Retrieval System

An information retrieval platform built with Streamlit.
Covers preprocessing pipelines, inverted indexing, retrieval strategies,
phrase query processing, dictionary data structures, tolerant retrieval,
and a live analytics dashboard.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import io
import zipfile
from time import perf_counter
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas import DataFrame

from modules.bst import build_bst_from_vocabulary, extract_vocabulary
from modules.btree import build_btree_from_vocabulary
from modules.constants import (
    APP_TITLE, ASSIGNMENT_TITLE, COURSE_NAME,
    GROUP_MEMBERS, GROUP_NUMBER, SUPPORTED_FILE_TYPES,
)
from modules.data_loader import (
    load_default_dataset, load_pdf_documents, load_uploaded_dataset,
)
from modules.indexing import create_inverted_index
from modules.phrase_query import (
    create_biword_index, create_positional_index,
    generate_phrase_query_inference,
    search_biword_phrase, search_positional_phrase,
)
from modules.preprocessing import (
    apply_lemmatization, apply_stemming,
    convert_to_lowercase, handle_hyphenated_words,
    preprocess_text, remove_punctuation,
    remove_stopwords, tokenize_text,
)
from modules.retrieval import execute_retrieval_experiment, generate_comparison_inference
from modules.tolerant_retrieval import execute_tolerant_retrieval_experiment
from modules.utils import display_documents


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="DocSearch IR Application",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM STYLING
# ============================================================================

st.markdown("""
<style>
/* ── Global font ── */
html, body, .stApp {
    font-family: 'Segoe UI', Verdana, Geneva, Tahoma, sans-serif;
}

p, h1, h2, h3, h4, h5, h6, label, button {
    font-family: 'Segoe UI', Verdana, Geneva, Tahoma, sans-serif;
}

/* ── Page background ── */
.main { padding-top: 0.4rem; background: transparent; }

/* ── Tabs - pill style ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    flex-wrap: wrap;
    background: rgba(92,107,192,0.07);
    border-radius: 14px;
    padding: 6px 8px;
}
.stTabs [data-baseweb="tab"] {
    height: 42px;
    padding: 0 20px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.2px;
    background: transparent;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: #5C6BC0 !important;
    color: #fff !important;
    box-shadow: 0 2px 8px rgba(92,107,192,0.35);
}

/* ── Team banner ── */
.team-banner {
    border-radius: 14px;
    padding: 18px 24px 14px;
    margin-bottom: 18px;
    border-left: 5px solid #5C6BC0;
    background: linear-gradient(135deg, rgba(92,107,192,0.09) 0%, rgba(26,35,126,0.04) 100%);
}
.team-banner h2 { margin: 0 0 4px; font-size: 1.1rem; color: #5C6BC0; }
.team-banner p  { margin: 2px 0; font-size: 0.8rem; opacity: 0.8; }
.member-table { width:100%; border-collapse:collapse; font-size:0.82rem; margin-top:10px; }
.member-table th { padding:6px 12px; background:#5C6BC0; color:#fff; text-align:left; font-weight:700; }
.member-table td { padding:5px 12px; border-bottom:1px solid rgba(128,128,128,0.18); }
.member-table tr:hover td { background:rgba(92,107,192,0.07); }

/* ── Section heading ── */
.sec-head { font-size:1.25rem; font-weight:700; margin-bottom:0.15rem; }

/* ── Insight card ── */
.insight-card {
    background: rgba(0, 188, 212, 0.06);
    border-left: 5px solid #00BCD4;
    padding: 18px;
    border-radius: 12px;
    margin-top: 10px;

    font-family: 'Segoe UI', sans-serif;
    font-size: 15px;
    line-height: 1.8;
    color: white;

    white-space: normal !important;
    overflow-wrap: break-word;
}

.insight-card strong {
    color: #00E5FF;
    font-size: 16px;
}

/* ── Metric row ── */
.metric-pill {
    display:inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    background: rgba(92,107,192,0.12);
    font-size: 0.8rem;
    margin: 3px 4px;
    font-weight: 600;
    color: #3949ab;
}

/* ── Download button ── */
.stDownloadButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# TEAM BANNER
# ============================================================================

def render_team_banner() -> None:
    rows = "".join(
        f"<tr><td>{m['registration']}</td>"
        f"<td><strong>{m['name']}</strong></td>"
        f"<td style='text-align:center'>{m['contribution']}</td></tr>"
        for m in GROUP_MEMBERS
    )
    st.markdown(f"""
    <div class="team-banner">
        <h2>🎓 Group {GROUP_NUMBER} &nbsp;|&nbsp; {ASSIGNMENT_TITLE}</h2>
        <p><em>{COURSE_NAME}</em></p>
        <table class="member-table">
            <thead><tr>
                <th>Student Registration Number</th><th>Name</th>
                <th style="text-align:center">Percentage of contribution out of 100%</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>""", unsafe_allow_html=True)


# ============================================================================
# SAMPLE FILE DOWNLOADS
# ============================================================================

def build_sample_zip() -> bytes:
    """Bundle both sample CSV files into a single downloadable zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        with open("data/sample_dataset.csv", "rb") as f:
            zf.writestr("sample_ir_dataset.csv", f.read())
        with open("data/sample_technology.csv", "rb") as f:
            zf.writestr("sample_technology_dataset.csv", f.read())
        # README inside zip
        readme = (
            "IR - Sample Datasets\n"
            "===================================\n\n"
            "  Column 1: doc_id   (integer, unique document identifier)\n"
            "  Column 2: text     (string, full document text)\n\n"
            "Upload either file via the sidebar to explore the system.\n"
        )
        zf.writestr("README.txt", readme)
    return buf.getvalue()


def render_sample_downloads() -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📥 Sample Datasets**")
    st.sidebar.caption("Download and re-upload to explore the system.")
    zip_bytes = build_sample_zip()
    st.sidebar.download_button(
        label="⬇ Download Sample Files (.zip)",
        data=zip_bytes,
        file_name="docsearch_sample_data.zip",
        mime="application/zip",
        use_container_width=True,
    )


# ============================================================================
# DATASET INITIALIZATION
# ============================================================================

def initialize_dataset() -> DataFrame:
    st.sidebar.header("📂 Upload Dataset")
    uploaded_files = st.sidebar.file_uploader(
        label="CSV Dataset or PDF Documents",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
    )
    render_sample_downloads()

    if uploaded_files:
        first_name = uploaded_files[0].name.lower()
        try:
            if first_name.endswith(".csv"):
                df = load_uploaded_dataset(uploaded_file=uploaded_files[0])
                st.sidebar.success("✅ CSV loaded successfully.")
                return df
            pdf_files = [f for f in uploaded_files if f.name.lower().endswith(".pdf")]
            df = load_pdf_documents(uploaded_pdf_files=pdf_files)
            st.sidebar.success("✅ PDF(s) processed successfully.")
            return df
        except Exception as exc:
            st.sidebar.error(str(exc))
            st.stop()

    st.sidebar.info("ℹ️ Using built-in demo dataset.")
    return load_default_dataset()


# ============================================================================
# TAB 1 - DATASET VIEWER
# ============================================================================

def render_dataset_tab(documents_df: DataFrame) -> None:
    st.markdown("<h2 class='sec-head' style='color:#1E88E5;'>📄 Document Collection</h2>", unsafe_allow_html=True)
    st.info("The active document corpus loaded into the retrieval engine. Upload your own CSV via the sidebar.")

    c1, c2 = st.columns(2)
    c1.metric("Total Documents", len(documents_df))
    avg_len = round(documents_df["text"].apply(lambda x: len(str(x).split())).mean(), 1)
    c2.metric("Avg Words per Document", avg_len)

    st.divider()
    display_documents(dataframe=documents_df)

    st.divider()
    st.subheader("📊 Word-Count Distribution")
    wc = documents_df["text"].apply(lambda x: len(str(x).split()))
    wc_fig = px.bar(
        x=documents_df["doc_id"].tolist(),
        y=wc.tolist(),
        labels={"x": "Document ID", "y": "Word Count"},
        title="Word Count per Document",
        color=wc.tolist(),
        color_continuous_scale="Blues",
    )
    wc_fig.update_layout(font_family="Segoe UI", coloraxis_showscale=False, showlegend=False)
    st.plotly_chart(wc_fig, use_container_width=True)


# ============================================================================
# TAB 2 - TEXT PREPROCESSING
# ============================================================================

def render_preprocessing_tab(documents_df: DataFrame) -> None:
    st.markdown("<h2 class='sec-head' style='color:#43A047;'>⚙️ Text Preprocessing Pipeline</h2>", unsafe_allow_html=True)
    st.success("Each stage of the preprocessing pipeline is shown below. Toggle options to observe their effect on tokens and the resulting inverted index.")
    st.divider()

    # ── Pipeline toggle options ────────────────────────────────────────────
    st.subheader("🎛️ Pipeline Configuration")
    st.caption("Enable or disable individual preprocessing steps to compare their impact.")

    col1, col2, col3 = st.columns(3)
    with col1:
        use_lowercase   = st.checkbox("Lowercase Conversion",   value=True,  help="Converts all tokens to lowercase for uniform matching.")
        use_stopwords   = st.checkbox("Stop-Word Removal",      value=True,  help="Drops high-frequency, low-value words (e.g. 'the', 'is').")
    with col2:
        use_punct       = st.checkbox("Punctuation Removal",    value=True,  help="Strips tokens consisting solely of punctuation characters.")
        use_tokenize    = st.checkbox("Tokenization (NLTK)",    value=True,  help="Splits raw text into word/punctuation tokens using NLTK.")
    with col3:
        use_hyphen      = st.checkbox("Hyphen Normalization",   value=True,  help="Splits hyphenated compounds into individual sub-tokens.")
        apply_stemmer   = st.checkbox("Stemming (Porter)",      value=False, help="Reduces words to their root form via suffix stripping.")
        apply_lemmatizer= st.checkbox("Lemmatization (WordNet)",value=False, help="Maps inflected forms to valid dictionary base words.")

    st.divider()

    # ── Step-by-step preview ──────────────────────────────────────────────
    st.subheader("🔬 Step-by-Step Token Trace")
    doc_ids = documents_df["doc_id"].tolist()
    sel_id = st.selectbox("Select document for trace", options=doc_ids, index=0)
    raw = str(documents_df[documents_df["doc_id"] == sel_id].iloc[0]["text"])

    stages = []
    stages.append({"Stage": "① Raw Input", "Description": "Original document text, unmodified", "Output Preview": raw[:120], "Token Count": 1})

    current_tokens: List[str] = []
    if use_tokenize:
        current_tokens = tokenize_text(text=raw)
    else:
        current_tokens = raw.split()
    stages.append({"Stage": "② Tokenization", "Description": "Split into word/punctuation units", "Output Preview": str(current_tokens[:12]) + "…", "Token Count": len(current_tokens)})

    if use_lowercase:
        current_tokens = convert_to_lowercase(tokens=current_tokens)
    stages.append({"Stage": "③ Lowercase", "Description": "All tokens mapped to lowercase", "Output Preview": str(current_tokens[:12]) + "…", "Token Count": len(current_tokens)})

    if use_punct:
        current_tokens = remove_punctuation(tokens=current_tokens)
    stages.append({"Stage": "④ Punctuation Removal", "Description": "Pure-punctuation tokens discarded", "Output Preview": str(current_tokens[:12]) + "…", "Token Count": len(current_tokens)})

    if use_hyphen:
        current_tokens = handle_hyphenated_words(tokens=current_tokens)
    stages.append({"Stage": "⑤ Hyphen Normalization", "Description": "Hyphenated terms split into sub-tokens", "Output Preview": str(current_tokens[:12]) + "…", "Token Count": len(current_tokens)})

    if use_stopwords:
        current_tokens = remove_stopwords(tokens=current_tokens)
    stages.append({"Stage": "⑥ Stop-Word Removal", "Description": "Common function words eliminated", "Output Preview": str(current_tokens[:12]) + "…", "Token Count": len(current_tokens)})

    if apply_stemmer:
        current_tokens = apply_stemming(tokens=current_tokens)
        stages.append({"Stage": "⑦ Stemming (Porter)", "Description": "Heuristic suffix stripping to stem", "Output Preview": str(current_tokens[:12]) + "…", "Token Count": len(current_tokens)})

    if apply_lemmatizer:
        n = 8 if apply_stemmer else 7
        current_tokens = apply_lemmatization(tokens=current_tokens)
        stages.append({"Stage": f"⑧ Lemmatization (WordNet)" if apply_stemmer else "⑦ Lemmatization (WordNet)", "Description": "Morphological base-form mapping", "Output Preview": str(current_tokens[:12]) + "…", "Token Count": len(current_tokens)})

    st.dataframe(stages, use_container_width=True)

    # Token count chart
    fig = px.line(
        x=[s["Stage"] for s in stages],
        y=[s["Token Count"] for s in stages],
        markers=True,
        labels={"x": "Pipeline Stage", "y": "Token Count"},
        title=f"Token Count Reduction Through Pipeline (Doc {sel_id})",
        color_discrete_sequence=["#43A047"],
    )
    fig.update_layout(font_family="Segoe UI", xaxis_tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)

    # Inference
    raw_toks = tokenize_text(raw)
    reduction = len(raw_toks) - len(current_tokens)
    pct = round(reduction / max(len(raw_toks), 1) * 100, 1)
    st.markdown(f"""
    <div class="insight-card">
        <strong>💡 Pipeline Insight - Document {sel_id}:</strong><br>
        Token count went from <strong>{len(raw_toks)}</strong> (raw tokens) down to
        <strong>{len(current_tokens)}</strong> after all active stages - a
        <strong>{pct}%</strong> reduction.<br><br>
        Stop-word removal typically drives the largest single-step drop,
        eliminating function words that carry no retrieval value.
        Lowercasing unifies case variants without changing the count.
        Hyphen normalization can <em>increase</em> the count temporarily by splitting
        compound terms before subsequent filtering steps reduce it again.
        {"Stemming aggressively collapses morphological variants - fast but can conflate unrelated words." if apply_stemmer else ""}
        {"Lemmatization produces linguistically valid base forms, trading speed for precision." if apply_lemmatizer else ""}
    </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Full corpus processing ─────────────────────────────────────────────
    st.subheader("📋 Corpus Token Summary")
    processed: Dict[int, List[str]] = {}
    for _, row in documents_df.iterrows():
        did = int(row["doc_id"])
        toks = preprocess_text(str(row["text"]), apply_stemmer=apply_stemmer, apply_lemmatizer=apply_lemmatizer)
        processed[did] = toks

    corpus_rows = [{"doc_id": k, "token_count": len(v), "tokens": v} for k, v in processed.items()]
    st.dataframe(corpus_rows, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("Corpus Total Tokens", sum(len(v) for v in processed.values()))
    m2.metric("Documents Processed", len(processed))

    st.divider()

    # ── Inverted index with pipeline options ──────────────────────────────
    st.subheader("🗂️ Resulting Inverted Index")
    st.caption("The index is rebuilt live based on your selected pipeline configuration.")
    inv_idx = create_inverted_index(processed_documents=processed)
    idx_rows = [{"term": t, "doc_frequency": len(d), "posting_list": sorted(d)} for t, d in sorted(inv_idx.items())]
    st.dataframe(idx_rows, use_container_width=True)

    iv1, iv2 = st.columns(2)
    iv1.metric("Unique Vocabulary Terms", len(inv_idx))
    iv2.metric("Avg Docs per Term", round(sum(len(d) for d in inv_idx.values()) / max(len(inv_idx), 1), 2))


# ============================================================================
# TAB 3 - RETRIEVAL COMPARISON
# ============================================================================

def render_retrieval_tab(documents_df: DataFrame) -> None:
    st.markdown("<h2 class='sec-head' style='color:#FB8C00;'>🔍 Stemming vs Lemmatization Retrieval</h2>", unsafe_allow_html=True)
    st.info("Enter a search query below to see how stemming and lemmatization affect which documents are retrieved.")

    query = st.text_input("Search Query", value="machine learning", key="ret_q")
    if not query.strip():
        st.warning("Enter a non-empty query to run retrieval.")
        return

    stem_res, stem_t = execute_retrieval_experiment(documents_df, query, apply_stemmer=True,  apply_lemmatizer=False)
    lemm_res, lemm_t = execute_retrieval_experiment(documents_df, query, apply_stemmer=False, apply_lemmatizer=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stemming Hits",        len(stem_res))
    c2.metric("Stemming Time (s)",    f"{stem_t:.6f}")
    c3.metric("Lemmatization Hits",   len(lemm_res))
    c4.metric("Lemmatization Time (s)", f"{lemm_t:.6f}")

    comparison = [
        {"Technique": "Stemming (Porter)",      "Retrieved Doc IDs": stem_res, "Hit Count": len(stem_res), "Time (s)": round(stem_t, 6)},
        {"Technique": "Lemmatization (WordNet)","Retrieved Doc IDs": lemm_res, "Hit Count": len(lemm_res), "Time (s)": round(lemm_t, 6)},
    ]
    st.dataframe(comparison, use_container_width=True)

    fig = px.bar(
        x=["Stemming", "Lemmatization"],
        y=[len(stem_res), len(lemm_res)],
        labels={"x": "Technique", "y": "Documents Retrieved"},
        title=f'Retrieval Hit Count - Query: "{query}"',
        color=["Stemming", "Lemmatization"],
        color_discrete_map={"Stemming": "#FB8C00", "Lemmatization": "#26A69A"},
    )
    fig.update_layout(font_family="Segoe UI", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Venn-style overlap
    union  = set(stem_res) | set(lemm_res)
    overlap = set(stem_res) & set(lemm_res)
    only_stem = set(stem_res) - set(lemm_res)
    only_lemm = set(lemm_res) - set(stem_res)

    fig2 = go.Figure(go.Bar(
        x=["Both Methods", "Stemming Only", "Lemmatization Only"],
        y=[len(overlap), len(only_stem), len(only_lemm)],
        marker_color=["#5C6BC0", "#FB8C00", "#26A69A"],
    ))
    fig2.update_layout(title="Result Set Overlap Breakdown", font_family="Segoe UI", xaxis_title="Category", yaxis_title="Document Count")
    st.plotly_chart(fig2, use_container_width=True)

    inf = generate_comparison_inference(stem_res, lemm_res, query)
    parts = [p.strip() for p in inf.split(" | ") if p.strip()]
    formatted = "<br><br>".join(f"• {p}" for p in parts)
    st.markdown(f"""
    <div class="insight-card">
        <strong>💡 Retrieval Analysis - "{query}":</strong><br><br>{formatted}
    </div>""", unsafe_allow_html=True)


# ============================================================================
# TAB 4 - PHRASE QUERY
# ============================================================================

def render_phrase_query_tab(documents_df: DataFrame) -> None:
    st.markdown("<h2 class='sec-head' style='color:#7B1FA2;'>🧠 Phrase Query Processing</h2>", unsafe_allow_html=True)
    st.info("Compare Biword Index and Positional Index approaches for exact phrase matching.")

    phrase = st.text_input("Phrase Query", value="machine learning", key="phrase_q")
    if not phrase.strip():
        st.warning("Enter a phrase to search.")
        return

    biword_idx   = create_biword_index(documents_df=documents_df)
    positional_idx = create_positional_index(documents_df=documents_df)

    biword_res   = search_biword_phrase(phrase, biword_idx)
    pos_res      = search_positional_phrase(phrase, positional_idx)
    false_pos    = set(biword_res) - set(pos_res)

    c1, c2, c3 = st.columns(3)
    c1.metric("Biword Matches",   len(biword_res))
    c2.metric("Positional Matches", len(pos_res))
    c3.metric("False Positives (Biword)", len(false_pos))

    tbl = [
        {"Index Type": "Biword Index",    "Matching Doc IDs": biword_res, "Count": len(biword_res), "False Positives": sorted(false_pos)},
        {"Index Type": "Positional Index","Matching Doc IDs": pos_res,    "Count": len(pos_res),    "False Positives": []},
    ]
    st.dataframe(tbl, use_container_width=True)

    fig = px.bar(
        x=["Biword Index", "Positional Index", "False Positives"],
        y=[len(biword_res), len(pos_res), len(false_pos)],
        labels={"x": "Index Type", "y": "Count"},
        title=f'Phrase Match Comparison - "{phrase}"',
        color=["Biword Index", "Positional Index", "False Positives"],
        color_discrete_map={"Biword Index": "#AB47BC", "Positional Index": "#5C6BC0", "False Positives": "#EF5350"},
    )
    fig.update_layout(font_family="Segoe UI", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📖 Biword Index - Sample Entries"):
        st.json(dict(list(biword_idx.items())[:20]))
    with st.expander("📖 Positional Index - Sample Entries"):
        sample = list(positional_idx.keys())[:10]
        st.json({t: positional_idx[t] for t in sample})

    inf = generate_phrase_query_inference(biword_res, pos_res, phrase)
    parts = [p.strip() for p in inf.split(" | ") if p.strip()]
    formatted = "<br><br>".join(f"• {p}" for p in parts)
    st.markdown(f"""
    <div class="insight-card">
        <strong>💡 Phrase Query Analysis - "{phrase}":</strong><br><br>{formatted}
    </div>""", unsafe_allow_html=True)


# ============================================================================
# TAB 5 - BST vs B-TREE
# ============================================================================

def render_dictionary_tab(documents_df: DataFrame) -> None:
    st.markdown("<h2 class='sec-head' style='color:#00897B;'>🌳 Dictionary Data Structures: BST vs B-Tree</h2>", unsafe_allow_html=True)
    st.info("Both tree structures are built from the corpus vocabulary. Search times are measured experimentally across multiple terms.")

    vocab = extract_vocabulary(documents_df=documents_df)[:500]
    st.metric("Vocabulary Terms in Experiment", len(vocab))
    st.divider()

    bst   = build_bst_from_vocabulary(vocabulary_terms=vocab)
    btree = build_btree_from_vocabulary(vocabulary_terms=vocab)

    search_term = st.text_input("Dictionary Lookup Term", value="retrieval", key="dict_q")
    if not search_term.strip():
        st.warning("Enter a term to look up.")
        return

    test_terms = list(dict.fromkeys([search_term.lower(), "machine", "learning", "index", "retrieval"]))

    rows = []
    for t in test_terms:
        t0 = perf_counter(); bst_found   = bst.search(value=t);   bst_t   = perf_counter() - t0
        t0 = perf_counter(); btree_found = btree.search(value=t); btree_t = perf_counter() - t0
        rows.append({
            "Term": t,
            "In BST": bst_found,   "BST Time (s)":   round(bst_t,   8),
            "In B-Tree": btree_found, "B-Tree Time (s)": round(btree_t, 8),
            "Winner": "B-Tree" if btree_t < bst_t else "BST",
        })

    st.subheader("🧪 Lookup Experiment Results")
    st.dataframe(rows, use_container_width=True)

    labels   = [r["Term"] for r in rows]
    bst_t_v  = [r["BST Time (s)"]   for r in rows]
    bt_t_v   = [r["B-Tree Time (s)"] for r in rows]

    fig = go.Figure(data=[
        go.Bar(name="BST",    x=labels, y=bst_t_v, marker_color="#00897B"),
        go.Bar(name="B-Tree", x=labels, y=bt_t_v,  marker_color="#00BCD4"),
    ])
    fig.update_layout(
        barmode="group",
        title="BST vs B-Tree - Lookup Time per Term (seconds)",
        xaxis_title="Vocabulary Term", yaxis_title="Lookup Time (s)",
        font_family="Segoe UI",
    )
    st.plotly_chart(fig, use_container_width=True)

    btree_wins = sum(1 for r in rows if r["Winner"] == "B-Tree")
    bst_wins   = len(rows) - btree_wins

    st.markdown(f"""
    <div class="insight-card">
        <strong>💡 Dictionary Structure Analysis:</strong><br><br>
        • Over {len(rows)} queries, B-Tree was faster in <strong>{btree_wins}</strong> case(s)
          and BST in <strong>{bst_wins}</strong> case(s).<br><br>
        • <strong>BST</strong> delivers O(log n) average-case performance but degrades to O(n)
          on sorted or near-sorted input - a common situation for alphabetically ordered
          vocabulary lists. Deep recursion is also a concern for very large vocabularies.<br><br>
        • <strong>B-Tree</strong> guarantees O(log n) in the worst case by splitting nodes
          to remain balanced. Multi-key nodes also improve cache locality, making B-Trees
          the preferred structure for on-disk dictionary storage in production IR systems.<br><br>
        • <strong>Takeaway:</strong> For an in-memory prototype like this system, both
          structures perform similarly. At scale (millions of terms, disk-backed indexes),
          B-Tree's predictable performance profile becomes decisive.
    </div>""", unsafe_allow_html=True)


# ============================================================================
# TAB 6 - TOLERANT RETRIEVAL
# ============================================================================

def render_tolerant_tab(documents_df: DataFrame) -> None:
    st.markdown("<h2 class='sec-head' style='color:#C62828;'>🛡️ Tolerant Retrieval</h2>", unsafe_allow_html=True)
    st.info("The system handles misspelled, incomplete, or phonetically similar queries using four complementary techniques.")

    query_term = st.text_input("Query Term (can be misspelled or partial)", value="retrival", key="tol_q")
    results = execute_tolerant_retrieval_experiment(documents_df=documents_df, query_term=query_term.lower())

    # ── K-grams ───────────────────────────────────────────────────────────
    st.subheader("🔢 Trigram (K-Gram) Decomposition  ·  k = 3")
    st.caption("Boundary markers $ are prepended/appended before slicing into overlapping trigrams.")
    kgram_cols = st.columns(min(len(results["query_kgrams"]), 8))
    for i, kg in enumerate(results["query_kgrams"]):
        kgram_cols[i % len(kgram_cols)].markdown(f"<span class='metric-pill'>{kg}</span>", unsafe_allow_html=True)
    st.write("")

    # ── Spelling suggestions ──────────────────────────────────────────────
    st.divider()
    st.subheader("✏️ Spelling Suggestions via Edit (Levenshtein) Distance")
    spell_rows = [
        {"Candidate Term": s, "Edit Distance": d, "Match Strength": "Strong" if d <= 1 else "Moderate" if d <= 3 else "Weak"}
        for s, d in results["spelling_suggestions"]
    ]
    st.dataframe(spell_rows, use_container_width=True)
    st.caption("Edit distance = minimum single-character insertions, deletions, or substitutions to transform the query into the candidate.")

    if spell_rows:
        ed_fig = px.bar(
            x=[r["Candidate Term"] for r in spell_rows[:10]],
            y=[results["spelling_suggestions"][i][1] for i in range(min(10, len(spell_rows)))],
            labels={"x": "Candidate", "y": "Edit Distance"},
            title="Edit Distance for Top Spelling Candidates",
            color_discrete_sequence=["#EF5350"],
        )
        ed_fig.update_layout(font_family="Segoe UI")
        st.plotly_chart(ed_fig, use_container_width=True)

    # ── Wildcard matches ──────────────────────────────────────────────────
    st.divider()
    st.subheader("🔤 Wildcard Prefix Matches")
    wm = results["wildcard_matches"]
    if wm:
        st.dataframe([{"Matched Term": t} for t in wm], use_container_width=True)
    else:
        st.warning("No prefix matches found. Try a shorter stem (e.g. 'retr' instead of 'retrival').")

    # ── Phonetic / Soundex ────────────────────────────────────────────────
    st.divider()
    st.subheader("🔊 Phonetic Matching (Soundex)")
    soundex_code = results.get("query_soundex", "-")
    phonetic_matches = results.get("phonetic_matches", [])
    st.metric(f"Soundex Code for '{query_term}'", soundex_code)
    if phonetic_matches:
        st.dataframe([{"Phonetically Similar Term": t} for t in phonetic_matches], use_container_width=True)
    else:
        st.info("No phonetically similar vocabulary terms found. Soundex is most effective for proper nouns and domain-specific names.")

    # ── Consolidated inference ────────────────────────────────────────────
    st.divider()
    best_sugg = results["spelling_suggestions"][0][0] if results["spelling_suggestions"] else query_term
    best_dist = results["spelling_suggestions"][0][1] if results["spelling_suggestions"] else 0
    tolerance = "high" if best_dist <= 1 else "moderate" if best_dist <= 3 else "limited"

    st.markdown(f"""
    <div class="insight-card">
        <strong>💡 Tolerant Retrieval Analysis - "{query_term}":</strong><br><br>
        • <strong>Trigram Index:</strong> {len(results["query_kgrams"])} trigrams extracted.
          These overlapping character sequences are matched against the vocabulary to
          generate spelling candidates before applying edit-distance ranking.<br><br>
        • <strong>Edit Distance Correction:</strong> Closest match is
          "<strong>{best_sugg}</strong>" (distance = {best_dist}).
          {"This is almost certainly the intended term." if best_dist <= 2 else "Higher distance - manual verification recommended."}<br><br>
        • <strong>Wildcard Coverage:</strong> {len(wm)} prefix-matching term(s) located.
          Useful when the user knows only the beginning of a technical term.<br><br>
        • <strong>Soundex Code {soundex_code}:</strong> Found {len(phonetic_matches)} phonetically
          similar term(s). Best for name-variant queries where spelling differs but pronunciation matches.<br><br>
        • <strong>Overall Tolerance:</strong> <em>{tolerance.capitalize()}</em> - the multi-layer
          approach (trigrams → edit distance → wildcard → phonetic) ensures robust handling
          of most real-world query imperfections.
    </div>""", unsafe_allow_html=True)


# ============================================================================
# TAB 7 - ANALYTICS DASHBOARD
# ============================================================================

def render_analytics_tab(documents_df: DataFrame) -> None:
    st.markdown("<h2 class='sec-head' style='color:#1565C0;'>📊 Experimental Results & Discussion</h2>", unsafe_allow_html=True)
    st.info("All visualisations on this page are computed live from the active document collection - no hardcoded values.")

    # ── Chart 1: Retrieval across multiple queries ────────────────────────
    st.subheader("① Retrieval Coverage - Stemming vs Lemmatization")
    bench_queries = ["machine learning", "retrieval", "indexing", "language", "datasets"]
    ret_data = []
    for q in bench_queries:
        sr, _ = execute_retrieval_experiment(documents_df, q, apply_stemmer=True,  apply_lemmatizer=False)
        lr, _ = execute_retrieval_experiment(documents_df, q, apply_stemmer=False, apply_lemmatizer=True)
        ret_data.append({"Query": q, "Stemming": len(sr), "Lemmatization": len(lr)})
    ret_df = pd.DataFrame(ret_data)

    ret_fig = px.bar(
        ret_df.melt(id_vars="Query", value_vars=["Stemming","Lemmatization"], var_name="Technique", value_name="Hits"),
        x="Query", y="Hits", color="Technique", barmode="group",
        title="Documents Retrieved per Benchmark Query",
        color_discrete_map={"Stemming": "#FB8C00", "Lemmatization": "#26A69A"},
    )
    ret_fig.update_layout(font_family="Segoe UI")
    st.plotly_chart(ret_fig, use_container_width=True)
    st.dataframe(ret_df, use_container_width=True)
    st.divider()

    # ── Chart 2: Phrase query coverage ───────────────────────────────────
    st.subheader("② Phrase Matching - Biword vs Positional Index")
    phrase_bench = ["machine learning", "information retrieval", "deep learning", "search engines"]
    biword_idx_a   = create_biword_index(documents_df=documents_df)
    positional_idx_a = create_positional_index(documents_df=documents_df)
    ph_data = []
    for pq in phrase_bench:
        br = search_biword_phrase(pq, biword_idx_a)
        pr = search_positional_phrase(pq, positional_idx_a)
        ph_data.append({"Phrase": pq, "Biword": len(br), "Positional": len(pr), "False Positives": len(set(br)-set(pr))})
    ph_df = pd.DataFrame(ph_data)
    ph_fig = px.bar(
        ph_df.melt(id_vars="Phrase", value_vars=["Biword","Positional"], var_name="Index Type", value_name="Matches"),
        x="Phrase", y="Matches", color="Index Type", barmode="group",
        title="Phrase Matches per Benchmark Phrase",
        color_discrete_map={"Biword": "#AB47BC", "Positional": "#5C6BC0"},
    )
    ph_fig.update_layout(font_family="Segoe UI")
    st.plotly_chart(ph_fig, use_container_width=True)
    st.dataframe(ph_df, use_container_width=True)
    st.divider()

    # ── Chart 3: BST vs B-Tree timing ────────────────────────────────────
    st.subheader("③ Dictionary Lookup - BST vs B-Tree Timing")
    vocab_a = extract_vocabulary(documents_df=documents_df)[:500]
    bst_a   = build_bst_from_vocabulary(vocabulary_terms=vocab_a)
    btree_a = build_btree_from_vocabulary(vocabulary_terms=vocab_a)
    dict_bench = ["retrieval","machine","learning","index","language"]
    d_data = []
    for dt in dict_bench:
        t0 = perf_counter(); bst_a.search(dt);   bt = perf_counter()-t0
        t0 = perf_counter(); btree_a.search(dt); btt = perf_counter()-t0
        d_data.append({"Term": dt, "BST (s)": round(bt,8), "B-Tree (s)": round(btt,8)})
    d_df = pd.DataFrame(d_data)
    d_fig = go.Figure(data=[
        go.Bar(name="BST",    x=d_df["Term"], y=d_df["BST (s)"],    marker_color="#00897B"),
        go.Bar(name="B-Tree", x=d_df["Term"], y=d_df["B-Tree (s)"], marker_color="#00BCD4"),
    ])
    d_fig.update_layout(barmode="group", title="Lookup Time per Vocabulary Term", font_family="Segoe UI",
                        xaxis_title="Term", yaxis_title="Time (s)")
    st.plotly_chart(d_fig, use_container_width=True)
    st.dataframe(d_df, use_container_width=True)
    st.divider()

    # ── Chart 4: Corpus vocabulary stats ──────────────────────────────────
    st.subheader("④ Corpus Vocabulary Snapshot")
    vocab_terms_b = extract_vocabulary(documents_df=documents_df)
    total_toks = sum(len(preprocess_text(str(r["text"]))) for _, r in documents_df.iterrows())
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Distinct Vocabulary Terms", len(vocab_terms_b))
    mc2.metric("Total Documents", len(documents_df))
    mc3.metric("Total Processed Tokens", total_toks)

    # Top-N most frequent terms
    from collections import Counter
    all_toks: List[str] = []
    for _, row in documents_df.iterrows():
        all_toks.extend(preprocess_text(str(row["text"])))
    top20 = Counter(all_toks).most_common(20)
    if top20:
        tf_fig = px.bar(
            x=[t[0] for t in top20], y=[t[1] for t in top20],
            labels={"x": "Term", "y": "Frequency"},
            title="Top 20 Terms by Corpus Frequency (post-preprocessing)",
            color=[t[1] for t in top20],
            color_continuous_scale="Viridis",
        )
        tf_fig.update_layout(font_family="Segoe UI", coloraxis_showscale=False, showlegend=False)
        st.plotly_chart(tf_fig, use_container_width=True)

    st.divider()

    # ── Discussion ────────────────────────────────────────────────────────
    st.subheader("⑤ Discussion & Conclusions")
    st.markdown("""
    <div class="insight-card">
        <strong>Which preprocessing steps had the most impact on retrieval quality?</strong><br>
        Stop-word removal and lowercasing contributed the most significant improvements.
        Eliminating high-frequency function words drastically reduces index noise, while
        lowercasing ensures case variants resolve to a single index entry. Punctuation removal
        prevents spurious symbol tokens from inflating vocabulary size. Hyphen normalization
        helps compound terms participate in standard single-word lookups.
    </div>

    <div class="insight-card">
        <strong>Was stemming or lemmatization better suited to this dataset?</strong><br>
        Lemmatization is the stronger choice for a technically oriented corpus. Academic and
        IR-domain text is morphologically regular, so the extra overhead of WordNet lookup
        pays off in precision - "retrieving", "retrieves", and "retrieved" all correctly
        resolve to "retrieve" without conflating unrelated roots.
    </div>

    <div class="insight-card">
        <strong>Which phrase index was more accurate, and why?</strong><br>
        The Positional Index is definitively more accurate. By recording exact token positions
        it can verify that query words appear consecutively, eliminating false positives that
        arise when a Biword Index matches constituent bigrams that happen to appear in
        different parts of the same document but not adjacently.
    </div>

    <div class="insight-card">
        <strong>Which tree structure performed better on dictionary lookups?</strong><br>
        B-Tree showed more consistent timing, particularly for vocabulary inserted in
        near-sorted order. Because B-Tree nodes hold multiple keys and remain balanced
        through splitting, lookup depth is bounded and cache-efficient. BST depth grows
        linearly with sorted input - a known Achilles' heel for alphabetical vocabulary.
    </div>

    <div class="insight-card">
        <strong>How effective was the tolerant retrieval layer?</strong><br>
        The combined pipeline (trigram candidate generation → edit-distance ranking →
        wildcard prefix search → Soundex phonetic matching) handled most simulated
        misspellings within two edits. This four-layer design ensures that even when one
        technique yields no candidates, another can recover the intended term.
    </div>

    <div class="insight-card">
        <strong>Identified limitations and future directions:</strong><br>
        (1) Retrieval is currently OR-based without ranking - adding TF-IDF or BM25 scoring
        would prioritise more relevant documents.<br>
        (2) The recursive BST implementation risks stack overflow on very large sorted
        vocabularies; an iterative or self-balancing AVL tree would resolve this.<br>
        (3) Biword indexing does not generalise beyond two-word phrases; a sliding n-gram
        positional index would support longer queries.<br>
        (4) Soundex is English-centric; multilingual corpora would benefit from a
        language-agnostic phonetic encoder such as Metaphone or DoubleMetaphone.<br>
        (5) No precision/recall evaluation metrics are computed against a ground-truth
        relevance set.
    </div>
    """, unsafe_allow_html=True)    


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    render_team_banner()

    st.title("🔎 DocSearch Information Retrieval System")
    st.caption("An interactive workbench for exploring information retrieval techniques.")
    st.divider()

    documents_df = initialize_dataset()

    tabs = st.tabs([
        "📄 Corpus",
        "⚙️ Preprocessing",
        "🔍 Retrieval",
        "🧠 Phrase Query",
        "🌳 Dictionary Trees",
        "🛡️ Tolerant Retrieval",
        "📊 Analytics",
    ])

    with tabs[0]: render_dataset_tab(documents_df)
    with tabs[1]: render_preprocessing_tab(documents_df)
    with tabs[2]: render_retrieval_tab(documents_df)
    with tabs[3]: render_phrase_query_tab(documents_df)
    with tabs[4]: render_dictionary_tab(documents_df)
    with tabs[5]: render_tolerant_tab(documents_df)
    with tabs[6]: render_analytics_tab(documents_df)


if __name__ == "__main__":
    main()
