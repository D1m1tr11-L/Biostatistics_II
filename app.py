import json
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
from openai import OpenAI

from utils.llm import extract_km_data_with_sampling
from utils.reconstruct import (
    km_data_to_group_summary_tables,
    validate_group_summary_tables,
    get_group_survival_bounds,
    group_summary_tables_to_internal_dataframe,
    dataframe_to_group_real_points,
    build_logrank_dataframe_from_points
)
from utils.survival import run_km_analysis_from_dataframe
from utils.literature import summarize_uploaded_literature_files

client = OpenAI()

st.set_page_config(page_title="Extract Your Data", layout="wide")

# -------------------------
# Unified Global Style
# -------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.6rem;
            max-width: 1220px;
        }

        .main > div {
            padding-top: 0rem;
        }

        .hero-shell {
            padding: 2.2rem 2rem 1.9rem 2rem;
            border-radius: 28px;
            background:
                radial-gradient(circle at top left, rgba(67,97,238,0.16), transparent 35%),
                radial-gradient(circle at bottom right, rgba(76,201,240,0.13), transparent 35%),
                linear-gradient(135deg, rgba(255,255,255,0.85), rgba(248,250,252,0.88));
            border: 1px solid rgba(120,120,120,0.18);
            box-shadow: 0 12px 34px rgba(0,0,0,0.05);
            margin-bottom: 1.3rem;
        }

        .hero-title {
            font-size: 2.55rem;
            font-weight: 760;
            line-height: 1.08;
            margin-bottom: 0.45rem;
            letter-spacing: -0.02em;
        }

        .hero-subtitle {
            font-size: 1.04rem;
            color: #5b6472;
            line-height: 1.55;
            max-width: 860px;
        }

        .page-panel {
            padding: 1.1rem 1.15rem 0.9rem 1.15rem;
            border-radius: 22px;
            border: 1px solid rgba(120,120,120,0.16);
            background: linear-gradient(180deg, rgba(255,255,255,0.82), rgba(248,250,252,0.72));
            box-shadow: 0 8px 24px rgba(0,0,0,0.04);
            margin-bottom: 1.05rem;
        }

        .page-title {
            font-size: 2rem;
            font-weight: 740;
            line-height: 1.18;
            margin-bottom: 0.22rem;
            letter-spacing: -0.02em;
        }

        .page-subtitle {
            color: #667085;
            font-size: 0.98rem;
            line-height: 1.45;
            margin-bottom: 0.1rem;
        }

        .feature-card {
            padding: 1.25rem 1.15rem 1.05rem 1.15rem;
            border-radius: 22px;
            border: 1px solid rgba(120,120,120,0.18);
            background: linear-gradient(180deg, rgba(255,255,255,0.82), rgba(248,250,252,0.74));
            box-shadow: 0 10px 24px rgba(0,0,0,0.04);
            min-height: 205px;
        }

        .feature-title {
            font-size: 1.16rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .feature-text {
            color: #667085;
            font-size: 0.96rem;
            line-height: 1.55;
            margin-bottom: 0.95rem;
        }

        .section-chip {
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 650;
            color: #335caa;
            background: rgba(67,97,238,0.08);
            border: 1px solid rgba(67,97,238,0.12);
            margin-bottom: 0.65rem;
        }

        .soft-box {
            padding: 0.95rem 1rem 0.8rem 1rem;
            border-radius: 18px;
            border: 1px solid rgba(120,120,120,0.14);
            background: rgba(250,250,250,0.45);
            margin-bottom: 0.95rem;
        }

        .minor-note {
            color: #7a8191;
            font-size: 0.9rem;
            line-height: 1.45;
        }

        .summary-line {
            font-size: 0.95rem;
            color: #475467;
            line-height: 1.45;
        }

        div[data-testid="stButton"] > button {
            border-radius: 14px !important;
            border: 1px solid rgba(120,120,120,0.22) !important;
            font-weight: 650 !important;
            min-height: 2.8rem;
        }

        div[data-testid="stDownloadButton"] > button {
            border-radius: 14px !important;
            border: 1px solid rgba(120,120,120,0.22) !important;
            font-weight: 650 !important;
            min-height: 2.8rem;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(120,120,120,0.14);
        }

        div[data-testid="stExpander"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(120,120,120,0.10);
        }

        div[data-testid="stFileUploader"] {
            border-radius: 16px;
        }

        div[data-baseweb="input"] {
            border-radius: 14px !important;
        }

        div[data-baseweb="select"] > div {
            border-radius: 14px !important;
        }

        .footer-note {
            margin-top: 1rem;
            color: #7a8191;
            font-size: 0.88rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Session State
# -------------------------
if "page_mode" not in st.session_state:
    st.session_state["page_mode"] = "home"

if "pending_group_tables" not in st.session_state:
    st.session_state["pending_group_tables"] = None

if "show_warning" not in st.session_state:
    st.session_state["show_warning"] = False

if "applied_group_tables" not in st.session_state:
    st.session_state["applied_group_tables"] = None

if "literature_summaries" not in st.session_state:
    st.session_state["literature_summaries"] = None

if "normalized_literature_summaries" not in st.session_state:
    st.session_state["normalized_literature_summaries"] = None

if "relationship_result" not in st.session_state:
    st.session_state["relationship_result"] = None


def parse_optional_float(text):
    text = text.strip()
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None


def reset_pending():
    st.session_state["pending_group_tables"] = None
    st.session_state["show_warning"] = False


def hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def page_header(title: str, subtitle: str, chip: str = None):
    chip_html = f'<div class="section-chip">{chip}</div>' if chip else ""
    st.markdown(
        f"""
        <div class="page-panel">
            {chip_html}
            <div class="page-title">{title}</div>
            <div class="page-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def soft_summary(text: str):
    st.markdown(
        f"""
        <div class="soft-box">
            <div class="summary-line">{text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def normalize_literature_terms_llm(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt = f"""
You are given multiple structured literature summaries.

Goal:
Normalize concept names across papers so that equivalent terms map to the same canonical label.

Rules:
- Return JSON only
- Be conservative
- Keep original values
- Add normalized fields
- If a value is missing, keep normalized value as null

Return exactly this schema:

{{
  "normalized_summaries": [
    {{
      "paper_name": null,
      "title": null,
      "research_question": null,
      "population": null,
      "study_design": null,
      "intervention_or_exposure": null,
      "comparison": null,
      "outcome": null,
      "main_finding": null,
      "effect_type": null,
      "effect_value": null,
      "ci_lower": null,
      "ci_upper": null,
      "p_value": null,
      "key_terms": [],
      "plain_summary": null,
      "normalized_intervention_or_exposure": null,
      "normalized_comparison": null,
      "normalized_outcome": null
    }}
  ]
}}

Input summaries:
{json.dumps(summaries, indent=2)}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        top_p=1,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "developer",
                "content": "Normalize literature variables across studies. Return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return json.loads(response.choices[0].message.content)


def infer_relationships_from_normalized_summaries(
    normalized_summaries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    prompt = f"""
You are given multiple normalized literature summaries.

Goal:
1. Detect direct chains such as:
   A -> B
   B -> C
2. Infer A -> C if justified.

Use normalized fields preferentially:
- normalized_intervention_or_exposure
- normalized_comparison
- normalized_outcome

Rules:
- Return JSON only
- Be strict and conservative
- Do not invent unsupported relationships
- If no valid chain exists, return an empty list

Return exactly this schema:

{{
  "chains": [
    {{
      "A": null,
      "B": null,
      "C": null,
      "paper_AB": null,
      "paper_BC": null,
      "inferred_relationship": null,
      "confidence": "high",
      "explanation": null
    }}
  ]
}}

Input normalized summaries:
{json.dumps(normalized_summaries, indent=2)}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        top_p=1,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "developer",
                "content": "Detect cross-paper chains conservatively. Return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return json.loads(response.choices[0].message.content)


@st.dialog("Academic integrity warning")
def confirm_edit_dialog():
    st.warning(
        "Changing reconstructed survival values may affect the statistical result. "
        "Improper manual alteration may constitute academic misconduct. "
        "Only confirm changes when you have a valid methodological reason."
    )

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Cancel changes", key="dialog_cancel_changes", use_container_width=True):
            reset_pending()
            st.rerun()

    with c2:
        if st.button("Confirm changes", key="dialog_confirm_changes", use_container_width=True):
            if st.session_state["pending_group_tables"] is not None:
                st.session_state["applied_group_tables"] = {
                    k: v.copy() for k, v in st.session_state["pending_group_tables"].items()
                }
            reset_pending()
            st.rerun()


# =========================
# HOME
# =========================
if st.session_state["page_mode"] == "home":
    hero(
        "Extract Your Data",
        "A unified workspace for Kaplan–Meier curve extraction, literature summarization, "
        "variable normalization, and cross-paper relationship inference."
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="section-chip">KM module</div>
                <div class="feature-title">Extract Data From KM curve</div>
                <div class="feature-text">
                    Upload a Kaplan–Meier image, extract structured JSON with an LLM,
                    generate editable grouped tables, and perform a log-rank test.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open KM Extractor", use_container_width=True):
            st.session_state["page_mode"] = "km"
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="section-chip">Literature module</div>
                <div class="feature-title">Extract Their Relationship</div>
                <div class="feature-text">
                    Upload any number of papers, generate standardized summaries,
                    normalize key variables, and infer cross-paper A→B→C chains.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Relationship Extractor", use_container_width=True):
            st.session_state["page_mode"] = "relation"
            st.rerun()

    st.markdown(
        '<div class="footer-note">Everything runs inside a single app flow with a unified interface.</div>',
        unsafe_allow_html=True
    )
    st.stop()


# =========================
# RELATIONSHIP PAGE
# =========================
if st.session_state["page_mode"] == "relation":
    top1, top2 = st.columns([1, 10])

    with top1:
        if st.button("← Home", use_container_width=True):
            st.session_state["page_mode"] = "home"
            st.rerun()

    with top2:
        page_header(
            "Extract Their Relationship",
            "Upload literature files, summarize them in a standardized format, normalize variables, and infer cross-paper relationships.",
            chip="Literature workflow"
        )

    st.subheader("1. Upload Literature")
    uploaded_papers = st.file_uploader(
        "Upload papers (.pdf, .docx, .txt, .md)",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_papers:
        soft_summary(f"{len(uploaded_papers)} file(s) uploaded and ready for processing.")

    st.subheader("2. Generate Standardized Summaries")
    st.write(
        "The LLM will read each uploaded paper and return a consistent structured summary."
    )

    if uploaded_papers:
        if st.button("Summarize Literature", use_container_width=True):
            with st.spinner("Reading and summarizing uploaded literature..."):
                summaries = summarize_uploaded_literature_files(uploaded_papers)
                st.session_state["literature_summaries"] = summaries
                st.session_state["normalized_literature_summaries"] = None
                st.session_state["relationship_result"] = None

    if st.session_state.get("literature_summaries") is not None:
        st.subheader("3. Summary Table")

        summaries = st.session_state["literature_summaries"]

        table_rows = []
        for item in summaries:
            if "error" in item:
                table_rows.append({
                    "paper_name": item.get("paper_name"),
                    "title": None,
                    "research_question": None,
                    "population": None,
                    "study_design": None,
                    "outcome": None,
                    "main_finding": None,
                    "effect_type": None,
                    "effect_value": None,
                    "p_value": None,
                    "error": item.get("error")
                })
            else:
                table_rows.append({
                    "paper_name": item.get("paper_name"),
                    "title": item.get("title"),
                    "research_question": item.get("research_question"),
                    "population": item.get("population"),
                    "study_design": item.get("study_design"),
                    "outcome": item.get("outcome"),
                    "main_finding": item.get("main_finding"),
                    "effect_type": item.get("effect_type"),
                    "effect_value": item.get("effect_value"),
                    "p_value": item.get("p_value"),
                    "error": None
                })

        summary_df = pd.DataFrame(table_rows)
        st.dataframe(summary_df, use_container_width=True)

        st.download_button(
            "Download literature summary CSV",
            summary_df.to_csv(index=False),
            file_name="literature_summaries.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.subheader("4. Normalize Variables")
        st.write("Standardize key variables so equivalent terms align across papers.")

        if st.button("Normalize Variables", use_container_width=True):
            valid_summaries = [x for x in summaries if "error" not in x]
            with st.spinner("Normalizing variables across papers..."):
                norm_result = normalize_literature_terms_llm(valid_summaries)
                st.session_state["normalized_literature_summaries"] = norm_result.get(
                    "normalized_summaries", []
                )
                st.session_state["relationship_result"] = None

    if st.session_state.get("normalized_literature_summaries") is not None:
        st.subheader("5. Normalized Summary Table")

        normalized = st.session_state["normalized_literature_summaries"]

        norm_rows = []
        for item in normalized:
            norm_rows.append({
                "paper_name": item.get("paper_name"),
                "normalized_intervention_or_exposure": item.get("normalized_intervention_or_exposure"),
                "normalized_comparison": item.get("normalized_comparison"),
                "normalized_outcome": item.get("normalized_outcome"),
                "effect_type": item.get("effect_type"),
                "effect_value": item.get("effect_value"),
                "p_value": item.get("p_value")
            })

        norm_df = pd.DataFrame(norm_rows)
        st.dataframe(norm_df, use_container_width=True)

        st.download_button(
            "Download normalized summary CSV",
            norm_df.to_csv(index=False),
            file_name="normalized_literature_summaries.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.subheader("6. Infer Cross-paper Relationships")
        st.write("Search for A→B and B→C chains and infer A→C where justified.")

        if st.button("Infer Relationships", use_container_width=True):
            with st.spinner("Inferring cross-paper relationships..."):
                result = infer_relationships_from_normalized_summaries(normalized)
                st.session_state["relationship_result"] = result

    if st.session_state.get("relationship_result") is not None:
        st.subheader("7. Inferred Relationships")

        result = st.session_state["relationship_result"]
        chains = result.get("chains", [])

        if len(chains) == 0:
            st.info("No valid A→B→C chain detected.")
        else:
            table_rows = []
            for c in chains:
                table_rows.append({
                    "A": c.get("A"),
                    "B": c.get("B"),
                    "C": c.get("C"),
                    "paper_AB": c.get("paper_AB"),
                    "paper_BC": c.get("paper_BC"),
                    "inferred_relationship": c.get("inferred_relationship"),
                    "confidence": c.get("confidence"),
                    "explanation": c.get("explanation")
                })

            df_chain = pd.DataFrame(table_rows)
            st.dataframe(df_chain, use_container_width=True)

            st.download_button(
                "Download relationship results CSV",
                df_chain.to_csv(index=False),
                file_name="relationship_inference.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.subheader("8. Detailed JSON Outputs")

        if st.session_state.get("literature_summaries") is not None:
            for i, item in enumerate(st.session_state["literature_summaries"], start=1):
                label = item.get("paper_name", f"Paper {i}")
                with st.expander(f"Raw summary JSON: {label}", expanded=False):
                    st.code(json.dumps(item, indent=2), language="json")

        if st.session_state.get("normalized_literature_summaries") is not None:
            with st.expander("View normalized summaries JSON", expanded=False):
                st.code(
                    json.dumps(st.session_state["normalized_literature_summaries"], indent=2),
                    language="json"
                )

        with st.expander("View inferred relationship JSON", expanded=False):
            st.code(json.dumps(result, indent=2), language="json")

    st.stop()


# =========================
# KM PAGE
# =========================
if st.session_state["page_mode"] == "km":
    top1, top2 = st.columns([1, 10])

    with top1:
        if st.button("← Home", use_container_width=True):
            st.session_state["page_mode"] = "home"
            st.rerun()

    with top2:
        page_header(
            "Extract Data From KM curve",
            "Upload a KM image, extract structured results, generate editable grouped tables, and run a log-rank test.",
            chip="KM workflow"
        )

    st.subheader("1. Upload and Extract")

    c1, c2 = st.columns([3, 2], gap="large")

    with c1:
        uploaded_file = st.file_uploader("Upload KM plot image", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            st.session_state["image_bytes"] = uploaded_file.read()

    with c2:
        n_samples = st.slider("LLM sampling runs", 1, 20, 5)
        st.session_state["n_samples"] = n_samples
        if "image_bytes" in st.session_state:
            if st.button("Extract Data", use_container_width=True):
                with st.spinner("Running repeated LLM extraction..."):
                    km_data, meta = extract_km_data_with_sampling(
                        image_bytes=st.session_state["image_bytes"],
                        n_samples=n_samples
                    )

                    st.session_state["km_data"] = km_data
                    st.session_state["sampling_meta"] = meta
                    st.session_state["group_tables"] = None
                    st.session_state["applied_group_tables"] = None
                    reset_pending()

    if "km_data" in st.session_state:
        sampling_meta = st.session_state.get("sampling_meta", {})
        soft_summary(
            f"<b>Extraction Summary</b><br>"
            f"Requested runs: {sampling_meta.get('requested_samples', 'NA')} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Successful runs: {sampling_meta.get('successful_samples', 'NA')}"
        )

        with st.expander("View JSON Output", expanded=False):
            st.code(json.dumps(st.session_state["km_data"], indent=2), language="json")

    if "km_data" in st.session_state:
        st.subheader("2. Analysis Settings")

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("**Sample Size Assumptions**")
            groups = st.session_state["km_data"]["groups"]
            sample_sizes = {}

            cols = st.columns(min(2, len(groups)))
            for i, g in enumerate(groups[:2]):
                with cols[i]:
                    sample_sizes[g["name"]] = st.number_input(
                        f"{g['name']} sample size",
                        value=100,
                        min_value=1
                    )

            st.session_state["sample_sizes"] = sample_sizes

        with right:
            st.markdown("**Significance Level**")
            alpha_input = st.text_input("Alpha (default 0.05)", value="")
            alpha = parse_optional_float(alpha_input) if alpha_input else 0.05
            if alpha is None:
                alpha = 0.05

            if 0 < alpha < 1:
                st.session_state["alpha"] = alpha
                st.success(f"Current alpha = {alpha}")
            else:
                st.error("Alpha must be between 0 and 1.")

    x_max_override = None
    x_tick_override = None

    if "km_data" in st.session_state:
        st.subheader("3. Generate Editable Group Tables")

        mode = st.radio(
            "Choose table source",
            [
                "Use current data",
                "Rerun image reading with custom X-axis"
            ]
        )

        if mode == "Rerun image reading with custom X-axis":
            st.write(
                "Provide x-axis maximum and tick interval to force the LLM to reread the image using the custom axis."
            )
            col1, col2 = st.columns(2)
            with col1:
                x_max_text = st.text_input("Custom x-axis maximum", value="")
            with col2:
                x_tick_text = st.text_input("Custom x-axis tick interval", value="")

            x_max_override = parse_optional_float(x_max_text)
            x_tick_override = parse_optional_float(x_tick_text)

            if st.button("Generate Editable Group Tables", use_container_width=True):
                if x_max_override is None or x_tick_override is None:
                    st.error("Please provide both custom x-axis maximum and custom x-axis tick interval.")
                else:
                    with st.spinner("Re-reading image with custom X-axis..."):
                        new_km_data, new_meta = extract_km_data_with_sampling(
                            image_bytes=st.session_state["image_bytes"],
                            n_samples=st.session_state.get("n_samples", 5),
                            x_max=x_max_override,
                            x_tick=x_tick_override
                        )

                    st.session_state["km_data"] = new_km_data
                    st.session_state["sampling_meta"] = new_meta

                    group_tables = km_data_to_group_summary_tables(
                        new_km_data,
                        sample_sizes=st.session_state.get("sample_sizes", {}),
                        x_max_override=x_max_override,
                        x_tick_override=x_tick_override
                    )

                    st.session_state["group_tables"] = group_tables
                    st.session_state["applied_group_tables"] = {
                        k: v.copy() for k, v in group_tables.items()
                    }
                    reset_pending()

        else:
            if st.button("Generate Editable Group Tables", use_container_width=True):
                group_tables = km_data_to_group_summary_tables(
                    st.session_state["km_data"],
                    sample_sizes=st.session_state.get("sample_sizes", {})
                )

                st.session_state["group_tables"] = group_tables
                st.session_state["applied_group_tables"] = {
                    k: v.copy() for k, v in group_tables.items()
                }
                reset_pending()

    if st.session_state.get("group_tables") is not None:
        st.subheader("4. Group-specific Tables (Natural Result)")

        natural_tables = st.session_state["group_tables"]
        names = list(natural_tables.keys())

        cols = st.columns(len(names)) if len(names) > 0 else []
        for i, name in enumerate(names):
            with cols[i]:
                st.markdown(f"**{name}**")
                st.dataframe(natural_tables[name], use_container_width=True)

    if st.session_state.get("group_tables") is not None:
        st.subheader("5. Editable Group-specific Tables")

        st.warning(
            "Warning: modifying reconstructed survival values may change the statistical test result. "
            "Improper manual alteration of results may constitute academic misconduct."
        )

        current_tables = (
            {k: v.copy() for k, v in st.session_state["applied_group_tables"].items()}
            if st.session_state["applied_group_tables"] is not None
            else {k: v.copy() for k, v in st.session_state["group_tables"].items()}
        )

        bounded_tables = get_group_survival_bounds(current_tables)
        edited_tables = {}

        names = list(bounded_tables.keys())
        cols = st.columns(len(names)) if len(names) > 0 else []

        for i, group_name in enumerate(names):
            with cols[i]:
                st.markdown(f"**Edit: {group_name}**")
                df_edit = bounded_tables[group_name].copy()

                edited_df = st.data_editor(
                    df_edit,
                    use_container_width=True,
                    num_rows="fixed",
                    key=f"editor_{group_name}",
                    column_config={
                        "time": st.column_config.NumberColumn("time", disabled=True),
                        "event_count": st.column_config.NumberColumn("event_count", disabled=True),
                        "survival": st.column_config.NumberColumn(
                            "survival",
                            min_value=0.0,
                            max_value=1.0,
                            step=0.01,
                            help="Must remain <= upper_bound and >= lower_bound."
                        ),
                        "lower_bound": st.column_config.NumberColumn("lower_bound", disabled=True),
                        "upper_bound": st.column_config.NumberColumn("upper_bound", disabled=True),
                    },
                    disabled=["time", "event_count", "lower_bound", "upper_bound"]
                )

                edited_tables[group_name] = edited_df[["time", "event_count", "survival"]].copy()

        valid, msg = validate_group_summary_tables(edited_tables)

        if valid:
            st.success("Current edited values pass monotonicity checks.")
        else:
            st.error(msg)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Confirm edits", use_container_width=True):
                if not valid:
                    st.error(f"Cannot confirm edits: {msg}")
                else:
                    st.session_state["pending_group_tables"] = {
                        k: v.copy() for k, v in edited_tables.items()
                    }
                    st.session_state["show_warning"] = True
                    st.rerun()

        with c2:
            if st.button("Cancel edits", use_container_width=True):
                reset_pending()
                st.session_state["applied_group_tables"] = {
                    k: v.copy() for k, v in st.session_state["group_tables"].items()
                }
                st.rerun()

    if st.session_state.get("show_warning", False):
        confirm_edit_dialog()

    if st.session_state.get("applied_group_tables") is not None:
        st.subheader("6. Group-specific Tables (Current Final Data)")

        final_tables = st.session_state["applied_group_tables"]
        names = list(final_tables.keys())

        cols = st.columns(len(names)) if len(names) > 0 else []
        for i, name in enumerate(names):
            with cols[i]:
                st.markdown(f"**{name}**")
                st.dataframe(final_tables[name], use_container_width=True)

    if st.session_state.get("applied_group_tables") is not None and "sample_sizes" in st.session_state:
        st.subheader("7. Log-rank Test")

        if st.button("Run Log-rank Test from Current Tables", use_container_width=True):
            final_tables = {
                k: v.copy() for k, v in st.session_state["applied_group_tables"].items()
            }

            valid, msg = validate_group_summary_tables(final_tables)

            if not valid:
                st.error(f"Cannot run analysis: {msg}")
            else:
                internal_df = group_summary_tables_to_internal_dataframe(final_tables)
                group_points = dataframe_to_group_real_points(internal_df)

                df_logrank = build_logrank_dataframe_from_points(
                    group_points,
                    sample_sizes=st.session_state["sample_sizes"]
                )

                st.markdown("**Pseudo Dataset for Log-rank Test**")
                st.dataframe(df_logrank, use_container_width=True)

                st.download_button(
                    "Download pseudo dataset CSV",
                    df_logrank.to_csv(index=False),
                    file_name="km_pseudo_logrank_dataset.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                if not df_logrank.empty:
                    alpha_value = st.session_state.get("alpha", 0.05)

                    result = run_km_analysis_from_dataframe(
                        df_logrank,
                        alpha=alpha_value
                    )

                    st.markdown("**Reconstructed Kaplan-Meier Curve**")
                    st.pyplot(result["plot"])

                    st.markdown("**Log-rank Test Result**")

                    result_table = pd.DataFrame([{
                        "test_statistic": result["test_statistic"],
                        "critical_value": result["critical_value"],
                        "p_value": result["p_value_display"]
                    }])

                    st.dataframe(result_table, use_container_width=True)

                    if result["test_statistic"] > result["critical_value"]:
                        st.success(f"Reject H0 at alpha = {result['alpha']}.")
                    else:
                        st.info(f"Fail to reject H0 at alpha = {result['alpha']}.")