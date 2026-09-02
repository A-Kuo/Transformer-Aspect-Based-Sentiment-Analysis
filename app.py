"""
app.py — ABSA Triage Dashboard

Brings the main cohort's `src.inference.Predictor` (BERT aspect + sentiment
heads, see README) into an interactive dashboard: browse curated example
reviews and their extracted aspect + sentiment, or run your own text through
the model live.

Run from the repo root:
    streamlit run app.py

Model source, in priority order:
  1. The Neon-backed model registry (src/db.py): the currently-active
     training run pushed from a Kaggle notebook via
     scripts/load_absa_results.py, with weights downloaded from Hugging
     Face Hub. This is the only source with a genuinely trained model.
  2. A local checkpoint at models/checkpoint_best.pt, if one exists
     (produced by `python main.py train`).
  3. The untrained baseline (random head weights), same as
     notebooks/01_inference_walkthrough.ipynb — with a clear banner so
     predictions aren't mistaken for a trained model's output.
A missing/unconfigured DATABASE_URL, an unreachable Neon instance, or a
failed Hugging Face download all fall through to the next source rather
than crashing the app — see load_predictor().
"""

import html
import json
import logging
import os
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.evaluate import check_targets
from src.inference import Predictor

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "models" / "checkpoint_best.pt"
CONFIG_PATH = ROOT / "config.yaml"

st.set_page_config(
    page_title="ABSA Triage Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Styling constants
# ─────────────────────────────────────────────────────────────────────────────

SENTIMENT_COLORS = {
    "negative": {"bg": "#FEF2F2", "border": "#EF4444", "text": "#7F1D1D"},
    "neutral": {"bg": "#F3F4F6", "border": "#9CA3AF", "text": "#1F2937"},
    "positive": {"bg": "#ECFDF5", "border": "#10B981", "text": "#064E3B"},
}
SENTIMENT_EMOJI = {"negative": "🔴", "neutral": "⚪", "positive": "🟢"}
SENTIMENT_CHART_COLORS = {"negative": "#EF4444", "neutral": "#9CA3AF", "positive": "#10B981"}

ASPECT_EMOJI = {
    "quality": "🔧",
    "usability": "🖱️",
    "value": "💰",
    "shipping": "📦",
    "customer_service": "☎️",
}
ASPECT_CHART_COLORS = {
    "quality": "#6366F1",
    "usability": "#8B5CF6",
    "value": "#F59E0B",
    "shipping": "#3B82F6",
    "customer_service": "#EC4899",
}
ASPECT_BOX_COLOR = {"bg": "#EFF6FF", "border": "#3B82F6", "text": "#1E3A8A"}

EXAMPLE_REVIEWS = {
    "Select an example…": "",
    "[Mixed] Great quality, slow shipping": "Great quality but shipping took 3 weeks",
    "[Mixed] Food vs. service": "The food was excellent but the waiter was rude and slow.",
    "[Positive] Glowing review": "Absolutely love this product! Great quality and fast shipping.",
    "[Negative] Broke fast": "Terrible experience. Broke after one week. Customer service was unhelpful.",
    "[Neutral] Middling": "Decent for the price. Nothing special but gets the job done.",
    "[Negative] Damaged shipment": "Package arrived damaged. Took 3 weeks to get here. Very disappointed.",
    "[Positive] Easy setup": "Easy to set up and use. Instructions were clear. Good value for money.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Caching: load the model once per session
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_neon():
    """Try the Neon-backed model registry + Hugging Face Hub weights.

    Returns (predictor, active_run) on success, or None if unavailable for
    any reason (no DATABASE_URL, Neon unreachable, no active run, HF
    download failure, missing optional dependencies, ...). Imports are
    deliberately local to this function: if the `db` extras group isn't
    installed, the resulting ImportError is just another reason to fall
    through to the next model source, not a reason to crash before
    st.set_page_config() even runs.
    """
    try:
        from dotenv import load_dotenv
        from huggingface_hub import hf_hub_download
        from sqlalchemy import create_engine

        from src.db import ensure_schema, get_active_run, resolve_database_url

        load_dotenv()
        database_url = resolve_database_url()
        if not database_url:
            return None

        engine = create_engine(database_url)
        ensure_schema(engine)
        run = get_active_run(engine)
        if run is None:
            return None

        local_path = hf_hub_download(
            repo_id=run["hf_repo_id"],
            filename=run["hf_filename"],
            revision=run["hf_revision"],
            token=os.environ.get("HF_TOKEN") or None,
        )
        predictor = Predictor.from_checkpoint(local_path, config_path=str(CONFIG_PATH))
        return predictor, run
    except Exception:
        logger.exception("Neon/HF-backed model unavailable; falling back")
        return None


@st.cache_resource(show_spinner=False)
def load_predictor():
    neon_result = _load_from_neon()
    if neon_result is not None:
        predictor, run = neon_result
        return predictor, "neon", run

    if CHECKPOINT_PATH.exists():
        predictor = Predictor.from_checkpoint(
            str(CHECKPOINT_PATH), config_path=str(CONFIG_PATH)
        )
        return predictor, "local", None

    predictor = Predictor.from_pretrained(config_path=str(CONFIG_PATH))
    return predictor, "baseline", None


@st.cache_data(show_spinner=False)
def run_gallery(_predictor: Predictor, texts: tuple) -> list:
    """Cached batch inference. `_predictor` is excluded from the cache key
    (leading underscore), so this re-runs only when `texts` changes."""
    return _predictor.predict_batch(list(texts))


def initialize_with_progress():
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(
            "<h2 style='text-align:center;'>Loading ABSA Triage Dashboard…</h2>",
            unsafe_allow_html=True,
        )
        bar = st.progress(0, text="Loading configuration…")
        time.sleep(0.1)
        bar.progress(20, text="Checking for a trained model (Neon registry, then local checkpoint)…")
        predictor, source, active_run = load_predictor()
        bar.progress(80, text="Model ready. Rendering dashboard…")
        time.sleep(0.1)
        bar.progress(100, text="Done.")
        time.sleep(0.15)
    placeholder.empty()
    return predictor, source, active_run


predictor, model_source, active_run = initialize_with_progress()
config = predictor.config


# ─────────────────────────────────────────────────────────────────────────────
# Render helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_sentiment_card(sentiment: str, confidence: float) -> None:
    colors = SENTIMENT_COLORS.get(sentiment, SENTIMENT_COLORS["neutral"])
    emoji = SENTIMENT_EMOJI.get(sentiment, "❔")
    st.markdown(
        f"""
        <div style="padding:14px 18px;background:{colors['bg']};
                    border-left:5px solid {colors['border']};border-radius:6px;">
            <div style="font-size:15px;color:{colors['text']};">
                <span style="font-size:22px;margin-right:8px;">{emoji}</span>
                <strong style="font-size:18px;">{sentiment.capitalize()}</strong>
            </div>
            <div style="font-size:13px;color:{colors['text']}aa;margin-top:4px;">
                Confidence: {confidence:.1%}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_aspect_card(aspect: str, confidence: float) -> None:
    emoji = ASPECT_EMOJI.get(aspect, "🏷️")
    colors = ASPECT_BOX_COLOR
    st.markdown(
        f"""
        <div style="padding:14px 18px;background:{colors['bg']};
                    border-left:5px solid {colors['border']};border-radius:6px;">
            <div style="font-size:15px;color:{colors['text']};">
                <span style="font-size:22px;margin-right:8px;">{emoji}</span>
                <strong style="font-size:18px;">{aspect.replace('_', ' ').title()}</strong>
            </div>
            <div style="font-size:13px;color:{colors['text']}aa;margin-top:4px;">
                Confidence: {confidence:.1%}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probs_chart(probs: dict, color_map: dict, height: int = 140) -> None:
    df = pd.DataFrame(
        {"Category": list(probs.keys()), "Confidence": list(probs.values())}
    ).sort_values("Confidence", ascending=False)
    domain = list(color_map.keys())
    range_ = [color_map[k] for k in domain]
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X(
                "Confidence:Q",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, 1]),
                title=None,
            ),
            y=alt.Y("Category:N", sort="-x", title=None),
            color=alt.Color(
                "Category:N", legend=None, scale=alt.Scale(domain=domain, range=range_)
            ),
            tooltip=["Category", alt.Tooltip("Confidence:Q", format=".1%")],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def _badge(label: str, confidence: float, emoji: str, colors: dict) -> str:
    return (
        f'<span style="background:{colors["bg"]};color:{colors["text"]};'
        f'border:1px solid {colors["border"]};border-radius:12px;padding:2px 10px;'
        f'font-size:13px;white-space:nowrap;">{emoji} {label} ({confidence:.0%})</span>'
    )


def render_gallery_table(results: list) -> None:
    # Built as a plain HTML table (not st.dataframe) so it doesn't depend on
    # canvas rendering, and so its badges match the Live Triage result cards.
    rows_html = []
    for r in results:
        review = html.escape(r["text"])
        aspect_badge = _badge(
            r["aspect"].replace("_", " "),
            r["aspect_confidence"],
            ASPECT_EMOJI.get(r["aspect"], "🏷️"),
            ASPECT_BOX_COLOR,
        )
        sentiment_badge = _badge(
            r["sentiment"],
            r["sentiment_confidence"],
            SENTIMENT_EMOJI.get(r["sentiment"], "❔"),
            SENTIMENT_COLORS.get(r["sentiment"], SENTIMENT_COLORS["neutral"]),
        )
        rows_html.append(
            f'<tr>'
            f'<td style="padding:10px 12px;border-bottom:1px solid rgba(128,128,128,0.25);">{review}</td>'
            f'<td style="padding:10px 12px;border-bottom:1px solid rgba(128,128,128,0.25);">{aspect_badge}</td>'
            f'<td style="padding:10px 12px;border-bottom:1px solid rgba(128,128,128,0.25);">{sentiment_badge}</td>'
            f'</tr>'
        )
    st.markdown(
        f"""
        <table style="width:100%;border-collapse:collapse;font-size:14px;">
            <thead>
                <tr style="text-align:left;border-bottom:2px solid rgba(128,128,128,0.4);">
                    <th style="padding:8px 12px;">Review</th>
                    <th style="padding:8px 12px;">Aspect</th>
                    <th style="padding:8px 12px;">Sentiment</th>
                </tr>
            </thead>
            <tbody>{"".join(rows_html)}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render_per_class_table(per_class: dict) -> None:
    rows_html = "".join(
        f'<tr>'
        f'<td style="padding:6px 10px;border-bottom:1px solid rgba(128,128,128,0.25);">{html.escape(cls)}</td>'
        f'<td style="padding:6px 10px;border-bottom:1px solid rgba(128,128,128,0.25);">{m["precision"]:.3f}</td>'
        f'<td style="padding:6px 10px;border-bottom:1px solid rgba(128,128,128,0.25);">{m["recall"]:.3f}</td>'
        f'<td style="padding:6px 10px;border-bottom:1px solid rgba(128,128,128,0.25);">{m["f1"]:.3f}</td>'
        f'<td style="padding:6px 10px;border-bottom:1px solid rgba(128,128,128,0.25);">{m["support"]}</td>'
        f'</tr>'
        for cls, m in per_class.items()
    )
    st.markdown(
        f"""
        <table style="width:100%;border-collapse:collapse;font-size:14px;">
            <thead>
                <tr style="text-align:left;border-bottom:2px solid rgba(128,128,128,0.4);">
                    <th style="padding:6px 10px;">Class</th>
                    <th style="padding:6px 10px;">Precision</th>
                    <th style="padding:6px 10px;">Recall</th>
                    <th style="padding:6px 10px;">F1</th>
                    <th style="padding:6px 10px;">Support</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render_confusion_matrix(matrix: list, labels: list) -> None:
    header_cells = "".join(
        f'<th style="padding:6px 10px;text-align:center;">{html.escape(lbl)}</th>' for lbl in labels
    )
    body_rows = []
    for row_label, row in zip(labels, matrix):
        cells = "".join(f'<td style="padding:6px 10px;text-align:center;">{v}</td>' for v in row)
        body_rows.append(
            f'<tr><th style="padding:6px 10px;text-align:left;">{html.escape(row_label)}</th>{cells}</tr>'
        )
    st.caption("Rows: true label. Columns: predicted label.")
    st.markdown(
        f"""
        <table style="border-collapse:collapse;font-size:13px;">
            <thead><tr><th></th>{header_cells}</tr></thead>
            <tbody>{"".join(body_rows)}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render_success_criteria(checks: list) -> None:
    # See config.yaml's monitoring section for what "floor"/"target" mean
    # and why they're project-specific numbers, not generic ABSA benchmarks.
    if not checks:
        st.caption("No monitoring targets configured in config.yaml.")
        return

    status_style = {
        "BELOW FLOOR": ("#FEF2F2", "#EF4444", "#7F1D1D"),
        "above floor, below target": ("#FFFBEB", "#F59E0B", "#78350F"),
        "OK": ("#ECFDF5", "#10B981", "#064E3B"),
    }
    rows_html = []
    for c in checks:
        if c["meets_floor"] is False:
            status = "BELOW FLOOR"
        elif c["meets_target"] is False:
            status = "above floor, below target"
        else:
            status = "OK"
        bg, border, text = status_style[status]
        bar = " / ".join(
            s
            for s in (
                f"floor {c['floor']:.2f}" if c["floor"] is not None else "",
                f"target {c['target']:.2f}" if c["target"] is not None else "",
            )
            if s
        )
        rows_html.append(
            f'<tr>'
            f'<td style="padding:8px 12px;border-bottom:1px solid rgba(128,128,128,0.25);">{html.escape(c["name"])}</td>'
            f'<td style="padding:8px 12px;border-bottom:1px solid rgba(128,128,128,0.25);">{c["actual"]:.3f}</td>'
            f'<td style="padding:8px 12px;border-bottom:1px solid rgba(128,128,128,0.25);">{bar}</td>'
            f'<td style="padding:8px 12px;border-bottom:1px solid rgba(128,128,128,0.25);">'
            f'<span style="background:{bg};color:{text};border:1px solid {border};border-radius:12px;'
            f'padding:2px 10px;font-size:13px;">{status}</span></td>'
            f'</tr>'
        )
    st.markdown(
        f"""
        <table style="width:100%;border-collapse:collapse;font-size:14px;">
            <thead>
                <tr style="text-align:left;border-bottom:2px solid rgba(128,128,128,0.4);">
                    <th style="padding:8px 12px;">Metric</th>
                    <th style="padding:8px 12px;">Actual</th>
                    <th style="padding:8px 12px;">Floor / Target</th>
                    <th style="padding:8px 12px;">Status</th>
                </tr>
            </thead>
            <tbody>{"".join(rows_html)}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )
    if any(c["meets_floor"] is False for c in checks):
        st.warning(
            "One or more metrics are below their floor — this run likely has a real "
            "problem (a labeling bug, undertraining, or class imbalance not being "
            "corrected), not just room for improvement.",
            icon="⚠️",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🎯 ABSA Triage Dashboard")
    st.caption("BERT aspect + sentiment model")
    st.divider()

    st.markdown("**NAVIGATION**")
    page = st.radio(
        "Select page:",
        ["Live Triage", "Example Gallery", "Model Info"],
        label_visibility="collapsed",
    )
    st.divider()

    if model_source == "neon":
        st.success(
            f"Loaded trained run **{active_run['model_version']}**\n\n"
            f"`{active_run['hf_repo_id']}` @ `{active_run['hf_revision'][:8]}` "
            "(via Neon registry + Hugging Face Hub)"
        )
    elif model_source == "local":
        st.success(f"Loaded checkpoint:\n\n`{CHECKPOINT_PATH.relative_to(ROOT)}`")
    else:
        st.warning(
            "No trained checkpoint found — using the **untrained baseline** "
            "(random head weights). Predictions below are not meaningful yet.\n\n"
            "Run `python main.py train` to produce `models/checkpoint_best.pt` "
            "locally, or push a trained run to the Neon registry (see "
            "`notebooks/kaggle_train_and_push.ipynb` and "
            "`scripts/load_absa_results.py`), then reload this page."
        )

    st.caption(f"Encoder: {config['model']['name']}")
    st.caption(f"Aspects: {', '.join(config['aspects'].values())}")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Live Triage
# ─────────────────────────────────────────────────────────────────────────────

if page == "Live Triage":
    st.title("Live Review Triage")
    st.markdown(
        "Run any review text through the aspect + sentiment model and see "
        "the predicted **aspect** (what the review is about) and **sentiment** "
        "(positive / neutral / negative), each with a confidence breakdown."
    )
    if model_source == "baseline":
        st.info(
            "Running on the untrained baseline — predictions are essentially "
            "random until a checkpoint is trained. This page still works end "
            "to end so you can see the pipeline before training.",
            icon="ℹ️",
        )

    selected = st.selectbox("Quick load example:", list(EXAMPLE_REVIEWS.keys()))
    user_input = st.text_area("Review text:", value=EXAMPLE_REVIEWS[selected], height=120)

    if st.button("🔍 Run Triage", type="primary") and user_input.strip():
        with st.spinner("Running inference…"):
            result = predictor.predict_one(user_input)

        st.divider()
        st.subheader("Result")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Aspect**")
            render_aspect_card(result["aspect"], result["aspect_confidence"])
            render_probs_chart(result["aspect_probs"], ASPECT_CHART_COLORS)
        with col2:
            st.markdown("**Sentiment**")
            render_sentiment_card(result["sentiment"], result["sentiment_confidence"])
            render_probs_chart(result["sentiment_probs"], SENTIMENT_CHART_COLORS)

        st.caption(f"Latency: {result['latency_ms']:.1f}ms")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Example Gallery
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Example Gallery":
    st.title("Example Gallery")
    st.markdown(
        "A batch of curated reviews run through the model, for a quick look "
        "at how it behaves across positive, negative, neutral, and mixed-signal cases."
    )
    if model_source == "baseline":
        st.info(
            "Running on the untrained baseline — the aspect/sentiment columns "
            "below are essentially random until a checkpoint is trained.",
            icon="ℹ️",
        )

    texts = tuple(t for t in EXAMPLE_REVIEWS.values() if t)
    results = run_gallery(predictor, texts)

    render_gallery_table(results)

    st.caption(
        "Want to try your own text? Switch to **Live Triage** in the sidebar."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page: Model Info
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Model Info":
    st.title("Model Info")
    st.markdown(
        "Provenance and evaluation metrics for the model currently loaded above: from "
        "the Neon-backed model registry once a trained run has been pushed, or from a "
        "local `python main.py evaluate` run otherwise."
    )

    def render_metrics_report(metrics: dict) -> None:
        st.subheader("Evaluation metrics")
        st.caption(f"Evaluated on {metrics.get('num_samples', '?')} held-out samples.")

        for task in ("aspect", "sentiment"):
            task_metrics = metrics.get(task)
            if not task_metrics:
                continue
            st.markdown(f"**{task.capitalize()}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{task_metrics['accuracy']:.1%}")
            m2.metric("Macro F1", f"{task_metrics['macro_f1']:.1%}")
            m3.metric("Weighted F1", f"{task_metrics['weighted_f1']:.1%}")
            render_per_class_table(task_metrics["per_class"])
            with st.expander(f"{task.capitalize()} confusion matrix"):
                render_confusion_matrix(
                    task_metrics["confusion_matrix"], list(task_metrics["per_class"].keys())
                )

        st.divider()
        st.subheader("Success criteria")
        st.caption(
            "Project-specific engineering targets (weak-supervision labels, small "
            "dataset, few epochs) — not generic ABSA industry benchmarks. See "
            "config.yaml's monitoring section for the full rationale."
        )
        render_success_criteria(check_targets(metrics, config))

        latency = metrics.get("latency")
        if latency:
            st.markdown("**Latency**")
            l1, l2, l3 = st.columns(3)
            l1.metric("p50", f"{latency['p50_ms']:.1f}ms")
            l2.metric("p95", f"{latency['p95_ms']:.1f}ms")
            l3.metric(
                "p99",
                f"{latency['p99_ms']:.1f}ms",
                delta="SLA met" if latency.get("meets_sla") else "SLA breach",
                delta_color="normal" if latency.get("meets_sla") else "inverse",
            )

    local_metrics_path = ROOT / "results" / "metrics.json"

    if model_source == "neon" and active_run is not None:
        run = active_run
        st.subheader(run["model_version"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Track", run["track"])
        c2.metric("Environment", run["environment"])
        c3.metric("Run timestamp (UTC)", run["run_timestamp"].strftime("%Y-%m-%d %H:%M"))
        st.caption(
            f"Weights: `{run['hf_repo_id']}` @ `{run['hf_revision'][:8]}` "
            f"(`{run['hf_filename']}`)"
        )
        if run.get("notes"):
            st.caption(f"Notes: {run['notes']}")

        st.divider()
        render_metrics_report(run["metrics"])
    elif model_source == "local" and local_metrics_path.exists():
        st.subheader(f"Local checkpoint: `{CHECKPOINT_PATH.relative_to(ROOT)}`")
        st.caption(
            f"From `{local_metrics_path.relative_to(ROOT)}` (produced by "
            "`python main.py evaluate`) — not tracked in the Neon registry."
        )
        st.divider()
        render_metrics_report(json.loads(local_metrics_path.read_text()))
    else:
        st.info(
            "No trained run recorded in the Neon registry, and no local "
            f"`{local_metrics_path.relative_to(ROOT)}` either. Train on Kaggle "
            "(`notebooks/kaggle_train_and_push.ipynb`) and push the results with "
            "`python scripts/load_absa_results.py results.json`, or run "
            "`python main.py train` and `python main.py evaluate` locally, to see "
            "metrics here.",
            icon="ℹ️",
        )
