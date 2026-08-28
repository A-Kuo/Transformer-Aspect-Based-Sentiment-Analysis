"""
app.py — ABSA Triage Dashboard

Brings the main cohort's `src.inference.Predictor` (BERT aspect + sentiment
heads, see README) into an interactive dashboard: browse curated example
reviews and their extracted aspect + sentiment, or run your own text through
the model live.

Run from the repo root:
    streamlit run app.py

Uses a trained checkpoint at models/checkpoint_best.pt if one exists
(produced by `python main.py train`); otherwise falls back to the untrained
baseline, same as notebooks/01_inference_walkthrough.ipynb, with a clear
banner so predictions aren't mistaken for a trained model's output.
"""

import html
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.inference import Predictor

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

@st.cache_resource(show_spinner=False)
def load_predictor():
    if CHECKPOINT_PATH.exists():
        predictor = Predictor.from_checkpoint(
            str(CHECKPOINT_PATH), config_path=str(CONFIG_PATH)
        )
        return predictor, True
    predictor = Predictor.from_pretrained(config_path=str(CONFIG_PATH))
    return predictor, False


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
        bar.progress(20, text="Loading BERT tokenizer + encoder (first run downloads weights)…")
        predictor, has_checkpoint = load_predictor()
        bar.progress(80, text="Model ready. Rendering dashboard…")
        time.sleep(0.1)
        bar.progress(100, text="Done.")
        time.sleep(0.15)
    placeholder.empty()
    return predictor, has_checkpoint


predictor, has_checkpoint = initialize_with_progress()
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


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🎯 ABSA Triage Dashboard")
    st.caption("BERT aspect + sentiment model")
    st.divider()

    st.markdown("**NAVIGATION**")
    page = st.radio(
        "Select page:", ["Live Triage", "Example Gallery"], label_visibility="collapsed"
    )
    st.divider()

    if has_checkpoint:
        st.success(f"Loaded checkpoint:\n\n`{CHECKPOINT_PATH.relative_to(ROOT)}`")
    else:
        st.warning(
            "No trained checkpoint found — using the **untrained baseline** "
            "(random head weights). Predictions below are not meaningful yet.\n\n"
            "Run `python main.py train` to produce `models/checkpoint_best.pt`, "
            "then reload this page."
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
    if not has_checkpoint:
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
    if not has_checkpoint:
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
