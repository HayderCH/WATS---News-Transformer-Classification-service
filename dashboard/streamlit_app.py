"""Streamlit dashboard for the News Topic Intelligence API."""

from __future__ import annotations

import base64
import json
import os
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Add image generation imports
try:
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    IMAGE_GEN_AVAILABLE = True
except ImportError:
    IMAGE_GEN_AVAILABLE = False

# Add chatbot imports (for backwards compatibility, but now using API)
try:
    from app.services.chatbot.local_chatbot import LocalNewsChatbot

    CHATBOT_AVAILABLE = True
    # Create a new instance to ensure we have the latest code (for backwards compatibility)
    chatbot = LocalNewsChatbot()
except ImportError:
    CHATBOT_AVAILABLE = False

DEFAULT_API_BASE = os.getenv("NEWS_API_BASE", "http://localhost:8001")
DEFAULT_API_KEY = os.getenv("NEWS_API_KEY")


@lru_cache(maxsize=1)
def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"accept": "application/json"})
    return session


def _make_headers(api_key: Optional[str], header_name: str) -> Dict[str, str]:
    if not api_key:
        return {}
    return {header_name: api_key}


def _build_url(api_base: str, path: str) -> str:
    base = api_base.rstrip("/") + "/"
    return urljoin(base, path.lstrip("/"))


def call_api(
    method: str,
    path: str,
    api_base: str,
    headers: Dict[str, str],
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    url = _build_url(api_base, path)
    try:
        response = _session().request(
            method,
            url,
            headers=headers,
            timeout=30,
            **kwargs,
        )
    except requests.RequestException as exc:
        st.error(f"Request to {url} failed: {exc}")
        return None

    if response.status_code >= 400:
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        hint = ""
        trailing_segment = api_base.rstrip("/").split("/")[-1]
        if response.status_code == 404 and "/" in trailing_segment:
            hint = " (check base URL/root path)"
        if response.status_code == 401:
            hint = " (provide a valid API key in the sidebar" " to authorize calls)"
        st.error(
            " ".join(
                [
                    method.upper(),
                    path,
                    "returned",
                    str(response.status_code) + ":",
                    str(payload),
                ]
            )
            + hint
        )
        return None

    if not response.content:
        return None

    try:
        return response.json()
    except ValueError:
        st.error("API response was not valid JSON")
        return None


def render_classification_panel(
    api_base: str,
    headers: Dict[str, str],
) -> None:
    st.subheader("Classify an article")
    st.caption("📰 Text-only or 📸 Multimodal classification (text + image)")

    with st.form("classify_form"):
        col1, col2 = st.columns(2)

        with col1:
            title = st.text_input("Title", "Breaking Markets Rally")
            image_url = st.text_input(
                "Image URL (Optional)",
                "",
                help="URL to an image related to the article for multimodal classification",
            )

        with col2:
            uploaded_file = st.file_uploader(
                "Upload Image (Optional)",
                type=["png", "jpg", "jpeg", "gif", "webp"],
                help="Upload an image for multimodal classification",
            )

        text = st.text_area(
            "Article Text",
            (
                "Stocks surged today as markets reacted to strong earnings "
                "reports across the technology sector."
            ),
            height=180,
        )

        submitted = st.form_submit_button("Classify")

    if not submitted:
        return

    if len(text.strip()) < 10:
        st.warning("Please provide at least 10 characters of text.")
        return

    # Display image preview if provided
    display_image = None
    if image_url.strip():
        try:
            # For URL images, we'll show it after API call if successful
            st.info("📸 Image URL provided - will be processed with article")
        except Exception as e:
            st.warning(f"Could not load image from URL: {e}")
    elif uploaded_file is not None:
        try:
            # Display uploaded image preview
            display_image = uploaded_file
            st.image(display_image, caption="Uploaded Image Preview", width=300)
        except Exception as e:
            st.warning(f"Could not display uploaded image: {e}")

    # Prepare payload
    payload = {"title": title.strip() or None, "text": text.strip()}

    # Handle image input
    if image_url.strip():
        payload["image_url"] = image_url.strip()
    elif uploaded_file is not None:
        # Convert uploaded file to base64
        image_bytes = uploaded_file.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        payload["image_base64"] = f"data:{uploaded_file.type};base64,{image_base64}"

    result = call_api(
        "post",
        "/classify_news",
        api_base,
        headers,
        json=payload,
    )
    if not result:
        return

    # Extract multimodal information
    modalities = result.get("modalities", [])
    fusion_used = result.get("fusion_used", False)

    # Display processed image if URL was provided and API succeeded
    if image_url.strip() and "image" in modalities:
        try:
            st.image(image_url, caption="🖼️ Processed Image", width=300)
        except Exception as e:
            st.warning(f"Could not display image from URL: {e}")

    # Display classification results
    st.success(
        f"Top category: {result['top_category']} "
        f"(score {result['confidence_score']:.2f})"
    )

    # Add comparison feature for multimodal results
    if fusion_used and st.button(
        "🔍 Compare with Text-Only",
        help="See how multimodal classification compares to text-only",
    ):
        # Make text-only API call
        text_only_payload = {"title": title.strip() or None, "text": text.strip()}
        text_only_result = call_api(
            "post",
            "/classify_news",
            api_base,
            headers,
            json=text_only_payload,
        )

        if text_only_result:
            st.subheader("📊 Classification Comparison")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🤖 Multimodal Result**")
                st.metric(
                    f"{result['top_category']}",
                    f"{result['confidence_score']:.2f}",
                    help="Combined text + image analysis",
                )

            with col2:
                st.markdown("**📝 Text-Only Result**")
                st.metric(
                    f"{text_only_result['top_category']}",
                    f"{text_only_result['confidence_score']:.2f}",
                    help="Text analysis only",
                )

            # Show if results differ
            if result["top_category"] != text_only_result["top_category"]:
                st.info(
                    "🎯 **Different predictions!** Multimodal analysis changed the classification."
                )
            else:
                confidence_diff = (
                    result["confidence_score"] - text_only_result["confidence_score"]
                )
                if abs(confidence_diff) > 0.1:
                    direction = "increased" if confidence_diff > 0 else "decreased"
                    st.info(
                        f"📈 **Confidence {direction}** by {abs(confidence_diff):.2f} with multimodal analysis"
                    )
                else:
                    st.info(
                        "✅ **Same prediction** - multimodal analysis confirmed text-only result"
                    )

    # Show multimodal information if available
    if modalities:
        modality_icons = {"text": "📝", "image": "🖼️"}
        modality_display = [
            f"{modality_icons.get(mod, mod)} {mod}" for mod in modalities
        ]
        st.info(f"**Modalities used:** {', '.join(modality_display)}")

        if fusion_used:
            st.info("🤖 **Fusion model used** - Combined text and image analysis")
        else:
            st.info("📝 **Text-only classification** - Image processing unavailable")

        # Show per-modality confidence if available
        text_conf = result.get("text_confidence")
        image_conf = result.get("image_confidence")

        if text_conf is not None or image_conf is not None:
            st.subheader("🎯 Confidence Breakdown")
            conf_cols = st.columns(2)

            if text_conf is not None:
                with conf_cols[0]:
                    st.metric("📝 Text Confidence", f"{text_conf:.2f}")
                    st.progress(min(1.0, text_conf))

            if image_conf is not None:
                with conf_cols[1]:
                    st.metric("🖼️ Image Confidence", f"{image_conf:.2f}")
                    st.progress(min(1.0, image_conf))

            # Show fusion impact if both modalities used
            if text_conf is not None and image_conf is not None and fusion_used:
                st.info(
                    "🔄 **Fusion Impact**: Combined analysis may improve accuracy over single modalities"
                )

    categories = result.get("categories", [])
    if categories:
        chart_df = pd.DataFrame(categories)
        chart_df = chart_df.rename(columns={"name": "Category", "prob": "Probability"})
        chart_df["Probability"] = chart_df["Probability"].astype(float)
        fig = px.bar(
            chart_df,
            x="Category",
            y="Probability",
            text="Probability",
            title="Classification probabilities",
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(yaxis_range=[0, 1], margin=dict(t=60, b=20))
        st.plotly_chart(
            fig, use_container_width=True, key="classification_probabilities"
        )

    meta_cols = st.columns(2)
    meta_cols[0].metric("Model Version", result.get("classifier_version", "?"))
    meta_cols[1].metric("Latency (ms)", f"{result.get('latency_ms', 0):.1f}")

    if result.get("suggestion"):
        st.info(result["suggestion"])


def render_summarization_panel(api_base: str, headers: Dict[str, str]) -> None:
    st.subheader("Summarize text")
    with st.form("summarize_form"):
        text = st.text_area(
            "Content",
            (
                "The central bank announced new measures to stabilize the "
                "currency following a period of volatility in international "
                "markets."
            ),
            height=180,
        )
        max_len = st.slider(
            "Max summary length",
            min_value=32,
            max_value=256,
            value=120,
            step=8,
        )
        min_len = st.slider(
            "Min summary length",
            min_value=8,
            max_value=max_len,
            value=25,
            step=1,
        )
        submitted = st.form_submit_button("Summarize")

    if not submitted:
        return

    if len(text.strip()) < 10:
        st.warning("Please provide at least 10 characters of text.")
        return

    payload = {
        "text": text.strip(),
        "max_len": max_len,
        "min_len": min_len,
    }
    result = call_api("post", "/summarize", api_base, headers, json=payload)
    if not result:
        return

    st.write("### Generated Summary")
    st.write(result["summary"])

    meta_cols = st.columns(3)
    meta_cols[0].metric("Model Version", result.get("model_version", "?"))
    meta_cols[1].metric("Latency (ms)", f"{result.get('latency_ms', 0):.1f}")
    meta_cols[2].metric("Cached", "Yes" if result.get("cached") else "No")


def render_trends_panel(api_base: str, headers: Dict[str, str]) -> None:
    st.subheader("Topic trends")
    window = st.slider(
        "Window (days)",
        min_value=1,
        max_value=90,
        value=14,
        step=1,
    )
    data = call_api(
        "get",
        "/trends",
        api_base,
        headers,
        params={"window": window},
    )
    if not data:
        return

    buckets = data.get("buckets", [])
    if buckets:
        df = pd.DataFrame(buckets)
        df["date"] = pd.to_datetime(df["date"])
        fig = px.area(
            df,
            x="date",
            y="count",
            color="label",
            title="Topic volume by day",
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Articles",
            legend_title="Topic",
            margin=dict(t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True, key="trends_time_series")

    totals = data.get("totals", [])
    if totals:
        totals_df = pd.DataFrame(totals)
        totals_df = totals_df.rename(columns={"label": "Topic", "count": "Total"})
        fig_totals = px.bar(
            totals_df,
            x="Topic",
            y="Total",
            text="Total",
            title="Total articles per topic",
        )
        fig_totals.update_traces(
            texttemplate="%{text}",
            textposition="outside",
        )
        fig_totals.update_layout(margin=dict(t=60, b=40))
        st.plotly_chart(fig_totals, use_container_width=True, key="trends_totals")

    st.caption(f"Generated at {data.get('generated_at')}")


def render_forecasting_panel(api_base: str, headers: Dict[str, str]) -> None:
    """Render the time series forecasting panel."""
    st.subheader("🔮 Time Series Forecasting")
    st.caption("Predict future news category trends using hybrid ML/DL models")

    # Check if forecasting service is available
    status_result = call_api(
        "get", "/forecast/POLITICS?days_ahead=1", api_base, headers
    )
    if not status_result:
        st.error(
            "❌ Forecasting service not available. Please ensure " "models are trained."
        )
        st.info(
            "Run `python scripts/manage.py train-forecasting` to " "train the models."
        )

        # Offer to train models
        if st.button("🚀 Train Forecasting Models", type="primary"):
            with st.spinner(
                "🔧 Training forecasting models... " "(this may take several minutes)"
            ):
                train_result = call_api("post", "/forecast/train", api_base, headers)
                if train_result:
                    st.success(
                        "✅ Models trained successfully! "
                        "Refresh the page to use forecasting."
                    )
                    st.rerun()
                else:
                    st.error("❌ Failed to train models. " "Check the server logs.")
        return

    # Category selection
    categories = [
        "POLITICS",
        "WELLNESS",
        "ENTERTAINMENT",
        "TRAVEL",
        "STYLE & BEAUTY",
        "PARENTING",
        "HEALTHY LIVING",
        "QUEER VOICES",
        "FOOD & DRINK",
        "BUSINESS",
    ]

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_category = st.selectbox(
            "Select News Category",
            categories,
            index=0,
            help="Choose a news category to forecast trends for",
        )

    with col2:
        days_ahead = st.slider(
            "Forecast Horizon (days)",
            min_value=1,
            max_value=30,
            value=7,
            help="How many days ahead to forecast",
        )

    # Forecast button
    if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner(
            f"🔮 Forecasting {selected_category} trends for " f"{days_ahead} days..."
        ):
            forecast_data = call_api(
                "get",
                f"/forecast/{selected_category}",
                api_base,
                headers,
                params={"days_ahead": days_ahead},
            )

            if forecast_data:
                # Store forecast in session state for visualization
                st.session_state.current_forecast = forecast_data
                st.session_state.forecast_category = selected_category
                st.session_state.forecast_days = days_ahead

                # Display results
                display_forecast_results(
                    forecast_data, selected_category, days_ahead, key_suffix="new"
                )
            else:
                st.error("❌ Failed to generate forecast. Please try again.")

    # Display previous forecast if available
    if "current_forecast" in st.session_state:
        st.divider()
        st.subheader("📊 Current Forecast")
        display_forecast_results(
            st.session_state.current_forecast,
            st.session_state.forecast_category,
            st.session_state.forecast_days,
            key_suffix="previous",
        )


def display_forecast_results(
    forecast_data: Dict, category: str, days_ahead: int, key_suffix: str = ""
) -> None:
    """Display forecast results with charts and metrics."""
    dates = forecast_data.get("dates", [])
    forecast = forecast_data.get("forecast", [])
    confidence_lower = forecast_data.get("confidence_lower", [])
    confidence_upper = forecast_data.get("confidence_upper", [])
    model_info = forecast_data.get("model_info", {})

    if not dates or not forecast:
        st.error("Invalid forecast data received.")
        return

    # Convert to DataFrame for plotting
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "Forecast": forecast,
            "Lower_Bound": confidence_lower,
            "Upper_Bound": confidence_upper,
        }
    )

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Category", category)

    with col2:
        st.metric("Forecast Days", days_ahead)

    with col3:
        avg_forecast = sum(forecast) / len(forecast)
        st.metric("Avg Daily Articles", f"{avg_forecast:.1f}")

    with col4:
        peak_forecast = max(forecast)
        st.metric("Peak Forecast", f"{peak_forecast:.1f}")

    # Forecast chart
    st.subheader("📈 Forecast Chart")

    fig = px.line(
        df,
        x="Date",
        y="Forecast",
        title=f"{category} News Volume Forecast ({days_ahead} days)",
        markers=True,
    )

    # Add confidence interval
    fig.add_traces(
        [
            px.line(df, x="Date", y="Upper_Bound").data[0],
            px.line(df, x="Date", y="Lower_Bound").data[0],
        ]
    )

    # Update traces for confidence interval
    fig.data[1].line.color = "rgba(0,100,80,0.2)"
    fig.data[1].name = "Upper Bound"
    fig.data[1].showlegend = False

    fig.data[2].line.color = "rgba(0,100,80,0.2)"
    fig.data[2].name = "Lower Bound"
    fig.data[2].showlegend = False

    # Add fill between bounds
    fig.add_traces(
        [
            px.area(df, x="Date", y="Upper_Bound").data[0],
            px.area(df, x="Date", y="Lower_Bound").data[0],
        ]
    )

    fig.data[3].fill = "tonexty"
    fig.data[3].fillcolor = "rgba(0,100,80,0.1)"
    fig.data[3].line.color = "rgba(0,100,80,0.1)"
    fig.data[3].name = "Confidence Interval"
    fig.data[3].showlegend = True

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Predicted Articles",
        margin=dict(t=60, b=20),
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True, key=f"forecast_chart_{key_suffix}")

    # Forecast table
    st.subheader("📋 Detailed Forecast")

    table_df = df.copy()
    table_df["Date"] = table_df["Date"].dt.strftime("%Y-%m-%d")
    table_df = table_df.rename(
        columns={
            "Date": "Date",
            "Forecast": "Articles",
            "Lower_Bound": "Lower 80%",
            "Upper_Bound": "Upper 80%",
        }
    )

    st.dataframe(table_df, use_container_width=True)

    # Model information
    with st.expander("🤖 Model Information", expanded=False):
        st.json(model_info)

        st.markdown(
            """
        **Model Ensemble:**
        - **Prophet**: Statistical forecasting with seasonal decomposition
        - **XGBoost**: Gradient boosting with feature engineering
        - **LSTM**: Deep learning for sequential patterns (GPU accelerated)

        **Confidence Intervals**: 80% prediction intervals based on model uncertainty
        """
        )

    # Export options
    st.subheader("💾 Export Forecast")

    col1, col2 = st.columns(2)

    with col1:
        csv_data = table_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name=f"{category}_forecast_{days_ahead}days.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"download_csv_{key_suffix}",
        )

    with col2:
        json_data = forecast_data
        st.download_button(
            label="📥 Download as JSON",
            data=json.dumps(json_data, indent=2),
            file_name=f"{category}_forecast_{days_ahead}days.json",
            mime="application/json",
            use_container_width=True,
            key=f"download_json_{key_suffix}",
        )


def render_review_panel(api_base: str, headers: Dict[str, str]) -> None:
    st.subheader("Active learning review queue")

    label_catalog = call_api("get", "/labels", api_base, headers)
    if label_catalog and label_catalog.get("labels"):
        labels = label_catalog["labels"]
        label_count = label_catalog.get("count", len(labels))
        st.markdown(f"**Label catalog ({label_count})**")

        def _pretty_label(raw: str) -> str:
            return raw.replace("_", " ").title()

        grouped: Dict[str, list[str]] = defaultdict(list)
        for label in labels:
            head, *tail = label.split("_", 1)
            grouped[head].append(label)

        for family in sorted(grouped):
            members = sorted(grouped[family])
            family_title = _pretty_label(family)
            with st.expander(f"{family_title} ({len(members)})", expanded=False):
                columns = st.columns(3)
                for idx, raw_label in enumerate(members):
                    friendly = _pretty_label(raw_label)
                    columns[idx % len(columns)].markdown(f"- {friendly}")

    stats = call_api("get", "/review/active-learning", api_base, headers)
    if stats:
        cols = st.columns(4)
        cols[0].metric("Labeled review items", stats.get("review_labeled", 0))
        cols[1].metric("Labeled feedback", stats.get("feedback_labeled", 0))
        cols[2].metric("Total examples", stats.get("total_examples", 0))
        cols[3].metric(
            "Ready for training",
            "Yes" if stats.get("ready_for_training") else "No",
        )
        latest = stats.get("latest_label_at")
        if latest:
            st.caption(f"Most recent label: {latest}")
        labels = stats.get("distinct_labels") or []
        if labels:
            st.caption("Distinct labels: " + ", ".join(labels))

    st.write("### Pending queue")
    queue = call_api(
        "get",
        "/review/queue",
        api_base,
        headers,
        params={"limit": 20},
    )
    if not queue:
        st.info("No items currently waiting for review.")
        return

    for item in queue:
        header = (
            f"#{item['id']} • pred={item['predicted_label']} "
            f"• score={item['confidence_score']:.2f}"
        )
        with st.expander(header):
            st.write(item.get("text", ""))
            top_labels = item.get("top_labels") or []
            if top_labels:
                top_df = pd.DataFrame(top_labels)
                top_df = top_df.rename(columns={"name": "Label", "prob": "Probability"})
                top_df["Probability"] = top_df["Probability"].astype(float)
                top_view = (
                    top_df.sort_values("Probability", ascending=False)
                    .head(5)
                    .reset_index(drop=True)
                )
                st.write("Top probabilities")
                fig_top = px.bar(
                    top_view.sort_values("Probability"),
                    x="Probability",
                    y="Label",
                    orientation="h",
                    text="Probability",
                    range_x=[0, 1],
                    title="Top 5 labels",
                )
                fig_top.update_traces(
                    texttemplate="%{text:.2f}", textposition="outside"
                )
                fig_top.update_layout(margin=dict(t=50, b=10))
                st.plotly_chart(
                    fig_top,
                    use_container_width=True,
                    key=f"top-chart-{item['id']}",
                )
                st.dataframe(top_view, use_container_width=True)
            default_label = item.get("predicted_label", "")
            with st.form(f"label_form_{item['id']}"):
                true_label = st.text_input(
                    "True label",
                    value=default_label,
                    help="Provide the human-reviewed category",
                )
                submitted = st.form_submit_button("Submit label")
                if submitted:
                    if not true_label.strip():
                        st.warning("Enter a label before submitting.")
                    else:
                        payload = {
                            "item_id": item["id"],
                            "true_label": true_label.strip(),
                        }
                        resp = call_api(
                            "post",
                            "/review/label",
                            api_base,
                            headers,
                            json=payload,
                        )
                        if resp and resp.get("updated"):
                            st.success("Label saved.")
                            st.rerun()


def render_stream_review_panel(api_base: str, headers: Dict[str, str]) -> None:
    """Render the streaming review panel for human-in-the-loop labeling of streaming articles."""
    st.header("🎬 Stream Review Queue")
    st.caption(
        "Review low-confidence articles from real-time streaming with anomaly detection context"
    )

    # Get streaming-specific stats
    stats = call_api("get", "/review/stats", api_base, headers)
    if stats:
        streaming_total = stats.get("by_source_total", {}).get("streaming", 0)
        streaming_unlabeled = stats.get("by_source_unlabeled", {}).get("streaming", 0)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Streaming Articles to Review", streaming_unlabeled)
        with col2:
            st.metric("Total Streaming Reviews", streaming_total)
        with col3:
            completion_rate = (
                (streaming_total - streaming_unlabeled) / max(1, streaming_total) * 100
                if streaming_total > 0
                else 0
            )
            st.metric("Review Completion", f"{completion_rate:.1f}%")

    st.write("### 📊 Streaming Review Queue")

    # Get streaming review queue
    queue = call_api(
        "get", "/review/stream-queue", api_base, headers, params={"limit": 20}
    )
    if not queue:
        st.info("No streaming articles currently waiting for review.")
        return

    for item in queue:
        # Enhanced header with streaming context
        anomaly_indicator = "🚨 " if item.get("anomaly_score") else ""
        header = (
            f"{anomaly_indicator}#{item['id']} • pred={item['predicted_label']} "
            f"• score={item['confidence_score']:.2f}"
        )
        if item.get("anomaly_score"):
            header += f" • anomaly={item['anomaly_score']:.3f}"

        with st.expander(header):
            # Show streaming context
            if item.get("stream_id"):
                st.caption(f"📡 Stream ID: {item['stream_id']}")

            st.write(item.get("text", ""))

            # Show anomaly information
            if item.get("anomaly_score"):
                st.warning(
                    f"🚨 **Anomaly Detected** (score: {item['anomaly_score']:.3f})"
                )
                st.caption(
                    "This article was flagged by anomaly detection during streaming"
                )

            top_labels = item.get("top_labels") or []
            if top_labels:
                top_df = pd.DataFrame(top_labels)
                top_df = top_df.rename(columns={"name": "Label", "prob": "Probability"})
                top_df["Probability"] = top_df["Probability"].astype(float)
                top_view = (
                    top_df.sort_values("Probability", ascending=False)
                    .head(5)
                    .reset_index(drop=True)
                )
                st.write("Top probabilities")
                fig_top = px.bar(
                    top_view.sort_values("Probability"),
                    x="Probability",
                    y="Label",
                    orientation="h",
                    text="Probability",
                    range_x=[0, 1],
                    title="Top 5 labels",
                )
                fig_top.update_traces(
                    texttemplate="%{text:.2f}", textposition="outside"
                )
                fig_top.update_layout(margin=dict(t=50, b=10))
                st.plotly_chart(
                    fig_top,
                    use_container_width=True,
                    key=f"stream-review-chart-{item['id']}",
                )
                st.dataframe(top_view, use_container_width=True)

            default_label = item.get("predicted_label", "")
            with st.form(f"stream_label_form_{item['id']}"):
                true_label = st.text_input(
                    "True label",
                    value=default_label,
                    help="Provide the human-reviewed category for this streaming article",
                    key=f"stream_true_label_{item['id']}",
                )
                submitted = st.form_submit_button("Submit Stream Review")
                if submitted:
                    if not true_label.strip():
                        st.warning("Enter a label before submitting.")
                    else:
                        payload = {
                            "item_id": item["id"],
                            "true_label": true_label.strip(),
                        }
                        resp = call_api(
                            "post",
                            "/review/label",
                            api_base,
                            headers,
                            json=payload,
                        )
                        if resp and resp.get("updated"):
                            st.success("✅ Streaming article review saved!")
                            st.rerun()


def render_metrics_panel(api_base: str, headers: Dict[str, str]) -> None:
    st.subheader("Service metrics")
    data = call_api("get", "/metrics", api_base, headers)
    if not data:
        return

    summary_cols = st.columns(4)
    summary_cols[0].metric("Backend", data.get("backend", "?"))
    summary_cols[1].metric(
        "Classifier Version",
        data.get("classifier_version", "?"),
    )
    summary_cols[2].metric("Labels", data.get("label_count", 0))
    summary_cols[3].metric("Feedback Entries", data.get("feedback_total", 0))

    latency_payload = data.get("latency_ms") or {}
    if latency_payload:
        st.write("### Latency (ms)")
        latency_rows: list[dict[str, Any]] = []
        for route, stats in latency_payload.items():
            row = {
                "Route": route,
                "Count": int(stats.get("count", 0)),
                "Avg": float(stats.get("avg_ms", 0.0)),
                "p50": float(stats.get("p50_ms", 0.0)),
                "p95": float(stats.get("p95_ms", 0.0)),
            }
            recent = stats.get("recent") or {}
            if recent:
                row["Recent p95"] = float(recent.get("p95_ms", 0.0))
                row["Recent window"] = int(recent.get("window", 0))
            latency_rows.append(row)
        if latency_rows:
            latency_df = pd.DataFrame(latency_rows)
            latency_df = latency_df.sort_values("Avg", ascending=False)
            st.dataframe(latency_df, use_container_width=True)
            fig_latency = px.bar(
                latency_df,
                x="Route",
                y=["Avg", "p95"],
                title="Average vs p95 latency",
                barmode="group",
            )
            fig_latency.update_layout(margin=dict(t=60, b=40))
            st.plotly_chart(
                fig_latency, use_container_width=True, key="metrics_latency"
            )

    counters_payload = data.get("request_counters") or {}
    if counters_payload:
        st.write("### Request counters")
        counter_rows: list[dict[str, Any]] = []
        for key, value in counters_payload.items():
            route, _, status = key.partition(":")
            counter_rows.append(
                {
                    "Route": route,
                    "Status": status or "?",
                    "Count": int(value),
                }
            )
        counter_df = pd.DataFrame(counter_rows)
        counter_df = counter_df.sort_values("Count", ascending=False)
        st.dataframe(counter_df, use_container_width=True)
        fig_counters = px.bar(
            counter_df,
            x="Route",
            y="Count",
            color="Status",
            title="Request volume by route/status",
        )
        fig_counters.update_layout(margin=dict(t=60, b=40))
        st.plotly_chart(fig_counters, use_container_width=True, key="metrics_counters")

    st.write("### Raw Metrics Payload")
    st.json(data)


def render_image_generation_panel(
    api_base: str,
    headers: Dict[str, str],
) -> None:
    """Render the AI image generation panel for news articles."""
    st.subheader("🎨 AI Image Generation")
    st.caption("Generate professional images for news articles using RTX 4060 GPU")

    if not IMAGE_GEN_AVAILABLE:
        st.error(
            "❌ Image generation service not available. Please ensure RTX 4060 setup is complete."
        )
        st.info("Run `python test_rtx_generation.py` to verify your setup.")
        return

    # Check service status
    status_result = call_api("get", "/images/status", api_base, headers)
    if status_result:
        gpu_status = (
            "✅ RTX 4060 Ready"
            if status_result.get("gpu_available")
            else "❌ GPU Not Available"
        )
        st.success(f"Service Status: {gpu_status}")
        if status_result.get("gpu_available"):
            vram = status_result.get("vram_gb", 0)
            st.info(
                f"GPU Memory: {vram:.1f}GB | Typical Generation: "
                f"{status_result.get('typical_generation_time', '3-5s')}"
            )
    else:
        st.warning(
            "⚠️ Image service not responding. Make sure your FastAPI server "
            "includes image routes."
        )

    # Generation mode selection
    mode = st.radio(
        "Generation Mode",
        [
            "📰 Automatic (Article-based)",
            "✏️ Assisted (Prompt Refinement)",
            "🎯 Manual (Full Control)",
        ],
        help="Choose how you want to generate images",
    )

    if mode == "📰 Automatic (Article-based)":
        render_automatic_generation(api_base, headers)
    elif mode == "✏️ Assisted (Prompt Refinement)":
        render_assisted_generation(api_base, headers)
    else:  # Manual
        render_manual_generation(api_base, headers)


def render_automatic_generation(api_base: str, headers: Dict[str, str]) -> None:
    """Automatic image generation based on article content."""
    st.subheader("📰 Automatic Image Generation")

    with st.form("auto_image_form"):
        col1, col2 = st.columns(2)

        with col1:
            title = st.text_input(
                "Article Title",
                "Tesla Unveils Revolutionary Electric Vehicle Technology",
                help="The headline of the news article",
            )

        with col2:
            _ = st.text_input(
                "Article URL (Optional)", "", help="Source URL for reference"
            )

        summary = st.text_area(
            "Article Summary/Content",
            "Tesla CEO Elon Musk announced breakthrough advancements in autonomous driving technology during the AI Safety Summit. The new Full Self-Driving (FSD) system demonstrates unprecedented safety improvements and handles complex urban environments with human-like decision making.",
            height=120,
            help="Key points from the article (used for prompt generation)",
        )

        _ = st.multiselect(
            "Visual Style",
            [
                "Professional",
                "Editorial",
                "News Magazine",
                "Technology Focus",
                "Corporate",
            ],
            default=["Professional"],
            help="Choose visual themes for the generated image",
        )

        submitted = st.form_submit_button("🎨 Generate Image", use_container_width=True)

    if submitted and title.strip():
        with st.spinner("🎨 Generating image with RTX 4060... (3-5 seconds)"):
            # Prepare payload
            payload = {
                "title": title.strip(),
                "summary": summary.strip() if summary.strip() else None,
            }

            result = call_api(
                "post", "/images/generate-news-image", api_base, headers, json=payload
            )

            if result and result.get("success"):
                st.success("✅ Image generated successfully!")
                st.info(f"💾 Saved to: {result.get('image_path', 'Unknown')}")

                # Try to display the image
                image_path = result.get("image_path")
                if image_path and os.path.exists(image_path):
                    st.image(
                        image_path,
                        caption=f"Generated image for: {title}",
                        use_column_width=True,
                    )
                else:
                    st.info(
                        "📸 Image generated but preview not available in dashboard. Check the generated_images folder."
                    )

                # Show generation details
                with st.expander("📊 Generation Details", expanded=False):
                    st.json(result)

            else:
                st.error("❌ Failed to generate image. Check the service logs.")
    elif submitted:
        st.warning("⚠️ Please provide at least an article title.")


def render_assisted_generation(api_base: str, headers: Dict[str, str]) -> None:
    """Assisted generation with prompt suggestions and refinement."""
    st.subheader("✏️ Assisted Image Generation")

    with st.form("assisted_image_form"):
        title = st.text_input("Article Title", "Apple Announces New AI Features")
        summary = st.text_area(
            "Article Summary", "Apple unveiled advanced AI capabilities..."
        )

        # Generate prompt suggestions
        if st.form_submit_button("💡 Generate Prompt Suggestions"):
            if title.strip():
                suggestions = generate_prompt_suggestions(title, summary)
                st.session_state.prompt_suggestions = suggestions
                st.rerun()
            else:
                st.warning("Please enter an article title first.")

    # Show prompt suggestions if available
    if "prompt_suggestions" in st.session_state:
        st.subheader("💡 Suggested Prompts")

        selected_prompt = st.radio(
            "Choose a prompt to use:", st.session_state.prompt_suggestions, index=0
        )

        # Allow editing
        edited_prompt = st.text_area(
            "Edit Prompt (Optional)", selected_prompt, height=80
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            width = st.selectbox("Width", [512, 768, 1024], index=0)
        with col2:
            height = st.selectbox("Height", [512, 768, 1024], index=0)
        with col3:
            steps = st.slider("Quality Steps", 15, 50, 25)

        if st.button("🎨 Generate with Selected Prompt", use_container_width=True):
            with st.spinner(f"🎨 Generating on RTX 4060... (~{3 + steps//10}s)"):
                payload = {
                    "prompt": edited_prompt,
                    "width": width,
                    "height": height,
                    "steps": steps,
                }

                result = call_api(
                    "post", "/images/generate-image", api_base, headers, json=payload
                )

                if result and result.get("success"):
                    st.success("✅ Image generated!")
                    image_path = result.get("image_path")
                    if image_path and os.path.exists(image_path):
                        st.image(
                            image_path, caption="Generated Image", use_column_width=True
                        )
                else:
                    st.error("❌ Generation failed")


def render_manual_generation(api_base: str, headers: Dict[str, str]) -> None:
    """Manual prompt-based image generation."""
    st.subheader("🎯 Manual Image Generation")

    with st.form("manual_image_form"):
        prompt = st.text_area(
            "Prompt",
            "A professional news photograph of a CEO announcing new technology, modern office background, high quality",
            height=100,
        )

        negative_prompt = st.text_area(
            "Negative Prompt (Optional)",
            "blurry, low quality, deformed, ugly, cartoon, anime",
            height=60,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            width = st.selectbox("Width", [512, 768, 1024], index=0)
        with col2:
            height = st.selectbox("Height", [512, 768, 1024], index=0)
        with col3:
            guidance = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)

        steps = st.slider("Steps", 10, 100, 25)
        _ = st.selectbox("Sampler", ["Euler a", "DPM++ 2M Karras", "DDIM"], index=0)

        submitted = st.form_submit_button("🎨 Generate Image", use_container_width=True)

    if submitted and prompt.strip():
        with st.spinner(f"🎨 Generating on RTX 4060... (~{3 + steps//10}s)"):
            payload = {
                "prompt": prompt.strip(),
                "negative_prompt": (
                    negative_prompt.strip() if negative_prompt.strip() else None
                ),
                "width": width,
                "height": height,
                "steps": steps,
                "guidance_scale": guidance,
            }

            result = call_api(
                "post", "/images/generate-image", api_base, headers, json=payload
            )

            if result and result.get("success"):
                st.success("✅ Image generated successfully!")
                image_path = result.get("image_path")
                if image_path and os.path.exists(image_path):
                    st.image(
                        image_path,
                        caption=f"Generated: {prompt[:50]}...",
                        use_column_width=True,
                    )

                    # Download button
                    with open(image_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Image",
                            data=file,
                            file_name=os.path.basename(image_path),
                            mime="image/png",
                        )
                else:
                    st.info("Image generated but preview not available.")
            else:
                st.error("❌ Image generation failed.")
    elif submitted:
        st.warning("Please provide a prompt.")


def generate_prompt_suggestions(title: str, summary: str = "") -> list[str]:
    """Generate prompt suggestions based on article content."""
    base_suggestions = [
        f"Professional editorial illustration of: {title}, news magazine style, high quality",
        f"Modern news photograph representing: {title}, journalistic, detailed",
        f"Corporate presentation slide about: {title}, business professional, clean design",
    ]

    if summary:
        # Add context-aware suggestions
        if any(
            word in summary.lower()
            for word in ["technology", "ai", "digital", "software"]
        ):
            base_suggestions.append(
                f"Technology themed illustration: {title}, futuristic, digital art"
            )
        if any(word in summary.lower() for word in ["business", "economy", "market"]):
            base_suggestions.append(
                f"Business concept visualization: {title}, corporate, professional"
            )
        if any(word in summary.lower() for word in ["politics", "government"]):
            base_suggestions.append(
                f"Political news illustration: {title}, serious, documentary style"
            )

    return base_suggestions


def render_streaming_panel(api_base: str, headers: Dict[str, str]) -> None:
    """Render the streaming control and monitoring panel."""
    st.header("🎬 Real-Time Streaming & Anomaly Detection")
    st.caption(
        "Monitor and control the live news article streaming service with real-time analytics."
    )

    # Stream Control Section
    st.subheader("Stream Control")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("▶️ Start Stream", type="primary", use_container_width=True):
            try:
                rate = st.session_state.get("stream_rate", 1.0)
                response = call_api(
                    "POST", "/streaming/start", api_base, headers, json={"rate": rate}
                )
                if response is not None:
                    st.success("✅ Streaming started successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to start streaming")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    with col2:
        if st.button("⏹️ Stop Stream", type="secondary", use_container_width=True):
            try:
                response = call_api("POST", "/streaming/stop", api_base, headers)
                if response is not None:
                    st.success("✅ Streaming stopped successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to stop streaming")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    with col3:
        # Stream rate control
        rate = st.slider("Rate (articles/sec)", 0.1, 5.0, 1.0, 0.1, key="stream_rate")
        if st.button("⚙️ Update Rate", use_container_width=True):
            try:
                response = call_api(
                    "POST",
                    "/streaming/config/rate",
                    api_base,
                    headers,
                    json={"rate": rate},
                )
                if response is not None:
                    st.success(f"✅ Rate updated to {rate} articles/sec!")
                else:
                    st.error("❌ Failed to update rate")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    with col4:
        if st.button("📊 Refresh Status", use_container_width=True):
            st.rerun()

    # Manual refresh indicator for active streaming
    try:
        status_response = call_api("GET", "/streaming/status", api_base, headers)
        if status_response and status_response.get("active", False):
            st.info(
                "🔄 **STREAMING ACTIVE** - Click 'Refresh Status' to update dashboard data"
            )
    except Exception:
        # Silently handle errors to avoid disrupting the UI
        pass

    # Stream Status Section
    st.subheader("Stream Status")
    try:
        response = call_api("GET", "/streaming/status", api_base, headers)
        if response is not None:
            status_data = response
            col1, col2, col3 = st.columns(3)

            with col1:
                status_color = "🟢" if status_data.get("active", False) else "🔴"
                st.metric(
                    "Status",
                    f"{status_color} {'Active' if status_data.get('active', False) else 'Inactive'}",
                )

            with col2:
                st.metric("Rate", f"{status_data.get('rate', 0):.1f} articles/sec")

            with col3:
                st.metric(
                    "Articles Processed",
                    status_data.get("stats", {}).get("articles_processed", 0),
                )

            # Additional stats
            stats = status_data.get("stats", {})
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Anomalies Detected", stats.get("anomalies_detected", 0))

            with col2:
                categories = stats.get("categories_count", {})
                st.metric("Categories", len(categories))

            with col3:
                start_time = stats.get("start_time")
                if start_time:
                    st.metric(
                        "Uptime",
                        f"{(pd.Timestamp.now() - pd.Timestamp(start_time)).seconds}s",
                    )
                else:
                    st.metric("Uptime", "N/A")

            with col4:
                processing_rate = stats.get("processing_rate", 0)
                st.metric("Processing Rate", f"{processing_rate:.1f} articles/sec")

            # Category Distribution Chart
            st.subheader("📊 Category Distribution")
            categories = stats.get("categories_count", {})
            if categories:
                df_categories = pd.DataFrame(
                    list(categories.items()), columns=["Category", "Count"]
                ).sort_values("Count", ascending=False)

                fig = px.bar(
                    df_categories,
                    x="Category",
                    y="Count",
                    title="Articles by Category",
                    color="Count",
                    color_continuous_scale="viridis",
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categories processed yet.")

            # Confidence Distribution
            st.subheader("🎯 Classification Confidence")
            confidence_dist = stats.get("confidence_distribution", {})
            if confidence_dist:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("High Confidence (≥80%)", confidence_dist.get("high", 0))

                with col2:
                    st.metric(
                        "Medium Confidence (60-80%)", confidence_dist.get("medium", 0)
                    )

                with col3:
                    low_conf = confidence_dist.get("low", 0)
                    st.metric("Low Confidence (<60%)", f"{low_conf} ⚠️")

                # Confidence distribution pie chart
                labels = ["High (≥80%)", "Medium (60-80%)", "Low (<60%)"]
                values = [
                    confidence_dist.get("high", 0),
                    confidence_dist.get("medium", 0),
                    confidence_dist.get("low", 0),
                ]
                colors = ["#00ff00", "#ffff00", "#ff0000"]

                if sum(values) > 0:
                    fig_conf = px.pie(
                        values=values,
                        names=labels,
                        title="Confidence Distribution",
                        color_discrete_sequence=colors,
                    )
                    fig_conf.update_layout(height=300)
                    st.plotly_chart(fig_conf, use_container_width=True)

            # Human-in-the-Loop Learning Section
            st.subheader("🤖 Human-in-the-Loop Learning")
            low_confidence_count = stats.get("low_confidence_count", 0)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Articles Needing Review", low_confidence_count)

            with col2:
                total_processed = stats.get("articles_processed", 0)
                review_percentage = (
                    low_confidence_count / max(1, total_processed)
                ) * 100
                st.metric("Review Rate", f"{review_percentage:.1f}%")

            if low_confidence_count > 0:
                st.info(
                    f"📋 {low_confidence_count} articles have low confidence classifications and may need human review."
                )

            # Real-time Activity Indicator
            st.subheader("⚡ Real-Time Activity")
            if status_data.get("active", False):
                st.success(
                    "🔴 **STREAMING ACTIVE** - Articles are being processed in real-time"
                )
                st.caption(
                    "Click 'Refresh Status' button to update dashboard with latest data"
                )
            else:
                st.info(
                    "⏸️ **STREAMING INACTIVE** - Start streaming to see live updates"
                )

        else:
            st.error("❌ Failed to get status")
    except Exception as e:
        st.error(f"❌ Error getting status: {e}")

    # Placeholder for real-time charts
    # In a production system, this would connect to WebSocket or Server-Sent Events
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Processing Rate")
        # Create sample data for demonstration
        if "stream_history" not in st.session_state:
            st.session_state.stream_history = []

        # Add current data point
        import time

        current_time = pd.Timestamp.now()
        st.session_state.stream_history.append(
            {
                "time": current_time,
                "rate": st.session_state.get("stream_rate", 1.0),
                "articles": len(st.session_state.stream_history),
            }
        )

        # Keep only last 50 points
        st.session_state.stream_history = st.session_state.stream_history[-50:]

        if st.session_state.stream_history:
            df = pd.DataFrame(st.session_state.stream_history)
            fig = px.line(df, x="time", y="rate", title="Streaming Rate Over Time")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Category Distribution")
        # Sample category data
        sample_categories = {
            "Politics": 25,
            "Technology": 20,
            "Sports": 15,
            "Business": 18,
            "Entertainment": 12,
            "Health": 10,
        }

        df = pd.DataFrame(
            list(sample_categories.items()), columns=["Category", "Count"]
        )
        fig = px.pie(df, values="Count", names="Category", title="Article Categories")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Anomaly Detection Section
    st.subheader("🚨 Anomaly Detection")
    try:
        response = call_api("GET", "/streaming/anomalies/stats", api_base, headers)
        if response is not None:
            anomaly_data = response
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Anomalies", anomaly_data.get("total_anomalies", 0))

            with col2:
                st.metric(
                    "Detection Rate", f"{anomaly_data.get('detection_rate', 0):.1f}%"
                )

            with col3:
                st.metric("False Positives", anomaly_data.get("false_positives", 0))

            # Anomaly timeline chart
            if "anomaly_history" in anomaly_data:
                st.subheader("Anomaly Timeline")
                df = pd.DataFrame(anomaly_data["anomaly_history"])
                if not df.empty:
                    fig = px.scatter(
                        df,
                        x="timestamp",
                        y="score",
                        title="Anomaly Scores Over Time",
                        color="is_anomaly",
                        color_discrete_map={True: "red", False: "blue"},
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "Anomaly detection stats not available yet. Start streaming to see data."
            )
    except Exception as e:
        st.info("Anomaly detection stats will appear once streaming is active.")

    # Manual Article Streaming
    st.subheader("📝 Manual Article Streaming")
    with st.expander("Add Custom Article"):
        title = st.text_input("Article Title", key="manual_title")
        text = st.text_area("Article Text", height=100, key="manual_text")
        category = st.selectbox(
            "Expected Category",
            [
                "POLITICS",
                "WELLNESS",
                "ENTERTAINMENT",
                "TRAVEL",
                "STYLE & BEAUTY",
                "PARENTING",
                "HEALTHY LIVING",
                "QUEER VOICES",
                "FOOD & DRINK",
                "BUSINESS",
                "COMEDY",
                "SPORTS",
                "BLACK VOICES",
                "HOME & LIVING",
                "PARENTS",
                "THE WORLDPOST",
                "WEDDINGS",
                "WOMEN",
                "IMPACT",
                "DIVORCE",
                "CRIME",
                "MEDIA",
                "WEIRD NEWS",
                "GREEN",
                "WORLDPOST",
                "RELIGION",
                "TECH",
                "TASTE",
                "MONEY",
                "ARTS",
                "FIFTY",
                "GOOD NEWS",
                "ARTS & CULTURE",
                "ENVIRONMENT",
                "COLLEGE",
                "LATINO VOICES",
                "CULTURE & ARTS",
                "EDUCATION",
            ],
            key="manual_category",
        )

        if st.button("🚀 Stream Article", type="primary"):
            if title and text:
                try:
                    payload = {"title": title, "text": text, "category": category}
                    response = call_api(
                        "POST", "/streaming/article", api_base, headers, json=payload
                    )
                    if response is not None:
                        result = response
                        st.success("✅ Article streamed successfully!")
                        st.json(result)
                    else:
                        st.error("❌ Failed to stream article")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            else:
                st.warning("Please provide both title and text.")


def main() -> None:
    st.set_page_config(
        page_title="News Intelligence Dashboard",
        page_icon="🗞️",
        layout="wide",
    )
    st.title("News Intelligence Dashboard")
    st.caption(
        "Interact with the FastAPI service for classification, "
        "summarization, and trends."
    )

    with st.sidebar:
        st.header("Connection")
        api_base = st.text_input("API Base URL", value=DEFAULT_API_BASE)
        api_key = st.text_input(
            "API Key",
            value=DEFAULT_API_KEY or "",
            type="password",
        )
        header_name = st.text_input(
            "API Key Header",
            value=os.getenv("API_KEY_HEADER", "x-api-key"),
        )

    headers = _make_headers(api_key.strip() or None, header_name.strip())

    tabs = st.tabs(
        [
            "Classify",
            "Summarize",
            "Trends",
            "Forecasting",
            "Review",
            "Stream Review",
            "Metrics",
            "Images",
            "Streaming",
            "Chatbot",
        ]
    )

    with tabs[0]:
        render_classification_panel(api_base, headers)
    with tabs[1]:
        render_summarization_panel(api_base, headers)
    with tabs[2]:
        render_trends_panel(api_base, headers)
    with tabs[3]:
        render_forecasting_panel(api_base, headers)
    with tabs[4]:
        render_review_panel(api_base, headers)
    with tabs[5]:
        render_stream_review_panel(api_base, headers)
    with tabs[6]:
        render_metrics_panel(api_base, headers)
    with tabs[7]:
        render_image_generation_panel(api_base, headers)
    with tabs[8]:
        render_streaming_panel(api_base, headers)
    with tabs[9]:
        render_chatbot_panel(api_base, headers)


def render_chatbot_panel(api_base: str, headers: Dict[str, str]) -> None:
    """Render the chatbot interface panel using REST API."""
    st.subheader("📰 News Chatbot")
    st.markdown("Ask me about news articles, trends, or get help!")

    # Check if chatbot API is available
    health_result = call_api("get", "/chatbot/health", api_base, headers)
    if not health_result or health_result.get("status") != "healthy":
        st.error("Chatbot service is not available. Please check the server logs.")
        return

    # Initialize session state for chatbot
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
        st.session_state.chat_session_id = (
            f"streamlit_{hash(str(st.session_state)) % 10000}"
        )
        # Add welcome message
        welcome_content = (
            "Hello! I'm your news chatbot. I can help you with:\n\n"
            "• Latest news articles and trends\n"
            "• Platform documentation and features\n"
            "• Review queue analytics\n"
            "• General questions about news topics\n\n"
            "What would you like to know?"
        )
        welcome_msg = {"role": "assistant", "content": welcome_content}
        st.session_state.chat_messages.append(welcome_msg)

    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about news..."):
        if not prompt.strip():
            return

        # Add user message to history
        user_msg = {"role": "user", "content": prompt}
        st.session_state.chat_messages.append(user_msg)

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get chatbot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = {
                        "message": prompt,
                        "session_id": st.session_state.chat_session_id,
                    }
                    response = call_api(
                        "post", "/chatbot/chat", api_base, headers, json=payload
                    )

                    if response:
                        st.markdown(response["response"])

                        # Show intent classification
                        if response.get("intent"):
                            st.caption(
                                f"💭 Intent: {response['intent'].replace('_', ' ').title()}"
                            )

                        # Show sources if available
                    if response.get("sources"):
                        num_sources = len(response["sources"])
                        st.markdown(f"### 📚 Sources ({num_sources} articles)")

                        for i, source in enumerate(response["sources"], 1):
                            title = source.get("title", f"Document {i}")
                            with st.expander(f"**{i}. {title}**"):
                                cat = source.get("category", "N/A")
                                st.caption(f"Category: {cat}")

                                if source.get("short_description"):
                                    desc = source["short_description"]
                                    # Show preview (first 200 chars)
                                    if len(desc) > 200:
                                        preview = desc[:200] + "..."
                                        st.text(preview)

                                        # Show expand option based on category
                                        if cat == "Documentation":
                                            # Use button to show/hide full content (more reliable than checkbox in expander)
                                            if st.button(
                                                "📖 Show Full Content",
                                                key=f"show_full_{i}",
                                            ):
                                                full_content = source.get(
                                                    "full_content", desc
                                                )
                                                st.text_area(
                                                    "Full Documentation",
                                                    full_content,
                                                    height=400,
                                                    key=f"full_content_{i}",
                                                )
                                        else:
                                            # For non-documentation sources, show read more button
                                            if st.button(
                                                "Read Full Article",
                                                key=f"read_more_{i}",
                                            ):
                                                st.text_area(
                                                    "Full Article",
                                                    desc,
                                                    height=300,
                                                    key=f"full_article_{i}",
                                                )
                                    else:
                                        # Short description, show directly
                                        if cat == "Documentation" and source.get(
                                            "full_content"
                                        ):
                                            st.text_area(
                                                "Full Documentation",
                                                source["full_content"],
                                                height=300,
                                                key=f"full_content_{i}",
                                            )
                                        else:
                                            st.text(desc)

                                    # Show additional metadata for all sources
                                    if source.get("authors"):
                                        authors = source["authors"]
                                        if isinstance(authors, list):
                                            author_text = ", ".join(authors)
                                        else:
                                            author_text = str(authors)
                                        st.caption(f"Authors: {author_text}")
                                    if source.get("date"):
                                        date_str = source["date"]
                                        st.caption(f"Date: {date_str}")
                                    if source.get("link"):
                                        link = source["link"]
                                        st.markdown(f"[Read Original]({link})")
                                else:
                                    st.text("No description available.")

                            st.markdown("---")

                    else:
                        error_msg = (
                            "Sorry, I couldn't get a response from the chatbot service."
                        )
                        st.error(error_msg)
                        response = {"response": error_msg}

                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    response = {"response": error_msg}

        # Add assistant response to history
        response_text = (
            response["response"]
            if "response" in locals() and response
            else "Sorry, I encountered an error."
        )
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": response_text}
        )

    # Chatbot statistics and session info
    with st.expander("📊 Chatbot Statistics", expanded=False):
        stats_result = call_api("get", "/chatbot/stats", api_base, headers)
        if stats_result:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Messages", stats_result.get("total_messages", 0))
            with col2:
                st.metric("Active Sessions", stats_result.get("active_sessions", 0))
            with col3:
                avg_latency = stats_result.get("average_latency_ms", 0)
                st.metric("Avg Response Time", f"{avg_latency:.1f}ms")

            # Intent distribution
            if stats_result.get("intent_distribution"):
                st.subheader("Intent Distribution")
                intents = stats_result["intent_distribution"]
                if intents:
                    intent_df = pd.DataFrame(
                        list(intents.items()), columns=["Intent", "Count"]
                    ).sort_values("Count", ascending=False)

                    fig = px.bar(
                        intent_df,
                        x="Intent",
                        y="Count",
                        title="Chatbot Intent Classification",
                        color="Count",
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

        # Session info
        if "chat_session_id" in st.session_state:
            st.caption(f"Session ID: {st.session_state.chat_session_id}")

            # Option to view conversation history
            if st.button("📜 View Full Conversation History"):
                history_result = call_api(
                    "get",
                    f"/chatbot/history/{st.session_state.chat_session_id}",
                    api_base,
                    headers,
                )
                if history_result and history_result.get("conversations"):
                    st.subheader("Conversation History")
                    for conv in history_result["conversations"]:
                        with st.expander(f"Message {conv.get('id', 'N/A')}"):
                            st.write(f"**User:** {conv.get('user_message', 'N/A')}")
                            st.write(
                                f"**Assistant:** {conv.get('assistant_response', 'N/A')}"
                            )
                            st.caption(
                                f"Intent: {conv.get('intent', 'N/A')} | Time: {conv.get('timestamp', 'N/A')}"
                            )
                else:
                    st.info("No conversation history found for this session.")

    # Feedback collection
    if st.session_state.chat_messages and len(st.session_state.chat_messages) > 1:
        with st.expander("💬 Provide Feedback", expanded=False):
            feedback_rating = st.slider(
                "How helpful was this conversation?",
                min_value=1,
                max_value=5,
                value=3,
                help="1 = Not helpful, 5 = Very helpful",
            )

            feedback_text = st.text_area(
                "Additional comments (optional)",
                height=80,
                placeholder="What did you like or dislike about the responses?",
            )

            if st.button("Submit Feedback", type="primary"):
                feedback_payload = {
                    "session_id": st.session_state.chat_session_id,
                    "rating": feedback_rating,
                    "comments": (
                        feedback_text.strip() if feedback_text.strip() else None
                    ),
                    "timestamp": pd.Timestamp.now().isoformat(),
                }

                feedback_result = call_api(
                    "post",
                    "/chatbot/feedback",
                    api_base,
                    headers,
                    json=feedback_payload,
                )

                if feedback_result:
                    st.success("✅ Thank you for your feedback!")
                else:
                    st.error("❌ Failed to submit feedback. Please try again.")


if __name__ == "__main__":
    main()
