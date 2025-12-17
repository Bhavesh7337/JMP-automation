"""
streamlit diff view: drop in two JSON reports and see which prompt did better.
kept super simple on purpose.
"""

import json
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Prompt Diff (Behavior)", page_icon="🪄", layout="wide")
st.title("Prompt Diff — Behavioral Outcomes")


def load_report(file):
    try:
        return json.load(file)
    except Exception:
        return None


def get_metric(summary: dict, key: str, default=0.0):
    return summary.get(key, default) if summary else default


def cat_pass_rate(summary: dict, category: str):
    if not summary:
        return 0.0
    categories = summary.get("categories", {})
    if category not in categories:
        return 0.0
    return categories[category].get("pass_rate", 0.0)


col_a, col_b = st.columns(2)
with col_a:
    file_a = st.file_uploader("Before report (JSON)", type=["json"], key="before")
with col_b:
    file_b = st.file_uploader("After report (JSON)", type=["json"], key="after")

if not file_a or not file_b:
    st.info("Upload two prompt eval reports to compare.")
    st.stop()

report_a = load_report(file_a)
report_b = load_report(file_b)
if not report_a or not report_b:
    st.error("Failed to parse one or both reports.")
    st.stop()

summary_a = report_a.get("summary", {})
summary_b = report_b.get("summary", {})

mean_risk_a = get_metric(summary_a, "mean_risk")
mean_risk_b = get_metric(summary_b, "mean_risk")
pass_a = get_metric(summary_a, "overall_pass_rate")
pass_b = get_metric(summary_b, "overall_pass_rate")

halluc_rate_a = 1 - cat_pass_rate(summary_a, "hallucination")
halluc_rate_b = 1 - cat_pass_rate(summary_b, "hallucination")
clarify_rate_a = cat_pass_rate(summary_a, "clarification")
clarify_rate_b = cat_pass_rate(summary_b, "clarification")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean risk (before)", f"{mean_risk_a:.3f}")
col2.metric("Mean risk (after)", f"{mean_risk_b:.3f}", f"{mean_risk_b-mean_risk_a:+.3f}")
col3.metric("Overall pass (before)", f"{pass_a:.3f}")
col4.metric("Overall pass (after)", f"{pass_b:.3f}", f"{pass_b-pass_a:+.3f}")

col5, col6 = st.columns(2)
col5.metric("Hallucination rate (before)", f"{halluc_rate_a:.3f}")
col6.metric("Hallucination rate (after)", f"{halluc_rate_b:.3f}", f"{halluc_rate_b-halluc_rate_a:+.3f}")

col7, col8 = st.columns(2)
col7.metric("Clarification rate (before)", f"{clarify_rate_a:.3f}")
col8.metric("Clarification rate (after)", f"{clarify_rate_b:.3f}", f"{clarify_rate_b-clarify_rate_a:+.3f}")

st.subheader("Category pass-rate sparkline")
cats = sorted(set(summary_a.get("categories", {}).keys()) | set(summary_b.get("categories", {}).keys()))
data = []
for c in cats:
    data.append(
        {
            "category": c,
            "before": cat_pass_rate(summary_a, c),
            "after": cat_pass_rate(summary_b, c),
        }
    )
df = pd.DataFrame(data)
st.line_chart(df.set_index("category"))

st.subheader("Failure previews (after)")
failures_after = [
    {
        "id": r["id"],
        "category": r["category"],
        "risk": r["metrics"].get("risk_score"),
        "preview": r["metrics"].get("output_preview", "")[:120],
    }
    for r in report_b.get("results", [])
    if not r["metrics"].get("passed")
]
if failures_after:
    st.dataframe(failures_after, use_container_width=True)
else:
    st.success("No failures in after report.")
