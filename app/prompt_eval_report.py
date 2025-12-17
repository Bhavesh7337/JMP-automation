"""
streamlit viewer for one eval report. quick.
"""

import json
import os
import streamlit as st


st.set_page_config(page_title="Prompt Eval Report", page_icon="📊", layout="wide")
st.title("Prompt Evaluation Report")

default_path = "output/prompt_eval_report.json"
path = st.text_input("Report JSON path", value=default_path)

if not path or not os.path.exists(path):
    st.warning("Provide a valid report JSON path.")
    st.stop()

with open(path, "r", encoding="utf-8") as f:
    report = json.load(f)

summary = report.get("summary", {})
categories = summary.get("categories", {})

col1, col2, col3 = st.columns(3)
col1.metric("Overall pass rate", summary.get("overall_pass_rate", 0))
col2.metric("Mean risk", summary.get("mean_risk", 0))
col3.metric("Patched pass rate", report.get("patched_overall_pass_rate", "n/a"))

st.subheader("Category pass rates")
cat_names = list(categories.keys())
cat_rates = [categories[c]["pass_rate"] for c in cat_names]
st.bar_chart({"category": cat_names, "pass_rate": cat_rates}, x="category", y="pass_rate")

st.subheader("Failures")
failures = [
    {
        "id": r["id"],
        "category": r["category"],
        "preview": r["metrics"].get("output_preview", ""),
        "risk": r["metrics"].get("risk_score", 0),
    }
    for r in report.get("results", [])
    if not r["metrics"].get("passed")
]
if failures:
    st.dataframe(failures, use_container_width=True)
else:
    st.success("All tests passed 🎉")
