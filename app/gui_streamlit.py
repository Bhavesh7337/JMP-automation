import streamlit as st
import pandas as pd
import os
from datetime import datetime
from utils import get_column_names, ensure_dir
from jmp_connector import run_jsl_script
from rag_engine import prompt_to_jsl_rag  # NEW
from ai_engine import prompt_to_jsl       # keep non-RAG as fallback

st.set_page_config(page_title="JMP Copilot", page_icon="||")

st.title("JMP Copilot (RAG)")
use_rag = st.toggle("Use JSL manual RAG")

# Dataset
src = st.radio("Dataset source", ["Upload CSV", "Local path"], horizontal=True)
data_path, columns = "", []
if src == "Upload CSV":
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        os.makedirs("data", exist_ok=True)
        data_path = os.path.abspath(os.path.join("data", up.name))
        pd.read_csv(up).to_csv(data_path, index=False)
        columns = pd.read_csv(data_path, nrows=1).columns.tolist()
        st.success(f"Saved: {data_path}")
else:
    p = st.text_input("Local CSV path")
    if p and os.path.exists(p):
        data_path = os.path.abspath(p)
        columns = pd.read_csv(data_path, nrows=1).columns.tolist()
        st.success("Dataset found.")

user_prompt = st.text_area("Instruction", height=120, placeholder="e.g., Create a scatter of efficiency vs temperature with a linear fit")
gen = st.button(" Generate JSL")
run = st.button(" Run in JMP")

output_dir = "output"; ensure_dir(output_dir)



if 'latest_jsl_path' not in st.session_state:

    st.session_state['latest_jsl_path'] = ''



if gen:

    if not user_prompt.strip():

        st.error("Enter an instruction.")

    elif not data_path:

        st.error("Provide a dataset.")

    else:

        with st.spinner("Generating with RAG..." if use_rag else "Generating..."):

            if use_rag:

                jsl = prompt_to_jsl_rag(user_prompt, data_path, columns)

            else:

                from ai_engine import prompt_to_jsl  # local import to avoid circulars

                jsl = prompt_to_jsl(user_prompt, data_path, columns)



            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            prompt_slug = "".join(filter(str.isalnum, user_prompt.replace(" ", "_")[:50]))

            filename = f"{timestamp}_{prompt_slug}.jsl"

            jsl_path = os.path.join(output_dir, filename)



            with open(jsl_path, "w", encoding="utf-8") as f:

                f.write(jsl)

            

            st.session_state['latest_jsl_path'] = jsl_path



        st.code(jsl, language="jsl")

        st.success(f"Saved: {os.path.abspath(jsl_path)}")



if run:

    jsl_path = st.session_state.get('latest_jsl_path')

    if not jsl_path or not os.path.exists(jsl_path):

        st.error("Generate first.")

    else:

        with st.spinner("Sending to JMP..."):

            run_jsl_script(jsl_path)

        st.success("Sent to JMP. Check JMP.")
