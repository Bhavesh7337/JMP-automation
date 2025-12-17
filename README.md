JMP Copilot

AI assistant that converts plain English into runnable JMP Scripting Language (JSL) and executes it directly in JMP.
Includes a JSL-specific prompt evaluation framework with risk scoring, failure logging, self-patching, and before/after behavioral diffs for prompt iteration.

Highlights

Natural language → JSL
Describe plots or analyses in English and receive executable JSL.

Streamlit UI + CLI
Upload a CSV or provide a local path, generate JSL, and send it to JMP.

Optional RAG support
Use a local JSL manual as retrieval context for more accurate syntax and APIs.

Prompt evaluation suite (JSL-focused)
Behavioral tests, hallucination detection, clarification checks, structural validation, risk scoring, failure corpus, and self-patch loop.

Prompt diffing
Compare two evaluation runs to see behavioral changes (risk ↓, hallucination ↓, clarification ↑) with visual summaries.

Project Layout (Key Components)
app/
├── ai_engine.py            # Base GPT-4o JSL generator (no RAG)
├── rag_engine.py           # RAG-augmented JSL generator
├── rag_build_index.py      # Build vector index from JSL manual PDF
├── gui_streamlit.py        # Streamlit UI
├── workflow.py             # CLI workflow (generate + send to JMP)
├── jmp_connector.py        # COM launcher for JMP with JSL file
├── utils.py                # Small helpers
├── prompt_eval_suite.py    # JSL behavioral evaluation runner
├── prompt_eval_report.py   # Streamlit viewer for one eval report
├── prompt_eval_diff.py     # Streamlit before/after behavioral diff
data/                       # Sample or uploaded CSVs (git-ignored)
output/                     # Generated JSL + eval reports (git-ignored)

Setup
Clone & Install
git clone https://github.com/<your-username>/JMP-automation.git
cd JMP-automation
pip install -r requirements.txt

Environment

Create a .env file:

OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxx

(Optional) Build RAG Index

Use your JSL manual (PDF) as retrieval context:

python app/rag_build_index.py "path/to/JSL_Manual.pdf"


This creates:

app/rag_index/

Run the App
Streamlit UI
streamlit run app/gui_streamlit.py


Upload a CSV or provide a local path

Enter a natural-language instruction

Generate JSL and send it to JMP

CLI Demo
python app/main.py


Uses the base ai_engine. Adjust prompt, data path, and columns directly in the file.

Prompt Evaluation (JSL-Only)
Run Evaluation + Save Report
python app/prompt_eval_suite.py \
  --output-json output/prompt_eval_report.json \
  --rag-index app/rag_index


Optional:

--system-prompt <file_or_text>

What It Checks

Invented JMP APIs or columns

Clarification behavior on ambiguous prompts

Prompt drift stability

Plain JSL output (no Markdown, no prose)

Structural validity (Open(), semicolons, runnable code)

Self-refinement after known failures

Instruction hierarchy enforcement

RAG similarity & length deltas

Risk scoring

Failures are logged to:

output/failures.jsonl


If outputs indicate invention, a self-patch rerun is automatically reported.

View Evaluation Results
Single Report Viewer
streamlit run app/prompt_eval_report.py


(Default: output/prompt_eval_report.json)

Before / After Prompt Diff
streamlit run app/prompt_eval_diff.py


Upload two evaluation reports to see:

Mean risk delta

Pass-rate delta

Hallucination rate (1 − hallucination pass rate)

Clarification rate

Per-category sparklines

Failure comparisons (post-prompt change)

How It Works (High-Level Flow)

User prompt + dataset context → LLM (GPT-4o)

(Optional) RAG retrieves relevant JSL manual snippets

LLM generates plain JSL only

Output is cleaned (no fences, valid paths, semicolons enforced)

Script is saved to output/

JMP is launched with the generated JSL

-----------------------------------------------------------------------------------------------
Troubleshooting
_______________________________________________________________________________________________

JMP path
Update JMP_PATH in app/jmp_connector.py if JMP is installed elsewhere.

Missing RAG index
Either build it or run without --rag-index.

Eval JSON not found
Point the Streamlit viewers to the correct file path.