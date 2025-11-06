import os
from .ai_engine import prompt_to_jsl
from .jmp_connector import run_jsl_script
from .utils import get_column_names, ensure_dir, log

def run_workflow(user_prompt: str, data_path: str):
    """Full workflow: generate → save → run JSL."""

    log("Starting JMP Copilot workflow...")

    #  Extract columns from CSV
    columns = get_column_names(data_path)
    log(f"Found {len(columns)} columns: {', '.join(columns)}")

    #  Generate JSL code
    log("Generating JSL code using GPT-4o-mini...")
    jsl_code = prompt_to_jsl(user_prompt, data_path, columns)

    # Save script
    output_dir = "output"
    ensure_dir(output_dir)
    jsl_path = os.path.join(output_dir, "generated_script.jsl")
    with open(jsl_path, "w", encoding="utf-8") as f:
        f.write(jsl_code)
    log(f"JSL script saved to: {jsl_path}")

    # Run script in JMP
    log("Sending JSL script to JMP...")
    run_jsl_script(jsl_path)

    log(" Workflow completed successfully!")
