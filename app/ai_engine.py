"""
ai_engine.py — Generates valid JMP JSL scripts from natural-language instructions.
Fixes incorrect dataset paths and removes code fences/triple quotes automatically.
"""

import os
import re
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


client = OpenAI()

# Functions 

def _normalize_path_for_jmp(path: str) -> str:
    
    abs_path = os.path.abspath(path)
    return abs_path.replace("\\", "/")

def _sanitize_jsl(text: str) -> str:

    if not text:
        return ""

    # Remove ```jsl ... ``` or ``` fences
    text = re.sub(r"^```[\w-]*\s*|\s*```$", "", text, flags=re.MULTILINE)

    # Remove triple quotes
    text = text.strip().strip("'''").strip('"""').strip()

    # Remove accidental markdown annotations like ```JSL
    text = text.replace("```JSL", "").replace("```jsl", "").replace("```", "")

    return text.strip()

with open("app/jmp_docs_snipets.txt", "r", encoding="utf-8") as f:
    docs_context = f.read()

# === Main Function ===

def prompt_to_jsl(prompt: str, data_path: str, column_names: list[str]) -> str:
    """
    Convert a natural language instruction into runnable JMP JSL code.
    Ensures dataset path correctness and strips invalid syntax.
    """
    # Normalize path for JMP
    jmp_path = _normalize_path_for_jmp(data_path)

    
    system_prompt = f"""
You are an expert in JMP Scripting Language (JSL).
Write ONLY valid, minimal JSL code — no markdown, no backticks, no explanations.
The dataset to use is located at: {jmp_path}
Available columns: {', '.join(column_names)}.
If the user's request involves plotting or analysis, include:
  dt = Open("{jmp_path}");
Follow JMP syntax precisely and end all statements with semicolons.
Use the following syntax reference examples to ensure valid code:
{docs_context}
"""

    try:
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": prompt.strip()}
            ],
            temperature=0.2,
        )

      
        raw_jsl = response.choices[0].message.content or ""
        jsl_code = _sanitize_jsl(raw_jsl)

        
        if "Open(" not in jsl_code:
            jsl_code = f'dt = Open("{jmp_path}");\n' + jsl_code
        else:
            
            jsl_code = re.sub(
                r'Open\((["\']).*?\1\)',
                f'Open("{jmp_path}")',
                jsl_code,
                count=1,
                flags=re.IGNORECASE
            )

        return jsl_code.strip()

    except Exception as e:
        return f"/*  Error calling OpenAI API: {e} */"



