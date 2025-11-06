import subprocess
import os

# Path to your JMP executable — adjust this if your path differs
JMP_PATH = r"C:\Program Files\JMP\JMPSTUDENT\19\jmp.exe"

def run_jsl_script(jsl_path: str):
    """
    Execute a JSL script using JMP.

    Parameters:
    - jsl_path: The absolute path to the JSL script file.
    """
    if not os.path.exists(JMP_PATH):
        print(f" Error: JMP executable not found at: {JMP_PATH}")
        return

    if not os.path.exists(jsl_path):
        print(f" Error: JSL script not found at: {jsl_path}")
        return

    print(f" Running JSL script: {jsl_path}")
    subprocess.Popen([JMP_PATH, jsl_path])
    print(" Script sent to JMP! Check JMP window for results.")
