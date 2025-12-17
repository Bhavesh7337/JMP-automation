import subprocess
import os

# path to your JMP exe - change if yours is elsewhere
JMP_PATH = r"C:\Program Files\JMP\JMPSTUDENT\19\jmp.exe"

def run_jsl_script(jsl_path: str):
    """Fire a JSL script into JMP. Nothing fancy, just spawns the exe.
    jsl_path = absolute path to the script."""
    if not os.path.exists(JMP_PATH):
        print(f" Error: JMP executable not found at: {JMP_PATH}")
        return

    if not os.path.exists(jsl_path):
        print(f" Error: JSL script not found at: {jsl_path}")
        return

    print(f" Running JSL script: {jsl_path}")
    subprocess.Popen([JMP_PATH, jsl_path])
    print(" Script sent to JMP! Check JMP window for results.")
