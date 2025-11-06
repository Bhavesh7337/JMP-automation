import os
from ai_engine import prompt_to_jsl
from jmp_connector import run_jsl_script

def main():
    """Main entry point for JMP Copilot."""
   
    user_prompt = "Create a scatter plot of efficiency versus temperature"
    data_path = r"D:\AI and ML lrn\week 4\data\processed\combined_data.csv"
    columns = ["temperature", "efficiency", "voltage", "current"]

    print(" Generating JMP script...")
    jsl_code = prompt_to_jsl(user_prompt, data_path, columns)

    print("\n Generated JSL Code:\n")
    print(jsl_code)

    # --- Save script ---
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    jsl_path = os.path.join(output_dir, "generated_script.jsl")

    with open(jsl_path, "w", encoding="utf-8") as f:
        f.write(jsl_code)
    print(f"\n Saved JSL script at: {jsl_path}")

    # --- Run it in JMP ---
    run_jsl_script(jsl_path)

    print("\n Done! Check JMP for your plot or analysis.")

if __name__ == "__main__":
    main()


