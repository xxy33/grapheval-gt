import os
import json

def fix_jsonl_encoding(input_path: str, output_path: str):
    """
    Reads a .jsonl file, decodes Unicode escapes, and writes to a new file.

    Args:
        input_path (str): The path to the file with Unicode escapes.
        output_path (str): The path to save the fixed file.
    """
    print(f"Fixing file: {input_path}")
    print(f"Saving to: {output_path}")

    try:
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping corrupted JSON line: {line.strip()}")
                    continue
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        print("Fixing complete.")
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
    except Exception as e:
        print(f"An error occurred: {e}")