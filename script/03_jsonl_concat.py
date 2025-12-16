import json
import os
from typing import List

def merge_jsonl_files(input_files: List[str], output_file: str):
    all_data = []
    for i, input_file in enumerate(input_files):
        if not os.path.exists(input_file):
            continue
        if not input_file.endswith('.jsonl'):
            continue
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        all_data.append(data)
                    except json.JSONDecodeError as e:
                        return
                    except Exception as e:
                        return
        except Exception as e:
            continue
            
    if not all_data:
        with open(output_file, 'w', encoding='utf-8') as f:
            pass 
        return    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    except Exception as e:
        return
