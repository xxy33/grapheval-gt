import os
import sys
import json
from tqdm import tqdm
from typing import Dict, Any, List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config_loader import load_config
from src.azure_llm_api import get_azure_client, generate_with_azure

def get_viewpoint_extraction_prompt(abstract_text: str) -> tuple[str, str]:
    """
    Creates the system and user prompts for the viewpoint extraction task.
    """
    system_prompt = """You are an expert AI assistant specializing in scientific text analysis. Your task is to analyze the following research abstract.
1. Decompose the abstract into a list of fine-grained, independent viewpoints. A viewpoint is a single, concise idea, argument, or finding.
2. For each viewpoint, classify its primary role into one of the following categories: ["Background", "Problem", "Method", "Result", "Contribution"].

Your response MUST be a valid JSON object containing a single key "viewpoints" which holds a list of objects. Each object must have two keys: "viewpoint" (the text) and "role" (the classified role).
"""
    user_prompt = f"""Please process the following abstract:

--- ABSTRACT START ---
{abstract_text}
--- ABSTRACT END ---
"""
    return system_prompt, user_prompt

def process_dataset(
    dataset_name: str,
    config: Dict[str, Any],
    client,
):
    """
    Processes a single raw dataset file to extract typed viewpoints.

    Args:
        dataset_name (str): The key of the dataset in the config (e.g., 'iclr_papers').
        config (Dict[str, Any]): The project configuration.
        client: The initialized Azure OpenAI client.
    """
    input_path = config['dataset_paths'][dataset_name]
    
    output_dir = "data/intermediate"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}_viewpoints.jsonl")
    
    print(f"\nProcessing dataset: '{dataset_name}'")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")

    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    data = json.loads(line)
                    if 'paper_id' in data:
                        processed_ids.add(data['paper_id'])
                except json.JSONDecodeError:
                    continue # Skip corrupted lines
        print(f"Found {len(processed_ids)} already processed papers. Resuming...")

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'a', encoding='utf-8') as f_out: 

        papers_to_process = []
        for line in f_in:
            paper_data = json.loads(line)
            if paper_data['paper_id'] not in processed_ids:
                papers_to_process.append(paper_data)
        
        if not papers_to_process:
            print("All papers have already been processed. Nothing to do.")
            return

        for paper_data in tqdm(papers_to_process, desc=f"Extracting viewpoints for {dataset_name}"):
            abstract = paper_data.get('abstract', '')

            if not abstract or not abstract.strip():
                print(f"Skipping paper_id {paper_data['paper_id']} due to empty abstract.")
                continue

            system_prompt, user_prompt = get_viewpoint_extraction_prompt(abstract)

            typed_viewpoints = generate_with_azure(
                client=client,
                prompt=user_prompt,
                system_prompt=system_prompt,
                config=config
            )

            paper_data['typed_viewpoints'] = typed_viewpoints

            f_out.write(json.dumps(paper_data) + '\n')

    print(f"Finished processing for '{dataset_name}'. Results saved to {output_path}")

def main():
    """
    Main execution function.
    """
    print("--- Starting API-based Viewpoint Extraction Script ---")
    
    try:
        config = load_config()
    except Exception as e:
        print(f"Failed to load configuration. Error: {e}")
        return

    try:
        client = get_azure_client(config)
        print("Azure OpenAI client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Azure OpenAI client. Error: {e}")
        print("Please check your API key and endpoint in 'configs/config.yaml' or environment variables.")
        return

    dataset_names = list(config['dataset_paths'].keys())
    if not dataset_names:
        print("No datasets found in 'configs/config.yaml' under 'dataset_paths'.")
        return

    # Process each dataset
    for name in dataset_names:
        process_dataset(name, config, client)

    print("\n--- All datasets processed. ---")

if __name__ == '__main__':
    main()