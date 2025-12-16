import os
import time
import json
from openai import AzureOpenAI
from typing import Dict, Any, List

def get_azure_client(config: Dict[str, Any]) -> AzureOpenAI:
    """Initializes and returns the AzureOpenAI client."""
    
    api_key = os.getenv("AZURE_OPENAI_API_KEY", config['azure_api']['api_key'])
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", config['azure_api']['endpoint'])
    
    if not api_key or not endpoint:
        raise ValueError("Azure OpenAI API key and endpoint must be provided either in config.yaml or as environment variables.")

    client = AzureOpenAI(
        api_version=config['azure_api']['api_version'],
        azure_endpoint=endpoint,
        api_key=api_key,
    )
    return client

def generate_with_azure(
    client: AzureOpenAI,
    prompt: str,
    system_prompt: str,
    config: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Generates a response from Azure OpenAI with retry logic and expects a JSON list output.

    Args:
        client (AzureOpenAI): The initialized Azure OpenAI client.
        prompt (str): The user prompt.
        system_prompt (str): The system prompt to guide the model's behavior.
        config (Dict[str, Any]): The application configuration dictionary.

    Returns:
        List[Dict[str, str]]: A list of typed viewpoints, e.g., [{'viewpoint': '...', 'role': '...'}]
                               Returns an empty list if generation fails or output is invalid.
    """
    max_retries = config['azure_api']['max_retries']
    retry_delay = config['azure_api']['retry_delay']

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config['azure_api']['deployment_name'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=config['preprocessing']['llm_temperature'],
                max_tokens=config['preprocessing']['llm_max_tokens'],
                timeout=config['azure_api']['request_timeout'],
                response_format={"type": "json_object"} 
            )

            content = response.choices[0].message.content
            parsed_json = json.loads(content)
            
            if isinstance(parsed_json, dict):
                for key, value in parsed_json.items():
                    if isinstance(value, list):
                        return value 
                print(f"Warning: LLM returned a JSON object but no list was found inside: {content}")
                return []

            elif isinstance(parsed_json, list):
                return parsed_json

            else:
                print(f"Warning: LLM output was not a JSON list or a dict containing a list: {content}")
                return []

        except Exception as e:
            print(f"Error calling Azure OpenAI API (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print("Error: Max retries reached. Failed to get a valid response from Azure OpenAI.")
                return [] 
    return []