import requests
import json
import re
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

deep_infra_key = os.getenv('DEEP_INFRA_KEY')
deep_infra_url = os.getenv('DEEP_INFRA_URL')
fireworks_key = os.getenv('FIREWORKS_KEY')

openai = OpenAI(
    api_key=deep_infra_key,
    base_url=deep_infra_url,
)


def load_prompt_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def call_deep_infra(raw_context, raw_content):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    payload = {
        "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "max_tokens": 16384,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.1,
        "messages": [{
                "role": "user",
                "content": load_prompt_file("prompt.txt")
            },
            {
                "role": "user",
                "content": f"""
                Main_content: "{raw_content}"
                Context: "{raw_context}"
                """
            }]
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {fireworks_key}"
    }
    # requests.request("POST", url, headers=headers, data=json.dumps(payload))

    # chat_completion = openai.chat.completions.create(
    #     model="meta-llama/Meta-Llama-3-70B-Instruct",
    #     messages=[
    #
    #   ])
    chat_completion = requests.request("POST", url, headers=headers, data=json.dumps(payload))

    result = json.loads(chat_completion.text).get('choices')[0].get('message')
    print(result.get('content'))
    result_llm = parse_custom_json_string(result.get('content'))
    if result_llm is None:
        result_llm = extract_json_from_string(result.get('content'))

    token_ = json.loads(chat_completion.content).get('usage')
    prompt_token = token_.get('prompt_tokens', None)
    output_token = token_.get('completion_tokens', None)

    return result_llm, prompt_token, output_token


def parse_custom_json_string(custom_string):
    try:
        json_data = re.search(r'```(.*?)```', custom_string, re.DOTALL).group(1).strip()
        parsed_result = json.loads(json_data)
        return parsed_result
    except (json.JSONDecodeError, AttributeError) as e:
        return None


import json
import re


def extract_json_from_string(text):
    json_pattern = r'\{[\s\S]*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    if matches:
        for match in matches:
            try:
                data = json.loads(match)
                return data
            except json.JSONDecodeError:
                return None
    return None
