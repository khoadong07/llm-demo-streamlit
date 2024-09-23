import requests
import json
import re
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

deep_infra_key = os.getenv('DEEP_INFRA_KEY')
deep_infra_url = os.getenv('DEEP_INFRA_URL')

openai = OpenAI(
    api_key=deep_infra_key,
    base_url=deep_infra_url,
)


def load_prompt_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def call_deep_infra(raw_context, raw_content):
    chat_completion = openai.chat.completions.create(
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        messages=[
            {
                "role": "user",
                "content": f"Dựa trên ngữ cảnh sau {raw_context} hãy phân tích nội dung {raw_content}. The result should be formatted as JSON with field names and corresponding values."
            },
            {
                "role": "user",
                "content": load_prompt_file("prompt.txt")
            }
      ])

    result = chat_completion.choices[0].message.content

    result_llm = parse_custom_json_string(result)
    if result_llm is None:
        result_llm = extract_json_from_string(result)
    prompt_token = chat_completion.usage.prompt_tokens
    output_token = chat_completion.usage.completion_tokens

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
