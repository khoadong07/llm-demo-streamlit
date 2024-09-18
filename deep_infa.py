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

def call_deep_infra(raw_content):
    chat_completion = openai.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[
        {
          "role": "user",
          "content": load_prompt_file("prompt.txt")
        },
          {
              "role": "user",
              "content": f"Return only in JSON format. Using the provided knowledge, please process the following content: {raw_content}. The result must be formatted as JSON with corresponding field names and values. If the returned result is not in the correct JSON format, it must be reformatted to ensure it is always correct."}
      ])

    result_llm = chat_completion.choices[0].message.content
    print(result_llm)
    result_llm = parse_custom_json_string(result_llm)
    prompt_token = chat_completion.usage.prompt_tokens
    output_token = chat_completion.usage.completion_tokens

    return result_llm, prompt_token, output_token


def parse_custom_json_string(custom_string):
    try:
        match = re.search(r'```json(.*?)```', custom_string, re.DOTALL)
        if not match:
            return {"error": "No JSON content found"}

        json_data = match.group(1).strip()

        parsed_result = json.loads(json_data)
        return parsed_result

    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON, reformatting required"}

    except AttributeError:
        return {"error": "No valid JSON structure found"}
