import os
import pandas as pd
import requests
import json
from time import sleep
from datetime import datetime

# Set your uploaded and save file path
repo_path = r"Your file path"
prompt_csv_path = os.path.join(repo_path, "prompt file") 
response_folder = os.path.join(repo_path, "feedback file path") 
response_csv_path = os.path.join(response_folder, "feedback file")
temp_csv_path = os.path.join(response_folder, "temp_progress.csv") #If there is a large amount of data, it is recommended to store it during the process
os.makedirs(response_folder, exist_ok=True)

# ⚙️ configure setting
MAX_RETRIES = 3
RETRY_DELAY = 5
REQUEST_INTERVAL = 1.5
BATCH_SAVE_INTERVAL = 20
TIMEOUT = 100
API_KEY = "set your DeepSeek API key"
MODEL_NAME = "deepseek/deepseek-r1-0528"
#MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b"

try:
    df = pd.read_csv(prompt_csv_path, encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv(prompt_csv_path, encoding='latin1')

df = df.reset_index(drop=True)

if os.path.exists(temp_csv_path):
    temp_df = pd.read_csv(temp_csv_path, encoding='utf-8-sig')
    responses = temp_df["deepseek_response"].tolist()
    start_idx = len(responses)
else:
    responses = []
    start_idx = 0

def query_deepseek(prompt, model=MODEL_NAME):
    url = "https://api.deepseek.com" # you can also use https://api.deepseek.com/v1 as the base_ur
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2048, # since deepseek r1/qwen have "think" function, you may want to set a larger max_token
        "top_p": 1.0
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=TIMEOUT)
            response.raise_for_status()

            if response.status_code == 429:
                wait_time = int(response.headers.get('Retry-After', RETRY_DELAY * (attempt + 1)))
                sleep(wait_time)
                continue

            response_json = response.json()
            return response_json['choices'][0]['message']['content'].strip()


for i in range(start_idx, len(df)):
    prompt = df.at[i, "prompt"]
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}]  [{i+1}/{len(df)}] Prompt: {prompt[:80]}...")

    response = query_deepseek(prompt)
    responses.append(response)
    print(f"Response: {response[:100]}...")

    if (i + 1) % BATCH_SAVE_INTERVAL == 0 or i == len(df) - 1:
        df_temp = df.iloc[:i+1].copy()
        df_temp["deepseek_response"] = responses
        df_temp.to_csv(temp_csv_path, index=False, encoding='utf-8-sig')

    sleep(REQUEST_INTERVAL)

df["deepseek_response"] = responses
df.to_csv(response_csv_path, index=False, encoding='utf-8-sig')
