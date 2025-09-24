!pip install -q pandas google-genai tqdm

import pandas as pd
from tqdm import tqdm
import time

uploaded = files.upload()
prompt_csv_filename = list(uploaded.keys())[0]
df = pd.read_csv(prompt_csv_filename)

if 'Response' not in df.columns:
    df['Response'] = ''

from google import genai
from google.genai import types

API_KEY = "Set your Gemini API Key" 
client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-flash"

for idx in tqdm(range(len(df))):
    prompt = str(df.loc[idx, 'prompt'])
    if pd.notna(df.loc[idx, 'Response']) and df.loc[idx, 'Response'].strip():
        continue
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
        )
        df.loc[idx, 'Response'] = response.text.strip()
        time.sleep(1) 
    except Exception as e:
        df.loc[idx, 'Response'] = f"[ERROR] {e}"


output_filename = "gemini_2_5_flash_responsesM962.csv"
df.to_csv(output_filename, index=False)
files.download(output_filename)
