# Install the required libraries.
!pip install openai pandas scipy numpy openpyxl tqdm

import openai
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from getpass import getpass

# Set OpenAI API Key
openai_api_key = getpass("Enter your OpenAI API Key: ")
openai.api_key = openai_api_key

# Function to retrieve text embeddings
def get_embedding(text, model="text-embedding-3-large"):
    response = openai.embeddings.create(
        input=text,
        model=model,
        encoding_format="float"
    )
    return np.array(response.data[0].embedding)

# Load Excel file (Please replace with your actual file path)
# excel_path = list(uploaded.keys())[0]
df = pd.read_csv('Your file')

# Extract text columns
texts_a = df.iloc[:, 0].astype(str).tolist()
texts_b = df.iloc[:, 1].astype(str).tolist()

# Generate embeddings
with tqdm(total=len(texts_a), desc="Embedding A") as pbar_a:
    embeddings_a = []
    for text in texts_a:
        embeddings_a.append(get_embedding(text))
        pbar_a.update(1)
    embeddings_a = np.array(embeddings_a)

with tqdm(total=len(texts_b), desc="Embedding B") as pbar_b:
    embeddings_b = []
    for text in texts_b:
        embeddings_b.append(get_embedding(text))
        pbar_b.update(1)
    embeddings_b = np.array(embeddings_b)




'''
If you have more than 2 groups of data, you can increase the following code accordingly.
    texts_c = df.iloc[:, 2].astype(str).tolist()

with tqdm(total=len(texts_c), desc="Embedding C") as pbar_c:
    embeddings_c = []
    for text in texts_c:
        embeddings_a.append(get_embedding(text))
        pbar_c.update(1)
    embeddings_c = np.array(embeddings_c)

    texts_d = df.iloc[:, 4].astype(str).tolist()
    ...
'''
