
import os
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity

openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-4"
EMBEDDING_DIR = "naic_embeddings_chunked"
METADATA_FILE = os.path.join(EMBEDDING_DIR, "chunks_metadata.txt")

def get_openai_embedding(text):
    response = openai.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return np.array(response.data[0].embedding, dtype=np.float32)

def load_embeddings():
    embeddings, metadata = [], []
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            filename, url, preview = parts
            npy_file = os.path.join(EMBEDDING_DIR, f"{filename}.npy")
            if os.path.exists(npy_file):
                emb = np.load(npy_file)
                embeddings.append(emb)
                metadata.append({"url": url, "preview": preview})
    return np.array(embeddings), metadata

def search_top_chunks(query, top_k=5):
    query_emb = get_openai_embedding(query).reshape(1, -1)
    embeddings, metadata = load_embeddings()
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    return [ {**metadata[i], "score": sims[i]} for i in top_indices ]

def summarize_chunks(chunks, query):
    combined = "\n\n".join(c["preview"] for c in chunks)
    prompt = f"Summarize the following in relation to: '{query}'\n\n{combined}"
    response = openai.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a legal document summarization expert."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

def search_and_summarize(query):
    top_chunks = search_top_chunks(query)
    summary = summarize_chunks(top_chunks, query)
    sources = list({chunk['url'] for chunk in top_chunks})
    return summary, sources
