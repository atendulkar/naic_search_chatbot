import os
import sys
import numpy as np
import openai
import hashlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chatbot.live_search import crawl_site

openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"
OUTPUT_DIR = "naic_embeddings_chunked"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def get_openai_embeddings(texts):
    response = openai.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [np.array(item.embedding, dtype=np.float32) for item in sorted_data]

def sanitize_filename(url):
    return hashlib.md5(url.encode()).hexdigest()

def save_embeddings_for_documents(pairs):
    metadata_path = os.path.join(OUTPUT_DIR, "chunks_metadata.txt")
    all_meta = []

    for text, url in pairs:
        if not text.strip():
            continue
        chunks = chunk_text(text)
        try:
            embeddings = get_openai_embeddings(chunks)
            base_name = sanitize_filename(url)
            for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{base_name}_chunk{i}"
                npy_path = os.path.join(OUTPUT_DIR, f"{chunk_id}.npy")
                np.save(npy_path, emb)
                preview = chunk_text.replace("\n", " ")[:1000]
                all_meta.append(f"{chunk_id}\t{url}\t{preview}")
        except Exception as e:
            print(f"Error for {url}: {e}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_meta))
    print(f"Saved {len(all_meta)} chunks.")

if __name__ == "__main__":
    texts, urls = crawl_site()
    save_embeddings_for_documents(zip(texts, urls))
