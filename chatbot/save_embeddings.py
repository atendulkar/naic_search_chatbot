import sys
import os
import numpy as np
import openai
import hashlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chatbot.live_search import crawl_site

openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"

OUTPUT_DIR = "naic_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_openai_embedding(text):
    """Embed a single document using OpenAI."""
    text = text.strip() or "empty"
    response = openai.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return np.array(response.data[0].embedding, dtype=np.float32)

def sanitize_filename(url):
    """Generate a unique filename using a hash of the URL."""
    return hashlib.md5(url.encode()).hexdigest()

def save_embedding_for_each_document(pairs):
    """Create and save embeddings per document."""
    mapping = []

    for i, (text, url) in enumerate(pairs):
        if not text.strip():
            print(f"[{i}] Skipping empty content from {url}")
            continue

        try:
            print(f"[{i}] Embedding: {url}")
            embedding = get_openai_embedding(text)
            filename = sanitize_filename(url) + ".npy"
            filepath = os.path.join(OUTPUT_DIR, filename)
            np.save(filepath, embedding)
            mapping.append((filename, url))
        except Exception as e:
            print(f"[{i}] Failed for {url}: {e}")

    # Save URL → file mapping
    with open(os.path.join(OUTPUT_DIR, "mapping.txt"), "w", encoding="utf-8") as f:
        for filename, url in mapping:
            f.write(f"{filename}\t{url}\n")

    print(f"Done. Saved {len(mapping)} embeddings.")

# Main logic
if __name__ == "__main__":
    # This should return [(text1, url1), (text2, url2), ...]
    texts, urls = crawl_site()
    document_pairs = list(zip(texts, urls))

    if not document_pairs:
        raise ValueError("crawl_site() returned no documents.")

    save_embedding_for_each_document(document_pairs)
