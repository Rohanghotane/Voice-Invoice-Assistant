import faiss
import os
import pickle
import numpy as np

FAISS_DIR = "./faiss_db"
os.makedirs(FAISS_DIR, exist_ok=True)


INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
META_PATH = os.path.join(FAISS_DIR, "metadata.pkl")


embedding_dim = 384  
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata_store = pickle.load(f)
else:
    index = faiss.IndexFlatL2(embedding_dim)
    metadata_store = []


def save_index():
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata_store, f)


def add_invoice(id: str, text: str, metadata: dict, embed_fn):
    emb = np.array(embed_fn(text)).astype("float32").reshape(1, -1)
    index.add(emb)
    metadata_store.append({
        "id": id,
        "text": text,
        "metadata": metadata
    })
    save_index()


def query_similar(query: str, k=3, embed_fn=None):
    emb = np.array(embed_fn(query)).astype("float32").reshape(1, -1)
    D, I = index.search(emb, k)
    results = []
    for i in I[0]:
        if i < len(metadata_store):
            results.append(metadata_store[i])
    return results
