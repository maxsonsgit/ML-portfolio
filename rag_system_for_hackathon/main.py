import os
import pickle

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm


CSV_WEBSITES = "websites_updated.csv"
CSV_QUESTIONS = "questions_clean.csv"
FAISS_INDEX_PATH = "faiss_index.bin"
EMBEDDINGS_PATH = "chunk_embeddings.npy"
CHUNK_INFO_PATH = "chunk_info.pkl"
OUTPUT_PATH = "submission.csv"

CHUNK_SIZE = 256
BATCH_SIZE = 64
TOP_K_INITIAL = 20
TOP_K_FINAL = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    print(f"Loading models on {DEVICE}...")
    bi_encoder = SentenceTransformer("intfloat/multilingual-e5-large", device=DEVICE)
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEVICE)
    return bi_encoder, cross_encoder


def build_chunks(csv_path: str) -> tuple[list[str], list[dict]]:
    df = pd.read_csv(csv_path)
    df["full_text"] = df["title"].fillna("") + " " + df["text"].fillna("")

    chunks, chunk_info = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        for i in range(0, len(row["full_text"]), CHUNK_SIZE):
            chunk = row["full_text"][i : i + CHUNK_SIZE]
            chunks.append(chunk)
            chunk_info.append({"web_id": row["web_id"], "chunk_text": chunk})

    return chunks, chunk_info


def build_and_save_index(bi_encoder: SentenceTransformer) -> tuple[faiss.Index, list[dict]]:
    chunks, chunk_info = build_chunks(CSV_WEBSITES)

    print("Encoding chunks...")
    embeddings = bi_encoder.encode(
        chunks,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    np.save(EMBEDDINGS_PATH, embeddings)
    with open(CHUNK_INFO_PATH, "wb") as f:
        pickle.dump(chunk_info, f)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

    return index, chunk_info


def load_index() -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNK_INFO_PATH, "rb") as f:
        chunk_info = pickle.load(f)
    return index, chunk_info


def encode_questions(bi_encoder: SentenceTransformer, questions: list[str]) -> np.ndarray:
    return bi_encoder.encode(
        ["query: " + q for q in questions],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")


def rerank(
    cross_encoder: CrossEncoder,
    df_questions: pd.DataFrame,
    indices: np.ndarray,
    chunk_info: list[dict],
) -> list[dict]:
    results = []
    for i, idx_list in enumerate(tqdm(indices, desc="Re-ranking")):
        query_text = df_questions.iloc[i]["query"]
        pairs = [(query_text, chunk_info[idx]["chunk_text"]) for idx in idx_list]
        scores = cross_encoder.predict(pairs)

        top = sorted(zip(scores, idx_list), key=lambda x: x[0], reverse=True)[:TOP_K_FINAL]
        top_web_ids = [chunk_info[idx]["web_id"] for _, idx in top]

        results.append({
            "q_id": df_questions.iloc[i]["q_id"],
            "web_list": "[" + ", ".join(map(str, top_web_ids)) + "]",
        })

    return results


def main():
    bi_encoder, cross_encoder = load_models()

    index_exists = os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNK_INFO_PATH)
    if index_exists:
        print("Loading existing FAISS index...")
        index, chunk_info = load_index()
    else:
        print("Building FAISS index from scratch...")
        index, chunk_info = build_and_save_index(bi_encoder)

    df_questions = pd.read_csv(CSV_QUESTIONS)
    df_questions["query"] = df_questions["query"].astype(str)

    print("Encoding questions...")
    question_embeddings = encode_questions(bi_encoder, df_questions["query"].tolist())

    print("Retrieving top candidates...")
    _, indices = index.search(question_embeddings, TOP_K_INITIAL)

    print("Reranking...")
    results = rerank(cross_encoder, df_questions, indices, chunk_info)

    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
    print(f"Done! Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()