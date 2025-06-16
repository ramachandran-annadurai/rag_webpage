import os
import uuid
import json
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.models import HnswConfigDiff
from functools import lru_cache
import requests

# === Configuration ===
collection_name = "csv_data_embeddings"
model_name = "all-MiniLM-L6-v2"
QDRANT_URL = "https://53937321-3ec5-4a4f-b96d-48299a9267f0.us-west-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.-GE64pnFobmMIXgG8IcyrhTf6OM9KrFA3phaqgZ1Bo8"  # Replace with your actual key


def initialize_system(use_gpu=False):
    print(f"🧠 Loading embedding model on device: cpu")
    model = SentenceTransformer(model_name)

    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False,
        https=True,
        timeout=30.0,
        check_compatibility=False
    )

    vector_size = model.get_sentence_embedding_dimension()

    if not qdrant_client.collection_exists(collection_name):
        print("📦 Creating collection...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
                on_disk=True
            ),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )

    print(f"✅ Qdrant ready with collection '{collection_name}'")
    return model, qdrant_client


@lru_cache(maxsize=1000)
def query_deepseek(prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "deepseek-r1:8b",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        print("🔍 DeepSeek raw response:", result)

        # Some Ollama models may return 'response' or 'output' or use stream tokens
        return result.get("response") or result.get("output") or "⚠️ No output field"

    except Exception as e:
        print("❌ DeepSeek error:", str(e))
        return "⚠️ LLM error"


def rag_query(model, qdrant_client, query: str, top_k=5, context_token_limit=1500) -> str:
    query_embedding = model.encode([query])[0].tolist()

    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    # Filter by score threshold manually
    filtered_results = [r for r in results if r.score and r.score >= 0.2]

    if not filtered_results:
        return "🤷 No relevant results found."

    context = "\n".join([r.payload['text'] for r in filtered_results])
    context = context[:context_token_limit]

    prompt = f"""
You are a medical data assistant. Use only the provided context below to answer the question. 
Do not use external knowledge or make assumptions.

Context:
{context}

Question:
Based on the patient's characteristics, classify the pregnancy risk level in one word and use one of the following: Low, Medium, or High.

Patient characteristics:
{query}

## Instructions ##
1. Output one word only: Low, Medium, or High.
2. No extra text or explanation.

## OUTPUT ##
"""
    print("📝 Final Prompt Sent to DeepSeek:\n", prompt)
    return query_deepseek(prompt)
