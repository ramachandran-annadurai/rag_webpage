import openai
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import HnswConfigDiff
import os

# === Configuration ===
collection_name = "csv_data_embeddings"
model_name = "all-MiniLM-L6-v2"
QDRANT_URL = "https://53937321-3ec5-4a4f-b96d-48299a9267f0.us-west-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.-GE64pnFobmMIXgG8IcyrhTf6OM9KrFA3phaqgZ1Bo8"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4"

openai.api_key = OPENAI_API_KEY

def initialize_system(use_gpu=False):
    print("🧠 Initializing model and Qdrant...")
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
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
                on_disk=True
            ),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )
    return model, qdrant_client

@lru_cache(maxsize=1000)
def query_chatgpt(prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a medical assistant helping classify pregnancy risk."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print("❌ OpenAI error:", str(e))
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
Based on the patient's characteristics, classify the pregnancy risk level in one word (Low, Medium, or High) and explain briefly why.

Patient characteristics:
{query}

## Instructions ##
1. Begin your answer with: <think>
2. Inside <think>, explain your thought process — what factors you're analyzing and why.
3. After </think>, provide the final risk level classification: one word only (Low, Medium, or High).
4. Do not add extra explanation outside <think>.

## OUTPUT ##
"""
    return query_chatgpt(prompt)
