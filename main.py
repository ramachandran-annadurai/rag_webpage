from fastapi import FastAPI
from pydantic import BaseModel
from rag_module import initialize_system, rag_query

app = FastAPI()

# Load once
model, qdrant_client = initialize_system(use_gpu=False)

class RAGInput(BaseModel):
    query: str

@app.post("/rag")
def rag_endpoint(rag_input: RAGInput):
    result = rag_query(model, qdrant_client, rag_input.query)
    return {"response": result}
