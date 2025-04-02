from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
import requests

app = FastAPI()

EMBEDDING_ENDPOINT = "http://tei:8080"
GENERATION_ENDPOINT = "http://tgi:8080/generate"

embed_model = TextEmbeddingsInference(model_url=EMBEDDING_ENDPOINT)
parser = SimpleNodeParser()
index = None

class UploadPayload(BaseModel):
    texts: List[str]

class MessagePayload(BaseModel):
    new_message: Dict[str, str]

@app.post("/upload")
async def upload_context(payload: UploadPayload):
    global index
    documents = [Document(text=t) for t in payload.texts]
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    return {"message": f"Uploaded {len(nodes)} chunks."}

@app.post("/generate")
async def generate(payload: MessagePayload):
    global index
    if index is None:
        return {"error": "Index not loaded."}

    query = payload.new_message["content"]
    retriever = index.as_retriever(similarity_top_k=3)
    retrieved_nodes = retriever.retrieve(query)
    context = "\n\n".join([node.text for node in retrieved_nodes])

    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    response = requests.post(GENERATION_ENDPOINT, json={"inputs": prompt})
    
    if response.status_code != 200:
        return {"error": f"LLM failed: {response.status_code}", "details": response.text}

    return {
        "generated_text": response.json().get("generated_text", ""),
        "contexts": [node.text for node in retrieved_nodes]
    }
