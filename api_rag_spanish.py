import os
import logging
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.core.node_parser.text import SentenceWindowNodeParser

try:
    from llama_index.core import Document, VectorStoreIndex
except ImportError:
    from llama_index.readers.schema.base import Document
    # Fallback: use default index if needed
    from llama_index.vector_stores.simple import SimpleVectorStoreIndex as VectorStoreIndex

from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, pipeline

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI application
app = FastAPI()

# ==========================
# CONFIGURATION: TEXT EMBEDDINGS INFERENCE (TEI)
# ==========================
tei_model_name = os.getenv("TEI_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
logging.info(f"Using embedding model: {tei_model_name}")

Settings.llm = None  # No language model is set directly in settings
Settings.embed_model = TextEmbeddingsInference(
    model_name=tei_model_name,
    base_url="http://tei:80",  # TEI service endpoint
    embed_batch_size=32
)

# ==========================
# CONFIGURATION: TEXT GENERATION INFERENCE (TGI)
# ==========================
logging.info("Configuring TGI client for text generation...")
generator = InferenceClient("http://tgi:80")
tgi_model_name = os.getenv("TGI_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
logging.info(f"Using generative model (TGI): {tgi_model_name}")
logging.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(tgi_model_name)

# ==========================
# OPTIONAL: Cross-Encoder Reranker
# ==========================
try:
    cross_encoder = pipeline("text-classification", model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    logging.info("Cross-encoder initialized for reranking.")
except Exception as e:
    logging.warning("Cross-encoder pipeline could not be initialized; reranking disabled.")
    cross_encoder = None

# Global vector index variable
index = None

class UploadRequest(BaseModel):
    texts: list[str]

def improved_chunking(documents, window_size=2, overlap=1):
    """
    Improved chunking using adjustable window size and overlap.
    """
    # Create a SentenceWindowNodeParser with tuned parameters
    sentence_window = SentenceWindowNodeParser(window_size=window_size, overlap=overlap)
    nodes = sentence_window.build_window_nodes_from_documents(documents)
    return nodes

def build_faiss_index(documents):
    """
    Build a FAISS-based vector store index for faster retrieval.
    This function attempts to use FAISS if available.
    """
    try:
        from llama_index.vector_stores.faiss import FaissVectorStore
        # Build FAISS vector store from documents using the current embedding model
        vector_store = FaissVectorStore.from_documents(documents, embed_model=Settings.embed_model)
        # Wrap the FAISS vector store in the VectorStoreIndex interface
        return VectorStoreIndex(vector_store=vector_store)
    except Exception as e:
        logging.warning("FAISS vector store not available or failed; falling back to standard index.")
        # Fallback: use improved_chunking to build nodes and create index
        nodes = improved_chunking(documents)
        return VectorStoreIndex(nodes)

@app.post("/upload")
async def upload_documents(req: UploadRequest):
    """
    Endpoint to create a vector index from a list of texts.
    """
    try:
        documents = [Document(text=text) for text in req.texts]
        if not documents:
            raise HTTPException(status_code=400, detail="No texts were received for indexing.")
        # Use improved chunking
        nodes = improved_chunking(documents)
        global index
        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir="./index_storage")
        return {"message": "Vector index successfully created", "nodes_count": len(nodes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while creating index: {e}")

@app.post("/upload_dataset")
async def upload_dataset():
    """
    Endpoint to load and index the SQAC dataset from Hugging Face.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise HTTPException(status_code=500, detail="The 'datasets' library is not installed.")
    try:
        dataset = load_dataset("avacaondata/sqac_fixed")
        texts = []
        total = len(dataset["train"])
        for i, record in enumerate(dataset["train"]):
            parts = []
            if "question" in record and record["question"]:
                parts.append("Question: " + record["question"])
            if "context" in record and record["context"]:
                parts.append("Context: " + record["context"])
            if "answers" in record and record["answers"]:
                if isinstance(record["answers"], dict) and "text" in record["answers"]:
                    answer_text = ", ".join(record["answers"]["text"])
                    parts.append("Answers: " + answer_text)
                else:
                    parts.append("Answers: " + str(record["answers"]))
            if parts:
                texts.append("\n".join(parts))
            if i % 100 == 0:
                logging.info(f"Processed {i}/{total} records")
        if not texts:
            raise HTTPException(status_code=400, detail="No data extracted from the dataset.")
        documents = [Document(text=text) for text in texts]
        # Build index using FAISS if possible, otherwise fallback
        global index
        index = build_faiss_index(documents)
        index.storage_context.persist(persist_dir="./index_storage")
        return {"message": "Vector index successfully created from dataset", "nodes_count": len(documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while creating index from dataset: {e}")

@app.post("/generate")
async def generate_text(request: Request):
    """
    Endpoint to generate responses using Retrieval-Augmented Generation (RAG).
    """
    global index
    try:
        data = await request.json()
        new_message = data.get("new_message", {})
        if "content" not in new_message:
            raise HTTPException(status_code=400, detail="The attribute 'content' is missing in 'new_message'.")
        if index is None:
            storage_context = StorageContext.from_defaults(persist_dir="./index_storage")
            index = load_index_from_storage(storage_context)
        # Retrieve candidates using the index (FAISS-based if available)
        query_engine = index.as_query_engine(streaming=False, similarity_top_k=10)
        nodes_retrieved = query_engine.retrieve(new_message["content"])
        # Optional reranking using cross-encoder
        if cross_encoder is not None and nodes_retrieved:
            candidate_texts = [node.text for node in nodes_retrieved]
            rerank_inputs = [(new_message["content"], text) for text in candidate_texts]
            rerank_scores = cross_encoder(rerank_inputs)
            # Re-sort nodes by score (assume higher score is better)
            nodes_retrieved = [node for score, node in sorted(zip([r["score"] for r in rerank_scores], nodes_retrieved), key=lambda x: x[0], reverse=True)]
        docs = "".join([f"<doc>\n{node.text}</doc>" for node in nodes_retrieved])
        system_prompt = "You are an assistant that responds strictly based on the provided document information."
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "input", "content": docs},
                new_message
            ],
            add_generation_prompt=True,
            tokenize=False
        )
        # Generate the answer (further quantization or optimization should be applied at the model level)
        answer = generator.text_generation(
            prompt,
            max_new_tokens=128,
            top_p=0.8,
            temperature=0.1,
            stop=[tokenizer.eos_token or "<|eot_id|>"],
            do_sample=True,
            return_full_text=False
        )
        return {"generated_text": answer, "contexts": [node.text for node in nodes_retrieved]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during generation: {e}")

@app.get("/")
def read_root():
    """
    Root endpoint to check API status.
    """
    return {"message": "RAG API is running successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
