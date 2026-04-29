import hashlib
import json
import os
import shutil
from dotenv import load_dotenv
load_dotenv()
import uuid
from pathlib import Path
from typing import Dict, List, Set

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

app = FastAPI(title="Document Q&A System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
FAISS_DIR = Path("faiss_index")
INDEXED_FILES_PATH = FAISS_DIR / "indexed_files.json"
UPLOAD_DIR.mkdir(exist_ok=True)
FAISS_DIR.mkdir(exist_ok=True)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store: FAISS | None = None


def load_indexed_files() -> Dict[str, str]:
    if INDEXED_FILES_PATH.exists():
        return json.loads(INDEXED_FILES_PATH.read_text())
    return {}


def save_indexed_files(files: Dict[str, str]):
    INDEXED_FILES_PATH.write_text(json.dumps(files))


def compute_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# maps content_hash -> original filename
indexed_files: Dict[str, str] = load_indexed_files()


def get_vector_store() -> FAISS:
    global vector_store
    if vector_store is None:
        index_path = FAISS_DIR / "index.faiss"
        if index_path.exists():
            vector_store = FAISS.load_local(
                str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True
            )
    return vector_store


def load_document(file_path: Path):
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif ext == ".docx":
        loader = Docx2txtLoader(str(file_path))
    elif ext == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()


class AskRequest(BaseModel):
    question: str
    k: int = 4


class AskResponse(BaseModel):
    answer: str
    sources: List[str]


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    global vector_store, indexed_files

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    content_hash = compute_hash(file_bytes)

    if content_hash in indexed_files:
        existing_name = indexed_files[content_hash]
        if existing_name == file.filename:
            raise HTTPException(
                status_code=409,
                detail=f"'{file.filename}' is already indexed.",
            )
        else:
            raise HTTPException(
                status_code=409,
                detail=f"This file's content is already indexed as '{existing_name}'.",
            )

    save_path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    with save_path.open("wb") as f:
        f.write(file_bytes)

    try:
        docs = load_document(save_path)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to parse document: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise HTTPException(status_code=422, detail="Document appears to be empty.")

    for chunk in chunks:
        chunk.metadata["source_filename"] = file.filename

    if vector_store is None:
        vector_store = FAISS.from_documents(chunks, embeddings)
    else:
        vector_store.add_documents(chunks)

    vector_store.save_local(str(FAISS_DIR))

    indexed_files[content_hash] = file.filename
    save_indexed_files(indexed_files)

    return UploadResponse(
        filename=file.filename,
        chunks_indexed=len(chunks),
        message=f"Successfully indexed {len(chunks)} chunks from '{file.filename}'.",
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(body: AskRequest):
    store = get_vector_store()
    if store is None:
        raise HTTPException(
            status_code=404, detail="No documents indexed yet. Upload a document first."
        )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    retriever = store.as_retriever(search_kwargs={"k": body.k})

    prompt = ChatPromptTemplate.from_template(
        "Use the following context to answer the question. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

    source_docs: list = []

    def retrieve_and_track(question: str):
        docs = retriever.invoke(question)
        source_docs.extend(docs)
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retrieve_and_track, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(body.question)

    sources = list(
        {
            doc.metadata.get("source_filename", doc.metadata.get("source", "unknown"))
            for doc in source_docs
        }
    )

    return AskResponse(answer=answer, sources=sources)


@app.delete("/index")
async def clear_index():
    global vector_store, indexed_files
    vector_store = None
    indexed_files = {}
    if FAISS_DIR.exists():
        shutil.rmtree(FAISS_DIR)
        FAISS_DIR.mkdir()
    return {"message": "Index cleared."}


@app.get("/indexed-files")
async def list_indexed_files():
    return {"indexed_files": [{"filename": name, "hash": h} for h, name in indexed_files.items()]}


@app.get("/health")
async def health():
    store = get_vector_store()
    return {
        "status": "ok",
        "index_loaded": store is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
