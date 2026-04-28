import os
import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
UPLOAD_DIR.mkdir(exist_ok=True)
FAISS_DIR.mkdir(exist_ok=True)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

embeddings = OpenAIEmbeddings()
vector_store: FAISS | None = None


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
    global vector_store

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    save_path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

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

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = store.as_retriever(search_kwargs={"k": body.k})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    result = qa_chain.invoke({"query": body.question})

    sources = list(
        {
            doc.metadata.get("source_filename", doc.metadata.get("source", "unknown"))
            for doc in result["source_documents"]
        }
    )

    return AskResponse(answer=result["result"], sources=sources)


@app.delete("/index")
async def clear_index():
    global vector_store
    vector_store = None
    if FAISS_DIR.exists():
        shutil.rmtree(FAISS_DIR)
        FAISS_DIR.mkdir()
    return {"message": "Index cleared."}


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
