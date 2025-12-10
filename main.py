from typing import List, Dict, Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from rag_pipeline import RagService

app = FastAPI(
    title="TrueFoundry RAG Demo",
    description="Simple RAG service using LangChain + FAISS",
    version="0.1.0",
)

rag_service = RagService(data_dir="data")


class RagRequest(BaseModel):
    question: str
    top_k: int = 3
    # 例: {"source": {"$eq": "news"}} や {"source": "news"}
    metadata_filter: Optional[Dict[str, Any]] = None


class RagSource(BaseModel):
    page_content: str
    metadata: Dict[str, Any]


class RagResponse(BaseModel):
    answer: str
    sources: List[RagSource]


@app.get("/")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/rag", response_model=RagResponse)
async def rag(req: RagRequest) -> RagResponse:
    """
    RAG endpoint.

    - question: user question
    - top_k: number of retrieved chunks
    - metadata_filter: optional filter passed to the vector store
    """
    result = rag_service.query(
        question=req.question,
        k=req.top_k,
        metadata_filter=req.metadata_filter,
    )

    answer: str = result["answer"]
    docs = result["documents"]

    sources = [
        RagSource(
            page_content=d.page_content,
            metadata=d.metadata,
        )
        for d in docs
    ]

    return RagResponse(answer=answer, sources=sources)

