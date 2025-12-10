import os
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


FAISS_DIR = "faiss_index"


class RagService:
    """
     Minimal RAG service:
    - Load documents
    - Chunk them
    - Embed & store vectors in FAISS
    - Retrieve relevant chunks and pass them to the LLM
    """

    def __init__(
        self,
        data_dir: str = "data",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Embeddings / LLM は TrueFoundry Gateway 経由の OpenAI 互換 API を想定
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        )
        self.llm = ChatOpenAI(
            model=os.getenv("CHAT_MODEL", "gpt-4.1-mini"),
            temperature=0.1,
        )

        self.vector_store = self._build_or_load_vectorstore()

    # =====================
    # Setup / Indexing
    # =====================

    def _load_documents(self) -> List[Document]:
        """
        Load documents from data/ directory.

        Supported formats:
        - .txt using TextLoader
        - .pdf using PyPDFLoader

        This can later be extended to other loaders
        (e.g., GCS, Confluence, Notion, etc.).
        """
        docs: List[Document] = []

        # Load text files
        txt_loader = DirectoryLoader(
            self.data_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True,
        )
        docs_txt: List[Document] = txt_loader.load()
        docs.extend(docs_txt)

        # Load PDF files
        pdf_loader = DirectoryLoader(
            self.data_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        docs_pdf: List[Document] = pdf_loader.load()
        docs.extend(docs_pdf)

        # Normalize source metadata
        for d in docs:
            if "source" not in d.metadata:
                # DirectoryLoader は file_path を metadata に入れてくれることが多い
                d.metadata["source"] = d.metadata.get("file_path", "local")
        return docs

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        """
        Split documents into chunks using LangChain's
        RecursiveCharacterTextSplitter.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )
        splits = splitter.split_documents(docs)
        return splits

    def _build_or_load_vectorstore(self) -> FAISS:
        """
        Load FAISS index from disk if present.
        Otherwise, build it from documents and save it.

        When deployed on TrueFoundry, this directory
        can be backed by a Persistent Volume.
        """
        path = Path(FAISS_DIR)
        if path.exists():
            vector_store = FAISS.load_local(
                FAISS_DIR,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            return vector_store

        # First-time indexing
        docs = self._load_documents()
        splits = self._split_documents(docs)
        vector_store = FAISS.from_documents(splits, embedding=self.embeddings)
        vector_store.save_local(FAISS_DIR)
        return vector_store

    # =====================
    # Query / Retrieval
    # =====================

    def query(
        self,
        question: str,
        k: int = 3,
        metadata_filter: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Execute a simple RAG query:
        1. Retrieve relevant chunks from vector DB
        2. Inject them into the prompt
        3. Generate an answer using the LLM
        """
        search_kwargs: Dict[str, Any] = {"k": k}
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter

        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

        docs: List[Document] = retriever.invoke(question)

        context_blocks = []
        for i, d in enumerate(docs, start=1):
            src = d.metadata.get("source", f"chunk-{i}")
            context_blocks.append(f"[Source {i}: {src}]\n{d.page_content}")

        context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "No context."

        prompt = (
            "You are a helpful assistant.\n"
            "Use ONLY the following context to answer the user's question.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            "Answer in the same language as the question."
        )

        completion = self.llm.invoke(prompt)

        return {
            "answer": completion.content,
            "documents": docs,
        }

