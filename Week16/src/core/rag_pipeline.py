"""
Retrieval-Augmented Generation (RAG) Pipeline for ShopAssist AI (Week 16).
Handles document ingestion, semantic chunking with overlap, dense vectorization,
in-memory vector storage, cosine-similarity retrieval, and failure simulation hooks.
"""

import os
import re
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
import numpy as np

from src.config import settings

class DocumentChunk(BaseModel):
    chunk_id: str
    source_file: str
    doc_title: str
    section: str
    content: str
    embedding: Optional[List[float]] = None

class RAGSearchResult(BaseModel):
    chunk_id: str
    source_file: str
    doc_title: str
    section: str
    content: str
    similarity_score: float

class DenseEmbeddingEngine:
    """
    Lightweight, deterministic dense semantic embedding engine.
    Computes normalized n-gram subword and semantic feature vectors,
    guaranteeing ultra-fast sub-millisecond vectorization without external network overhead.
    """
    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def embed_text(self, text: str) -> np.ndarray:
        words = re.findall(r"\b\w+\b", text.lower())
        vec = np.zeros(self.dimension, dtype=np.float32)
        if not words:
            return vec
            
        for word in words:
            h = hash(word)
            idx = abs(h) % self.dimension
            sign = 1.0 if (h // self.dimension) % 2 == 0 else -1.0
            vec[idx] += sign * (1.0 + math.log(1.0 + len(word)))
            
            for i in range(len(word) - 1):
                bg = word[i:i+2]
                h_bg = hash(bg)
                idx_bg = abs(h_bg) % self.dimension
                vec[idx_bg] += 0.5
                
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec

class RAGPipeline:
    """
    Production RAG Knowledge Pipeline indexing corporate e-commerce policies.
    """
    def __init__(self):
        self.embedding_engine = DenseEmbeddingEngine(dimension=256)
        self.chunks: List[DocumentChunk] = []
        self._index_documents()

    def _index_documents(self):
        kb_path = settings.kb_dir
        if not kb_path.exists():
            print(f"Warning: Knowledge base directory {kb_path} does not exist.")
            return

        chunk_counter = 0
        for md_file in sorted(kb_path.glob("*.md")):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading KB document {md_file}: {e}")
                continue

            file_chunks = self._chunk_markdown(md_file.name, text)
            for ch in file_chunks:
                ch.chunk_id = f"CHK-{chunk_counter:04d}"
                emb = self.embedding_engine.embed_text(ch.content)
                ch.embedding = emb.tolist()
                self.chunks.append(ch)
                chunk_counter += 1

        print(f"RAG Pipeline initialized: {len(self.chunks)} chunks indexed across knowledge base documents.")

    def _chunk_markdown(self, filename: str, content: str) -> List[DocumentChunk]:
        chunks = []
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        doc_title = title_match.group(1).strip() if title_match else filename.replace(".md", "").replace("_", " ").title()

        sections = re.split(r"(^##\s+.+$)", content, flags=re.MULTILINE)
        current_section = "Overview"

        i = 0
        while i < len(sections):
            part = sections[i].strip()
            if not part:
                i += 1
                continue

            if part.startswith("## "):
                current_section = part.replace("## ", "").strip()
                if i + 1 < len(sections):
                    section_body = sections[i + 1].strip()
                    i += 2
                else:
                    section_body = ""
                    i += 1
            else:
                section_body = part
                i += 1

            if not section_body:
                continue

            raw_chunks = self._sliding_window_split(
                section_body,
                chunk_size=settings.rag_chunk_size,
                overlap=settings.rag_chunk_overlap
            )

            for c_text in raw_chunks:
                chunks.append(DocumentChunk(
                    chunk_id="",
                    source_file=filename,
                    doc_title=doc_title,
                    section=current_section,
                    content=c_text.strip()
                ))

        return chunks

    def _sliding_window_split(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = end - overlap
        return chunks

    def search(self, query: str, top_k: int = 3) -> List[RAGSearchResult]:
        """
        Executes semantic search against indexed knowledge base policies.
        Includes failure injection simulation hook.
        """
        # Failure Injection Hook: Malformed RAG output
        if settings.failure_simulation_mode == "malformed_rag":
            return [
                RAGSearchResult(
                    chunk_id="CHK-ERR-00",
                    source_file="corrupted_vector_cache.bin",
                    doc_title="MALFORMED_DATA",
                    section="Error",
                    content="<DATA_CORRUPTION_TAG>???\x00\x00\x01\xFFInvalidUTF8Block</DATA_CORRUPTION_TAG>",
                    similarity_score=0.99
                )
            ]

        if not self.chunks:
            return []

        q_vec = self.embedding_engine.embed_text(query)
        scores: List[Tuple[float, DocumentChunk]] = []

        q_words = set(re.findall(r"\b\w+\b", query.lower()))

        for ch in self.chunks:
            if not ch.embedding:
                continue
            doc_vec = np.array(ch.embedding, dtype=np.float32)
            cosine_sim = float(np.dot(q_vec, doc_vec))

            # Keyword lexical overlap boost
            c_words = set(re.findall(r"\b\w+\b", ch.content.lower()))
            overlap_count = len(q_words.intersection(c_words))
            lexical_boost = 0.05 * min(overlap_count, 5)

            score = cosine_sim + lexical_boost
            scores.append((score, ch))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_matches = scores[:top_k]

        return [
            RAGSearchResult(
                chunk_id=ch.chunk_id,
                source_file=ch.source_file,
                doc_title=ch.doc_title,
                section=ch.section,
                content=ch.content,
                similarity_score=round(score, 4)
            )
            for score, ch in top_matches
        ]

    def retrieve_context(self, query: str, top_k: int = 3) -> Tuple[str, List[str]]:
        """Returns aggregated markdown context and unique source list."""
        results = self.search(query, top_k=top_k)
        if not results:
            return "No matching corporate policy documents found.", []

        context_blocks = []
        sources = set()
        for res in results:
            context_blocks.append(f"### Source: {res.doc_title} ({res.section})\n{res.content}")
            sources.add(f"{res.doc_title} - {res.section}")

        return "\n\n".join(context_blocks), sorted(list(sources))

rag_pipeline = RAGPipeline()
