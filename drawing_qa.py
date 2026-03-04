#!/usr/bin/env python3
"""
Drawing Q&A - ColPali-based visual document understanding for PDFs.

A streamlined POC for asking questions about PDF drawings.
Uses ColPali for visual understanding - no OCR required.

Usage:
    # Index a PDF
    qa = DrawingQA()
    qa.index("drawing.pdf")
    
    # Ask questions
    results = qa.ask("Where is the fire exit?")
    
    # Get answer with context
    answer = qa.answer("What's the room dimension?")
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Lazy imports for optional dependencies
_colpali_model = None
_colpali_processor = None


@dataclass
class SearchResult:
    """A search result with location and relevance."""
    filename: str
    page: int
    score: float
    region: Tuple[float, float, float, float]  # x1, y1, x2, y2 normalized
    patch_indices: List[int]
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Answer:
    """An answer with source references."""
    question: str
    answer: str
    sources: List[SearchResult]
    confidence: float
    
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "confidence": self.confidence
        }


class DrawingQA:
    """
    Simple interface to ask questions about PDF drawings.
    
    Uses ColPali for visual document understanding and ChromaDB for storage.
    """
    
    def __init__(
        self,
        index_path: str = "./drawing_index",
        model_name: str = "vidore/colpali-v1.2",
        device: str = None
    ):
        """
        Initialize the Drawing Q&A system.
        
        Args:
            index_path: Directory to store the vector index
            model_name: ColPali model to use
            device: "cuda", "mps", or "cpu" (auto-detected if None)
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.device = device or self._detect_device()
        self.patch_grid = 32
        
        # Lazy-loaded components
        self._model = None
        self._processor = None
        self._collection = None
        self._metadata: Dict[str, dict] = {}
        
        # Load existing metadata
        self._load_metadata()
    
    def _detect_device(self) -> str:
        """Auto-detect best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    @property
    def model(self):
        """Lazy-load ColPali model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def processor(self):
        """Lazy-load ColPali processor."""
        if self._processor is None:
            self._load_model()
        return self._processor
    
    @property
    def collection(self):
        """Lazy-load ChromaDB collection."""
        if self._collection is None:
            self._load_collection()
        return self._collection
    
    def _load_model(self):
        """Load ColPali model and processor."""
        try:
            import torch
            from colpali_engine.models import ColPali, ColPaliProcessor
        except ImportError as e:
            raise ImportError(
                "ColPali dependencies required. Install with:\n"
                "  pip install colpali-engine torch"
            ) from e
        
        print(f"Loading ColPali model on {self.device}...")
        
        # Use float32 on CPU to avoid meta tensor issues
        # Use float16 on GPU/MPS for memory efficiency
        if self.device == "cpu":
            self._model = ColPali.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            ).eval()
        else:
            self._model = ColPali.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device
            ).eval()
        
        self._processor = ColPaliProcessor.from_pretrained(self.model_name)
        print("Model loaded.")
    
    def _load_collection(self):
        """Load or create ChromaDB collection."""
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "ChromaDB required. Install with:\n"
                "  pip install chromadb"
            ) from e
        
        client = chromadb.PersistentClient(path=str(self.index_path / "chroma"))
        self._collection = client.get_or_create_collection(
            name="drawings",
            metadata={"hnsw:space": "cosine"}
        )
    
    def _load_metadata(self):
        """Load document metadata from disk."""
        meta_path = self.index_path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)
    
    def _save_metadata(self):
        """Save document metadata to disk."""
        meta_path = self.index_path / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self._metadata, f, indent=2)
    
    def _pdf_hash(self, pdf_path: str) -> str:
        """Generate hash for PDF file."""
        with open(pdf_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:12]
    
    def _embed_image(self, image) -> np.ndarray:
        """Generate embeddings for an image."""
        import torch
        
        with torch.no_grad():
            inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
            embeddings = self.model.forward_images(**inputs)
        
        return embeddings.cpu().numpy().squeeze(0)
    
    def _embed_query(self, query: str) -> np.ndarray:
        """Generate embeddings for a query."""
        import torch
        
        with torch.no_grad():
            inputs = self.processor(text=[query], return_tensors="pt").to(self.device)
            embeddings = self.model.forward_queries(**inputs)
        
        return embeddings.cpu().numpy()
    
    def index(
        self,
        pdf_path: str,
        dpi: int = 200,
        force: bool = False
    ) -> int:
        """
        Index a PDF document.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for rendering pages
            force: Re-index even if already indexed
            
        Returns:
            Number of pages indexed
        """
        try:
            from pdf2image import convert_from_path
        except ImportError as e:
            raise ImportError(
                "pdf2image required. Install with:\n"
                "  pip install pdf2image\n"
                "Also install poppler: brew install poppler (macOS)"
            ) from e
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        filename = pdf_path.name
        pdf_hash = self._pdf_hash(str(pdf_path))
        
        # Check if already indexed
        if not force and filename in self._metadata:
            if self._metadata[filename].get("hash") == pdf_hash:
                print(f"Already indexed: {filename}")
                return self._metadata[filename]["pages"]
        
        print(f"Indexing: {filename}")
        pages = convert_from_path(str(pdf_path), dpi=dpi)
        print(f"  Pages: {len(pages)}")
        
        indexed = 0
        for page_num, page_image in enumerate(pages, start=1):
            doc_id = f"{filename}:p{page_num}"
            
            # Generate embeddings
            embeddings = self._embed_image(page_image)
            
            # Store each patch
            for patch_idx in range(embeddings.shape[0]):
                patch_id = f"{doc_id}:patch{patch_idx}"
                row = patch_idx // self.patch_grid
                col = patch_idx % self.patch_grid
                
                self.collection.add(
                    ids=[patch_id],
                    embeddings=[embeddings[patch_idx].tolist()],
                    metadatas=[{
                        "doc_id": doc_id,
                        "filename": filename,
                        "page": page_num,
                        "patch": patch_idx,
                        "row": row,
                        "col": col,
                    }]
                )
            
            indexed += 1
            print(f"  Page {page_num}: {embeddings.shape[0]} patches")
        
        # Save metadata
        self._metadata[filename] = {
            "hash": pdf_hash,
            "pages": indexed,
            "path": str(pdf_path.absolute())
        }
        self._save_metadata()
        
        return indexed
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filename_filter: str = None
    ) -> List[SearchResult]:
        """
        Search indexed drawings.
        
        Args:
            query: Natural language question
            top_k: Number of results to return
            filename_filter: Only search specific file
            
        Returns:
            List of SearchResult objects
        """
        # Embed query
        query_embedding = self._embed_query(query)
        query_vector = query_embedding.mean(axis=1).squeeze().tolist()
        
        # Build filter
        where_filter = None
        if filename_filter:
            where_filter = {"filename": filename_filter}
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k * 20,  # Get more for aggregation
            where=where_filter,
            include=["metadatas", "distances"]
        )
        
        # Aggregate by document
        doc_scores: Dict[str, dict] = {}
        
        for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
            doc_id = metadata['doc_id']
            score = 1 - distance  # Convert distance to similarity
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'filename': metadata['filename'],
                    'page': metadata['page'],
                    'patches': [],
                    'scores': []
                }
            
            doc_scores[doc_id]['patches'].append(metadata['patch'])
            doc_scores[doc_id]['scores'].append(score)
        
        # Build results
        search_results = []
        for doc_id, data in doc_scores.items():
            avg_score = np.mean(data['scores'])
            region = self._patches_to_region(data['patches'])
            
            search_results.append(SearchResult(
                filename=data['filename'],
                page=data['page'],
                score=float(avg_score),
                region=region,
                patch_indices=data['patches']
            ))
        
        # Sort by score
        search_results.sort(key=lambda x: x.score, reverse=True)
        
        return search_results[:top_k]
    
    def _patches_to_region(self, patches: List[int]) -> Tuple[float, float, float, float]:
        """Convert patch indices to normalized bounding box."""
        if not patches:
            return (0.0, 0.0, 1.0, 1.0)
        
        rows = [p // self.patch_grid for p in patches]
        cols = [p % self.patch_grid for p in patches]
        
        x1 = min(cols) / self.patch_grid
        y1 = min(rows) / self.patch_grid
        x2 = (max(cols) + 1) / self.patch_grid
        y2 = (max(rows) + 1) / self.patch_grid
        
        return (x1, y1, x2, y2)
    
    def ask(
        self,
        question: str,
        top_k: int = 3,
        llm: str = "claude"
    ) -> Answer:
        """
        Ask a question and get an answer with sources.
        
        Args:
            question: Natural language question about the drawings
            top_k: Number of source documents to consider
            llm: LLM to use for answering ("claude", "openai", or None for search only)
            
        Returns:
            Answer object with response and sources
        """
        # Search for relevant documents
        sources = self.search(question, top_k=top_k)
        
        if not sources:
            return Answer(
                question=question,
                answer="No relevant documents found.",
                sources=[],
                confidence=0.0
            )
        
        # If no LLM, just return search results
        if llm is None:
            return Answer(
                question=question,
                answer=f"Found {len(sources)} relevant pages. Top result: {sources[0].filename} page {sources[0].page} ({sources[0].score:.0%} match)",
                sources=sources,
                confidence=sources[0].score
            )
        
        # Generate answer with LLM
        context = self._build_context(sources)
        answer_text = self._generate_answer(question, context, llm)
        
        return Answer(
            question=question,
            answer=answer_text,
            sources=sources,
            confidence=sources[0].score
        )
    
    def _build_context(self, sources: List[SearchResult]) -> str:
        """Build context string from search results."""
        lines = ["Relevant document locations found:"]
        for i, src in enumerate(sources, 1):
            x1, y1, x2, y2 = src.region
            lines.append(
                f"{i}. {src.filename} page {src.page} "
                f"(relevance: {src.score:.0%}, region: {x1:.0%}-{x2:.0%} x {y1:.0%}-{y2:.0%})"
            )
        return "\n".join(lines)
    
    def _generate_answer(self, question: str, context: str, llm: str) -> str:
        """Generate answer using LLM."""
        prompt = f"""Based on the visual search of construction drawings, answer this question.

QUESTION: {question}

SEARCH RESULTS:
{context}

Note: I searched the drawings visually using AI. The results show which pages and regions are most relevant to your question. Provide a helpful answer based on what was found, noting the specific page references.

ANSWER:"""

        if llm == "claude":
            return self._call_claude(prompt)
        elif llm == "openai":
            return self._call_openai(prompt)
        else:
            return f"Search completed. {context}"
    
    def _call_claude(self, prompt: str) -> str:
        """Call Claude API."""
        try:
            import anthropic
        except ImportError:
            return "Claude API not available. Install with: pip install anthropic"
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return "ANTHROPIC_API_KEY not set"
        
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        try:
            import openai
        except ImportError:
            return "OpenAI API not available. Install with: pip install openai"
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "OPENAI_API_KEY not set"
        
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024
        )
        return response.choices[0].message.content
    
    def visualize(
        self,
        result: SearchResult,
        output_path: str = None,
        show: bool = True
    ):
        """
        Visualize a search result with heatmap overlay.
        
        Args:
            result: SearchResult to visualize
            output_path: Save visualization to file
            show: Display visualization
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from pdf2image import convert_from_path
            from scipy.ndimage import zoom
        except ImportError as e:
            raise ImportError(
                "Visualization requires: pip install matplotlib scipy pdf2image"
            ) from e
        
        # Get PDF path
        if result.filename not in self._metadata:
            raise ValueError(f"Document not indexed: {result.filename}")
        
        pdf_path = self._metadata[result.filename]["path"]
        
        # Load page image
        pages = convert_from_path(
            pdf_path,
            first_page=result.page,
            last_page=result.page,
            dpi=150
        )
        page_image = pages[0]
        
        # Create heatmap
        heatmap = np.zeros((self.patch_grid, self.patch_grid))
        for patch in result.patch_indices:
            row = patch // self.patch_grid
            col = patch % self.patch_grid
            heatmap[row, col] = result.score
        
        # Resize heatmap to image size
        h, w = page_image.size[1], page_image.size[0]
        heatmap_resized = zoom(heatmap, (h / self.patch_grid, w / self.patch_grid), order=1)
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.imshow(page_image)
        
        # Overlay heatmap
        heatmap_masked = np.ma.masked_where(heatmap_resized < 0.1, heatmap_resized)
        ax.imshow(heatmap_masked, cmap='YlOrRd', alpha=0.5, extent=[0, w, h, 0])
        
        # Draw bounding box
        x1, y1, x2, y2 = result.region
        rect = mpatches.Rectangle(
            (x1 * w, y1 * h),
            (x2 - x1) * w,
            (y2 - y1) * h,
            fill=False,
            edgecolor='red',
            linewidth=2
        )
        ax.add_patch(rect)
        
        ax.set_title(f"{result.filename} | Page {result.page} | Score: {result.score:.0%}")
        ax.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            print(f"Saved: {output_path}")
        
        if show:
            plt.show()
        
        plt.close()
        return fig
    
    def list_documents(self) -> List[dict]:
        """List all indexed documents."""
        return [
            {"filename": k, **v}
            for k, v in self._metadata.items()
        ]
    
    def clear(self):
        """Clear all indexed documents."""
        import shutil
        
        if self._collection is not None:
            # Delete all from collection
            try:
                self.collection.delete(where={})
            except:
                pass
        
        # Clear metadata
        self._metadata = {}
        self._save_metadata()
        
        # Remove chroma directory
        chroma_path = self.index_path / "chroma"
        if chroma_path.exists():
            shutil.rmtree(chroma_path)
        
        self._collection = None
        print("Index cleared.")


# === CLI Interface ===

def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Drawing Q&A - Ask questions about PDF drawings"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index a PDF document")
    index_parser.add_argument("pdf", help="Path to PDF file")
    index_parser.add_argument("--force", action="store_true", help="Re-index if exists")
    index_parser.add_argument("--index-path", default="./drawing_index", help="Index directory")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search indexed documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    search_parser.add_argument("--index-path", default="./drawing_index", help="Index directory")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question and get an answer")
    ask_parser.add_argument("question", help="Your question")
    ask_parser.add_argument("--llm", choices=["claude", "openai", "none"], default="none")
    ask_parser.add_argument("--index-path", default="./drawing_index", help="Index directory")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List indexed documents")
    list_parser.add_argument("--index-path", default="./drawing_index", help="Index directory")
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear the index")
    clear_parser.add_argument("--index-path", default="./drawing_index", help="Index directory")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    server_parser.add_argument("--index-path", default="./drawing_index", help="Index directory")
    
    # Visualize command (key capability!)
    viz_parser = subparsers.add_parser("visualize", help="Generate heatmap visualization for a query")
    viz_parser.add_argument("query", help="Search query to visualize")
    viz_parser.add_argument("--output", "-o", default="heatmap.png", help="Output image path")
    viz_parser.add_argument("--show", action="store_true", help="Display the visualization")
    viz_parser.add_argument("--index-path", default="./drawing_index", help="Index directory")
    
    args = parser.parse_args()
    
    if args.command == "index":
        qa = DrawingQA(index_path=args.index_path)
        pages = qa.index(args.pdf, force=args.force)
        print(f"Indexed {pages} pages")
    
    elif args.command == "search":
        qa = DrawingQA(index_path=args.index_path)
        results = qa.search(args.query, top_k=args.top_k)
        
        print(f"\nResults for: {args.query}\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r.filename} page {r.page} ({r.score:.0%})")
            print(f"   Region: {r.region}")
    
    elif args.command == "ask":
        qa = DrawingQA(index_path=args.index_path)
        llm = None if args.llm == "none" else args.llm
        answer = qa.ask(args.question, llm=llm)
        
        print(f"\nQuestion: {answer.question}")
        print(f"\nAnswer: {answer.answer}")
        print(f"\nConfidence: {answer.confidence:.0%}")
        print("\nSources:")
        for s in answer.sources:
            print(f"  - {s.filename} page {s.page} ({s.score:.0%})")
    
    elif args.command == "list":
        qa = DrawingQA(index_path=args.index_path)
        docs = qa.list_documents()
        
        if not docs:
            print("No documents indexed.")
        else:
            print(f"\nIndexed documents ({len(docs)}):\n")
            for doc in docs:
                print(f"  {doc['filename']} ({doc['pages']} pages)")
    
    elif args.command == "clear":
        qa = DrawingQA(index_path=args.index_path)
        qa.clear()
    
    elif args.command == "server":
        start_server(args.host, args.port, args.index_path)
    
    elif args.command == "visualize":
        qa = DrawingQA(index_path=args.index_path)
        results = qa.search(args.query, top_k=1)
        
        if not results:
            print("No results found for query.")
            return
        
        result = results[0]
        print(f"\n🔍 Query: {args.query}")
        print(f"📄 Best match: {result.filename} page {result.page} ({result.score:.0%})")
        print(f"📍 Region: x={result.region[0]:.0%}-{result.region[2]:.0%}, y={result.region[1]:.0%}-{result.region[3]:.0%}")
        print(f"🔥 Generating heatmap...")
        
        try:
            qa.visualize(result, output_path=args.output, show=args.show)
            print(f"\n✅ Heatmap saved: {args.output}")
            print(f"   Red overlay shows regions matching your query.")
            print(f"   Bounding box highlights the detected area.")
        except ImportError:
            print("\n❌ Visualization requires: pip install matplotlib scipy")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    else:
        parser.print_help()


def start_server(host: str, port: int, index_path: str):
    """Start FastAPI server."""
    try:
        from fastapi import FastAPI, UploadFile, File, HTTPException
        from fastapi.responses import JSONResponse
        import uvicorn
        import tempfile
        import shutil
    except ImportError:
        print("Server requires: pip install fastapi uvicorn python-multipart")
        return
    
    app = FastAPI(
        title="Drawing Q&A API",
        description="Ask questions about PDF drawings using ColPali",
        version="1.0.0"
    )
    
    qa = DrawingQA(index_path=index_path)
    
    @app.get("/")
    def root():
        return {"status": "ok", "documents": len(qa.list_documents())}
    
    @app.get("/documents")
    def list_documents():
        return qa.list_documents()
    
    @app.post("/index")
    async def index_document(file: UploadFile = File(...), force: bool = False):
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        try:
            pages = qa.index(tmp_path, force=force)
            # Update metadata with original filename
            qa._metadata[file.filename] = qa._metadata.pop(Path(tmp_path).name)
            qa._metadata[file.filename]["path"] = tmp_path
            qa._save_metadata()
            return {"filename": file.filename, "pages": pages}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/search")
    def search(query: str, top_k: int = 5):
        results = qa.search(query, top_k=top_k)
        return [r.to_dict() for r in results]
    
    @app.get("/ask")
    def ask(question: str, llm: str = "none"):
        llm_choice = None if llm == "none" else llm
        answer = qa.ask(question, llm=llm_choice)
        return answer.to_dict()
    
    @app.delete("/clear")
    def clear():
        qa.clear()
        return {"status": "cleared"}
    
    print(f"Starting server at http://{host}:{port}")
    print("Docs: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
