#!/usr/bin/env python3
"""
Example usage of Drawing Q&A.

Downloads a sample floor plan and demonstrates indexing + search.
"""

import os
import urllib.request
from pathlib import Path

# Sample public domain floor plan
SAMPLE_URL = "https://www.hud.gov/sites/documents/DOC_12586.PDF"
SAMPLE_NAME = "sample_floor_plan.pdf"


def download_sample():
    """Download a sample PDF if not present."""
    if not Path(SAMPLE_NAME).exists():
        print(f"Downloading sample: {SAMPLE_NAME}")
        urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_NAME)
        print("Done.")
    return SAMPLE_NAME


def main():
    from drawing_qa import DrawingQA
    
    # Download sample
    pdf_path = download_sample()
    
    # Initialize
    print("\n=== Initializing Drawing Q&A ===")
    qa = DrawingQA(index_path="./example_index")
    
    # Index the document
    print("\n=== Indexing Document ===")
    pages = qa.index(pdf_path)
    print(f"Indexed {pages} pages")
    
    # Run some searches
    print("\n=== Search Examples ===")
    
    queries = [
        "bedroom dimensions",
        "bathroom location",
        "entrance",
        "window",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = qa.search(query, top_k=3)
        for r in results:
            print(f"  - Page {r.page}: {r.score:.0%}")
    
    # Ask a question (no LLM)
    print("\n=== Ask Question (search only) ===")
    answer = qa.ask("What are the room dimensions?", llm=None)
    print(f"Q: {answer.question}")
    print(f"A: {answer.answer}")
    
    # List indexed docs
    print("\n=== Indexed Documents ===")
    for doc in qa.list_documents():
        print(f"  {doc['filename']}: {doc['pages']} pages")
    
    print("\n✅ Example complete!")
    print("\nNext steps:")
    print("  - Try: drawing-qa ask 'your question' --llm claude")
    print("  - Try: drawing-qa server (then visit http://localhost:8000/docs)")


if __name__ == "__main__":
    main()
