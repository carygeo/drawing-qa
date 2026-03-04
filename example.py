#!/usr/bin/env python3
"""
Example usage of Drawing Q&A.

Downloads a sample floor plan and demonstrates:
- Indexing PDFs
- Visual search
- Heatmap visualization (key capability!)
- Region detection
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
    print("\n" + "="*60)
    print("  DRAWING Q&A - Visual Document Understanding Demo")
    print("="*60)
    
    qa = DrawingQA(index_path="./example_index")
    
    # Index the document
    print("\n📄 Indexing Document...")
    pages = qa.index(pdf_path)
    print(f"   Indexed {pages} pages ({32*32} patches per page)")
    
    # Run searches and show region detection
    print("\n🔍 Search Examples with Region Detection:")
    print("-"*60)
    
    queries = [
        "bedroom dimensions",
        "bathroom location", 
        "entrance door",
        "kitchen area",
    ]
    
    for query in queries:
        print(f"\nQuery: \"{query}\"")
        results = qa.search(query, top_k=2)
        
        for r in results:
            x1, y1, x2, y2 = r.region
            print(f"  📍 Page {r.page}: {r.score:.0%} confidence")
            print(f"     Region: x={x1:.0%}-{x2:.0%}, y={y1:.0%}-{y2:.0%}")
            print(f"     Patches matched: {len(r.patch_indices)}")
    
    # Demonstrate heatmap visualization
    print("\n" + "="*60)
    print("  🔥 HEATMAP VISUALIZATION")
    print("="*60)
    print("""
The key capability of ColPali is showing WHERE on the drawing
the answer is located. Each search result includes:

  1. Relevance score (0-100%)
  2. Bounding box (normalized x1,y1,x2,y2)
  3. Patch-level heatmap showing attention intensity

This lets users see exactly which part of the drawing matches
their query - critical for large construction documents.
""")
    
    # Generate visualization
    print("Generating heatmap visualization...")
    
    try:
        # Search for something specific
        results = qa.search("room dimensions", top_k=1)
        
        if results:
            result = results[0]
            output_file = "heatmap_example.png"
            
            qa.visualize(
                result,
                output_path=output_file,
                show=False  # Don't try to display (may not have GUI)
            )
            
            print(f"\n✅ Heatmap saved: {output_file}")
            print(f"   - Shows page {result.page} of {result.filename}")
            print(f"   - Red overlay indicates relevant regions")
            print(f"   - Bounding box shows detected area")
        else:
            print("   No results found for visualization")
            
    except ImportError as e:
        print(f"\n⚠️  Visualization requires extra dependencies:")
        print(f"    pip install matplotlib scipy")
    except Exception as e:
        print(f"\n⚠️  Visualization skipped: {e}")
    
    # Show the raw heatmap data
    print("\n📊 Raw Heatmap Data Structure:")
    print("-"*60)
    
    if results:
        r = results[0]
        print(f"""
SearchResult(
    filename='{r.filename}',
    page={r.page},
    score={r.score:.3f},
    region=({r.region[0]:.2f}, {r.region[1]:.2f}, {r.region[2]:.2f}, {r.region[3]:.2f}),
    patch_indices={r.patch_indices[:5]}{'...' if len(r.patch_indices) > 5 else ''}
)

The patch_indices map to a 32x32 grid overlaid on the page.
Each patch that matched the query contributes to the heatmap.
Higher density of patches = stronger visual match.
""")
    
    # Summary
    print("\n" + "="*60)
    print("  ✅ Demo Complete!")
    print("="*60)
    print("""
Key Capabilities Demonstrated:
  1. Visual search (no OCR needed)
  2. Region detection (WHERE on the page)
  3. Heatmap visualization (attention overlay)
  4. Confidence scoring

Try the CLI:
  drawing-qa search "your query" 
  drawing-qa visualize "your query" --output result.png

Or the API:
  drawing-qa server
  curl "http://localhost:8000/search?query=dimensions"
""")


if __name__ == "__main__":
    main()
