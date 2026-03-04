# Drawing Q&A

**Ask questions about PDF drawings using visual AI.** No OCR required.

Uses [ColPali](https://github.com/illuin-tech/colpali) for visual document understanding - the same technology that understands charts, diagrams, and technical drawings by looking at them, not just reading text.

## 🔥 Key Capability: Heatmap Visualization

The killer feature is showing **WHERE** on the drawing the answer is located:

```bash
drawing-qa visualize "fire exit location" --output result.png
```

This generates a heatmap overlay showing exactly which regions of the drawing match your query - critical for large construction documents where you need to find specific details fast.

![Heatmap Example](docs/heatmap_example.png)

## Quick Start

```bash
# Install
pip install drawing-qa

# Index a PDF
drawing-qa index blueprint.pdf

# Search
drawing-qa search "fire exit location"

# Ask with answer
drawing-qa ask "What's the room dimension?" --llm claude
```

## Installation

### Prerequisites

- Python 3.10+
- [Poppler](https://poppler.freedesktop.org/) for PDF rendering

```bash
# macOS
brew install poppler

# Ubuntu/Debian
apt-get install poppler-utils

# Windows
# Download from: https://github.com/oschwartz10612/poppler-windows/releases
```

### Install Package

```bash
# Basic (search only)
pip install drawing-qa

# With visualization
pip install drawing-qa[viz]

# With API server
pip install drawing-qa[server]

# With LLM answers (Claude/OpenAI)
pip install drawing-qa[llm]

# Everything
pip install drawing-qa[all]
```

### From Source

```bash
git clone https://github.com/carygreenwood/drawing-qa
cd drawing-qa

# Using uv (recommended)
uv sync --all-extras

# Or pip
pip install -e ".[all]"
```

## Usage

### Python API

```python
from drawing_qa import DrawingQA

# Initialize
qa = DrawingQA()

# Index PDFs
qa.index("floor_plan.pdf")
qa.index("electrical.pdf")
qa.index("plumbing.pdf")

# Search (returns locations)
results = qa.search("fire alarm panel")
for r in results:
    print(f"{r.filename} page {r.page}: {r.score:.0%}")

# Ask with LLM answer
answer = qa.ask("Where is the main electrical panel?", llm="claude")
print(answer.answer)
print(f"Sources: {[s.filename for s in answer.sources]}")

# Visualize result (shows heatmap overlay)
qa.visualize(results[0], output_path="result.png")
```

### CLI

```bash
# Index documents
drawing-qa index blueprint.pdf
drawing-qa index floor_plan.pdf --force

# Search (returns pages + regions)
drawing-qa search "hvac unit location"
drawing-qa search "dimension" --top-k 10

# Visualize with heatmap (key feature!)
drawing-qa visualize "fire exit" --output fire_exit.png
drawing-qa visualize "electrical panel" -o panel.png --show

# Get answers with LLM
drawing-qa ask "What size is the main duct?" --llm claude
drawing-qa ask "Fire rating requirement" --llm openai

# Manage index
drawing-qa list
drawing-qa clear
```

### REST API

```bash
# Start server
drawing-qa server --port 8000

# Or with uvicorn
uvicorn drawing_qa:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Status |
| GET | `/documents` | List indexed docs |
| POST | `/index` | Upload & index PDF |
| GET | `/search?query=...` | Search documents |
| GET | `/ask?question=...&llm=claude` | Get answer |
| DELETE | `/clear` | Clear index |

**Example:**

```bash
# Index a PDF
curl -X POST -F "file=@drawing.pdf" http://localhost:8000/index

# Search
curl "http://localhost:8000/search?query=fire%20exit"

# Ask
curl "http://localhost:8000/ask?question=room%20dimensions&llm=claude"
```

## How It Works

1. **Index**: PDFs are rendered to images, then ColPali generates 1024 patch embeddings per page (32x32 grid)
2. **Search**: Your query is embedded and matched against patches using cosine similarity
3. **Locate**: Matching patches are aggregated to identify relevant pages AND specific regions
4. **Visualize**: Heatmap overlay shows attention intensity across the drawing
5. **Answer**: Optionally, an LLM synthesizes the search results into a coherent answer

```
PDF → Images → ColPali Patches → ChromaDB
                                    ↓
Query → ColPali Embedding → Vector Search → Results + Regions
                                    ↓
                            Heatmap Visualization
                                    ↓
                              LLM Answer (optional)
```

### Heatmap Details

Each page is divided into a 32x32 grid (1024 patches). When you search:

- Each patch gets a relevance score based on visual similarity to your query
- High-scoring patches are highlighted in the heatmap (yellow → red)
- A bounding box is drawn around the region of interest
- You get normalized coordinates (x1, y1, x2, y2) for integration

```python
result = qa.search("electrical panel")[0]
print(result.region)  # (0.65, 0.20, 0.85, 0.45) = right side, upper quarter
print(result.score)   # 0.78 = 78% confidence
print(result.patch_indices)  # [234, 235, 266, 267, ...] = which patches matched
```

## Configuration

### Environment Variables

```bash
# For LLM answers
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

### Custom Index Location

```python
qa = DrawingQA(index_path="/path/to/index")
```

### Device Selection

```python
# Force CPU
qa = DrawingQA(device="cpu")

# Use CUDA
qa = DrawingQA(device="cuda")

# Use Apple Silicon
qa = DrawingQA(device="mps")
```

## Docker

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install -e ".[server]"

EXPOSE 8000
CMD ["drawing-qa", "server", "--host", "0.0.0.0"]
```

```bash
docker build -t drawing-qa .
docker run -p 8000:8000 -v $(pwd)/index:/app/drawing_index drawing-qa
```

## Integration Examples

### Streamlit App

```python
import streamlit as st
from drawing_qa import DrawingQA

qa = DrawingQA()

uploaded = st.file_uploader("Upload PDF", type="pdf")
if uploaded:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded.read())
    qa.index("temp.pdf")

question = st.text_input("Ask about the drawing:")
if question:
    answer = qa.ask(question, llm="claude")
    st.write(answer.answer)
    
    for src in answer.sources:
        st.write(f"📄 {src.filename} p.{src.page} ({src.score:.0%})")
```

### FastAPI Integration

```python
from fastapi import FastAPI
from drawing_qa import DrawingQA

app = FastAPI()
qa = DrawingQA(index_path="./my_index")

@app.get("/query")
def query_drawings(q: str):
    answer = qa.ask(q, llm="claude")
    return {
        "answer": answer.answer,
        "confidence": answer.confidence,
        "pages": [{"file": s.filename, "page": s.page} for s in answer.sources]
    }
```

## Hardware Requirements

| Mode | VRAM/RAM | Speed |
|------|----------|-------|
| GPU (CUDA) | 8GB VRAM | Fast |
| Apple Silicon (MPS) | 16GB unified | Good |
| CPU | 16GB RAM | Slow |

## Limitations

- First load downloads ~3GB model weights
- Indexing is slow on CPU (~30s/page vs ~3s/page on GPU)
- Best for technical drawings, diagrams, floor plans
- Not ideal for text-heavy documents (use traditional RAG)

## License

MIT

## Credits

- [ColPali](https://github.com/illuin-tech/colpali) - Visual document understanding
- [ChromaDB](https://www.trychroma.com/) - Vector storage
- Built for Suffolk Construction
