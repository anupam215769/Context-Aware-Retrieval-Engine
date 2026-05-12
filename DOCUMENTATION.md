# Technical Documentation

## 1. Similarity Metric Choice: Cosine vs Euclidean

### What We Use

This pipeline uses **Cosine Similarity** implemented via **FAISS `IndexFlatIP`** (Inner Product) on **L2-normalized vectors**.

```python
# From src/vector_store.py
self.index = faiss.IndexFlatIP(dimension)  # Inner product index

# Normalization at add/query time
faiss.normalize_L2(vectors)   # Normalizes in-place to unit length
faiss.normalize_L2(query)
```

For two unit-normalized vectors `a` and `b` (where `||a|| = ||b|| = 1`):

```
cosine_similarity(a, b) = a · b   (inner product)
```

### Why Cosine Similarity?

| Criterion | Cosine Similarity | Euclidean (L2) Distance |
|-----------|-------------------|------------------------|
| **Training objective** | sentence-transformers models are trained to maximize cosine similarity between semantically similar sentences | Not directly optimized for semantic NLP tasks |
| **Magnitude sensitivity** | Invariant to vector magnitude — only direction matters | Sensitive to both direction AND magnitude |
| **Score interpretability** | Bounded in `[-1, 1]`: 1.0 = identical, 0 = orthogonal, -1 = opposite | Unbounded; value depends on dimensionality and scale |
| **NLP suitability** | Standard for BERT-family embeddings, sentence embeddings | Better for coordinate/spatial data |
| **FAISS implementation** | `IndexFlatIP` + `normalize_L2()` → exact cosine similarity | `IndexFlatL2` natively |

### Mathematical Relationship

For unit-normalized vectors, Euclidean distance and cosine similarity are **monotonically related**:

```
||a - b||² = ||a||² - 2(a·b) + ||b||²
           = 1 - 2(a·b) + 1          [since ||a|| = ||b|| = 1]
           = 2(1 - cosine_similarity(a, b))
```

This means:
- When vectors are normalized, **L2 distance and cosine similarity produce identical rankings**
- Cosine similarity scores are more interpretable (bounded), so we prefer them
- FAISS `IndexFlatIP` is slightly faster than `IndexFlatL2` in practice

### When to Use Euclidean Instead

Euclidean distance is preferable when:
- Embedding magnitude carries semantic meaning (e.g., word frequency vectors like TF-IDF)
- Working with physical/spatial coordinates
- The embedding model was trained with a Euclidean objective (e.g., some image embedding models)

### Our Model: `all-mpnet-base-v2`

- **Architecture**: MPNet (768 dimensions)
- **Training**: Fine-tuned on 1B+ sentence pairs with **cosine similarity** as the training signal
- **Benchmark**: Highest-performing general-purpose sentence embedding model in MTEB (as of training cutoff)
- **Vertex AI equivalent**: `textembedding-gecko@003` (768d), same dimensionality, same cosine similarity metric

---

## 2. Production Migration to Vertex AI Vector Search (Matching Engine)

### Current Local Architecture

```
User Query
    │
    ▼
EmbeddingService          ← sentence-transformers (all-mpnet-base-v2)
(src/embeddings.py)
    │ 768d float32 vector
    ▼
QueryExpander             ← Mocked GenerativeModel (gemini-1.5-flash)
(src/query_expander.py)   [Strategy B only]
    │
    ▼
VectorStore               ← FAISS IndexFlatIP (in-memory, single node)
(src/vector_store.py)
    │ cosine similarity
    ▼
RetrievalEngine           ← Orchestrates both strategies
(src/retrieval.py)
```

### Target Production Architecture (Vertex AI)

```
User Query
    │
    ▼
Vertex AI TextEmbeddingModel    ← textembedding-gecko@003 (768d)
(via google-cloud-aiplatform)
    │
    ▼
Vertex AI GenerativeModel       ← gemini-1.5-flash (real, not mocked)
(Query Expansion — Strategy B)
    │
    ▼
Vertex AI Vector Search         ← Managed Matching Engine (ScaNN)
(Index Endpoint API)            ← Scales to billions of vectors
    │
    ▼
Application / RAG Response
```

### Step-by-Step Migration Guide

#### Step 1: Replace EmbeddingService

**Before (local):**
```python
# src/embeddings.py
from sentence_transformers import SentenceTransformer
self.model = SentenceTransformer("all-mpnet-base-v2")
embeddings = self.model.encode(texts, normalize_embeddings=True)
```

**After (Vertex AI):**
```python
# src/embeddings_vertex.py
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

vertexai.init(project="your-gcp-project", location="us-central1")

class VertexEmbeddingService:
    def __init__(self):
        self.model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        self.dimension = 768

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        inputs = [TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT") for text in texts]
        embeddings = self.model.get_embeddings(inputs)
        return [e.values for e in embeddings]

    def get_embedding(self, text: str) -> list[float]:
        inputs = [TextEmbeddingInput(text, "RETRIEVAL_QUERY")]
        return self.model.get_embeddings(inputs)[0].values
```

#### Step 2: Replace QueryExpander (remove mock)

**Before (mocked):**
```python
# Tests use: patch("src.query_expander.GenerativeModel")
```

**After (production):**
```python
# src/query_expander.py
import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project="your-gcp-project", location="us-central1")
self.model = GenerativeModel("gemini-1.5-flash")
response = self.model.generate_content(prompt)
```

#### Step 3: Prepare and Upload Data to Cloud Storage

Export embeddings and metadata from FAISS to JSONL:

```python
# scripts/export_for_vertex.py
import json

records = []
for i, doc in enumerate(vector_store.documents):
    records.append({
        "id": doc["id"],
        "embedding": vector_store.index.reconstruct(i).tolist(),
    })

with open("embeddings.json", "w") as f:
    for record in records:
        f.write(json.dumps(record) + "\n")

# Upload to GCS
# gsutil cp embeddings.json gs://your-bucket/vector-search/embeddings.json
```

#### Step 4: Create and Deploy Vertex AI Vector Search Index

```python
from google.cloud import aiplatform

# Create index
my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="rag-pipeline-index",
    contents_delta_uri="gs://your-bucket/vector-search/",
    dimensions=768,
    approximate_neighbors_count=150,
    distance_measure_type="DOT_PRODUCT_DISTANCE",  # = cosine on normalized vectors
)

# Deploy to endpoint
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="rag-pipeline-endpoint",
    public_endpoint_enabled=True,
)

my_index_endpoint.deploy_index(
    index=my_index,
    deployed_index_id="rag_pipeline_deployed",
)
```

#### Step 5: Replace VectorStore Search Call

**Before (FAISS local):**
```python
scores, indices = self.index.search(query_vector, top_k)
```

**After (Vertex AI Vector Search):**
```python
response = my_index_endpoint.find_neighbors(
    deployed_index_id="rag_pipeline_deployed",
    queries=[query_embedding],
    num_neighbors=top_k,
)
neighbors = response[0]  # List of MatchNeighbor objects
results = [(n.id, n.distance) for n in neighbors]
```

### Migration Comparison

| Aspect | Local (FAISS) | Production (Vertex AI Vector Search) |
|--------|--------------|---------------------------------------|
| **Scale** | ~10K docs (in-memory) | Billions of vectors |
| **Availability** | Single process | Managed, multi-zone HA |
| **Search algorithm** | Exact (IndexFlatIP) | ScaNN (approximate, tunable recall) |
| **Persistence** | Manual (save/load index file) | Managed (Cloud Storage + GCS) |
| **Metadata filtering** | Manual (post-filter) | Native pre/post filtering |
| **Latency** | Sub-ms (local) | ~10–50ms (network round-trip) |
| **Scaling ops** | Manual sharding | Automatic |
| **Cost** | Free (compute only) | Per-node/hour + query fees |

### Key Considerations

1. **Distance metric mapping**: Our FAISS `IndexFlatIP` on normalized vectors uses cosine similarity → maps to `DOT_PRODUCT_DISTANCE` in Vertex AI (same metric for normalized vectors).

2. **Streaming updates**: For frequently changing corpora, use Vertex AI Vector Search Streaming Ingestion to update the index without full re-indexing.

3. **Recall vs latency tradeoff**: Vertex AI Vector Search uses approximate nearest neighbor (ScaNN) by default. Tune `approximate_neighbors_count` and `leaf_nodes_to_search_percent` to balance recall and latency.

4. **A/B testing during migration**: Run FAISS and Vertex AI in parallel for shadow testing before cutover. Compare retrieved document IDs and scores to validate recall parity.

5. **IAM permissions**: The service account running the pipeline needs `aiplatform.indexes.query` and `aiplatform.indexEndpoints.queryVectors` roles.
