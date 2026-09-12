# Integration Guide & Handoff Contract: Team 4a ➔ Team 4b
**Multi-Modal Educational Intelligence Copilot**  
*Document Version: 1.0.0*

---

## 1. Executive Summary
This document defines the formal integration contract for the **Vector Database and Retrieval Team (Team 4b)**.  
Team 4a ingests multi-modal materials (**PDF documents**, **MP4 lecture videos**, and **YouTube URLs**), generates dense vector embeddings, enriches chunks with deep source-specific locator metadata, and publishes records into the shared **Qdrant Vector Database**.

---

## 2. Qdrant Connection Parameters

| Parameter | Local Lightweight Mode | Docker / Staging Cluster | Description |
| :--- | :--- | :--- | :--- |
| **Qdrant URL** | Embedded path `./data/qdrant_storage` | `http://localhost:6333` | REST / gRPC endpoint |
| **Collection Name** | `education_materials` | `education_materials` | Unified knowledge base |
| **Vector Dimension** | `384` | `384` | Matches `all-MiniLM-L6-v2` |
| **Distance Metric** | `Cosine` | `Cosine` | Normalized embeddings ($\|v\|_2 = 1.0$) |

---

## 3. Vector Point Structure & Payload Schema

Every point stored in the `education_materials` collection follows this schema:

```json
{
  "id": "<UUID4 chunk_id>",
  "vector": [0.034, -0.012, 0.089, "... 384 dimensions"],
  "payload": {
    "text": "The normalized educational text content...",
    "metadata": {
      "chunk_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
      "source_type": "pdf | mp4 | youtube",
      "job_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
      "ingestion_timestamp": "2026-09-12T12:00:00.000000+00:00",
      "embedding_model": "all-MiniLM-L6-v2",
      "embedding_model_version": "v2.2.0"
    }
  }
}
```

### Source-Specific Metadata Attributes

#### A. PDF Documents (`source_type: "pdf"`)
```json
{
  "page_number": 42,
  "headings": [
    { "text": "Chapter 3: Gradient Descent", "level": 1 },
    { "text": "3.2 Learning Rate Decay", "level": 2 }
  ],
  "filename": "machine_learning_textbook.pdf",
  "chunk_position": 2,
  "has_tables": true
}
```
*Note: If tables were detected on the page, they are formatted directly in `text` as Markdown tables (`| ... |`).*

#### B. MP4 Lecture Videos (`source_type: "mp4"`)
```json
{
  "start_timestamp": 125.4,
  "end_timestamp": 165.8,
  "filename": "lecture_04_optimization.mp4",
  "frame_reference": "./storage/frames/job-id_frame_150.jpg",
  "visual_slide_text": "Gradient Descent Formula: theta = theta - alpha * nabla J(theta)"
}
```
*Note: `visual_slide_text` contains OCR extracted from lecture slides captured at keyframe scene transitions!*

#### C. YouTube URLs (`source_type: "youtube"`)
```json
{
  "start_timestamp": 312.0,
  "duration": 48.5,
  "video_title": "Backpropagation Calculus | Deep Learning Chapter 3",
  "channel_name": "3Blue1Brown",
  "source_url": "https://www.youtube.com/watch?v=Ilg3gGewQ5U",
  "publication_date": "20171103"
}
```

---

## 4. Querying Examples for Team 4b

### Python Example: Hybrid Search with Metadata Filtering
```python
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

# 1. Connect to Qdrant
client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Encode user query
query = "What is the update rule for gradient descent?"
query_vector = model.encode(query, normalize_embeddings=True).tolist()

# 3. Perform semantic search (e.g. filtered by video lectures)
search_results = client.query_points(
    collection_name="education_materials",
    query=query_vector,
    query_filter=qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="metadata.source_type",
                match=qmodels.MatchValue(value="mp4"),
            )
        ]
    ),
    limit=5,
)

for hit in search_results.points:
    payload = hit.payload
    print(f"Score: {hit.score:.4f}")
    print(f"Text: {payload['text'][:100]}...")
    print(f"Timestamp: {payload['metadata']['start_timestamp']}s -> {payload['metadata']['end_timestamp']}s")
    print(f"Slide OCR: {payload['metadata'].get('visual_slide_text')}")
```

---

## 5. Handoff Contract for Video Player Synchronization (Team 4b ➔ Team 4c)
When Team 4b synthesizes a response for the user, retrieve the `start_timestamp` and include it in your LLM citations:

`[Timestamp: 02:05](seek://125)`

Team 4c's video player will listen for `seek://{start_timestamp}` and jump directly to that exact second in the lecture video!

