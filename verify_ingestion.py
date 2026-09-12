"""Interactive Ingestion & Vector DB Verification Tool.

Run this script to inspect what is currently stored in Qdrant and test semantic search.
Usage:
    python verify_ingestion.py
"""

import json
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from app.config.settings import get_settings
from app.utils.progress import get_progress_tracker


def inspect_qdrant():
    settings = get_settings()
    print("=" * 70)
    print("1. CHECKING JOB REGISTRY (API Layer)")
    print("=" * 70)

    tracker = get_progress_tracker()
    jobs = tracker.list_jobs(page=1, page_size=10)
    total_jobs = tracker.count_jobs()

    print(f"Total Ingestion Jobs Registered: {total_jobs}")
    for j in jobs[:5]:
        print(f"  • Job ID: {j.job_id}")
        print(f"    Source: {j.source_type} | Status: {j.status.value} | Progress: {j.progress}% | Chunks: {j.chunk_count}")

    print("\n" + "=" * 70)
    print("2. CHECKING STORED VECTORS IN QDRANT (Team 4b Database)")
    print("=" * 70)

    client = QdrantClient(path=settings.QDRANT_PATH)
    collection_name = settings.QDRANT_COLLECTION

    try:
        col_info = client.get_collection(collection_name)
        print(f"Collection Name: '{collection_name}'")
        print(f"Total Points in Qdrant: {col_info.points_count}")
        print(f"Vector Dimensions: {col_info.config.params.vectors.size} (Expected: 384)")
        print(f"Distance Metric: {col_info.config.params.vectors.distance} (Expected: Cosine)")
    except Exception as e:
        print(f"Could not retrieve collection '{collection_name}': {e}")
        return

    # Scroll first 3 points to inspect exact payload structure
    scroll_result = client.scroll(
        collection_name=collection_name,
        limit=3,
        with_payload=True,
        with_vectors=True,
    )
    points = scroll_result[0]

    print(f"\nSample Ingested Points (showing up to 2 of {len(points)} fetched):")
    for idx, pt in enumerate(points[:2]):
        print(f"\n--- Point {idx + 1} ---")
        print(f"Point ID (chunk_id): {pt.id}")
        print(f"Vector preview (first 5 of 384 floats): {pt.vector[:5]} ...")
        print(f"Text Content: \"{pt.payload.get('text', '')[:120]}...\"")
        print("Metadata Object:")
        print(json.dumps(pt.payload.get("metadata", {}), indent=4))

    print("\n" + "=" * 70)
    print("3. LIVE SEMANTIC SEARCH TEST (Simulating Team 4b)")
    print("=" * 70)

    query = "How does gradient descent find a local minimum?"
    print(f"Searching for Query: \"{query}\"")

    model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    query_vec = model.encode(query, normalize_embeddings=True).tolist()

    search_res = client.query_points(
        collection_name=collection_name,
        query=query_vec,
        limit=2,
    )

    if not search_res.points:
        print("No search results found yet. Ingest a document first via http://localhost:8000/docs!")
    else:
        for rank, hit in enumerate(search_res.points, start=1):
            print(f"\nRank {rank} (Cosine Similarity Score: {hit.score:.4f}):")
            print(f"Text: \"{hit.payload.get('text', '')[:140]}...\"")
            meta = hit.payload.get("metadata", {})
            print(f"Source Type: {meta.get('source_type')}")
            if meta.get("source_type") == "pdf":
                print(f"Page: {meta.get('page_number')}, File: {meta.get('filename')}")
                print(f"Headings: {meta.get('headings')}")
            elif meta.get("source_type") in ("mp4", "youtube"):
                print(f"Time: {meta.get('start_timestamp')}s -> {meta.get('end_timestamp', meta.get('duration'))}s")
                if meta.get("visual_slide_text"):
                    print(f"Slide OCR: {meta.get('visual_slide_text')}")


if __name__ == "__main__":
    inspect_qdrant()

