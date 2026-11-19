
from elasticsearch import Elasticsearch, helpers
from chunking import create_chunks

es = Elasticsearch("http://localhost:9200")

INDEX_NAME = "llm_zoomcamp_lessons"


def ingest_documents(documents: list[dict]):

    # Step 1: Convert documents into chunks
    chunks = create_chunks(documents)

    actions = []

    # Step 2: Prepare bulk indexing actions
    for chunk in chunks:

        file_name = chunk["filename"]
        chunk_id = chunk["chunkId"]

        actions.append({
            "_op_type": "index",
            "_index": INDEX_NAME,
            "_id": f"{file_name}::{chunk_id}",
            "_source": chunk
        })

    # Step 3: Remove previously indexed chunks
    # belonging to the files being re-ingested
    file_names = list(set(
        doc["filename"] for doc in documents
    ))

    for file_name in file_names:
        es.delete_by_query(
            index=INDEX_NAME,
            query={
                "term": {
                    "filename": file_name
                }
            },
            refresh=False
        )

    # Step 4: Bulk index new chunks
    success, errors = helpers.bulk(
        es,
        actions,
        refresh="wait_for",
        raise_on_error=False
    )

    print("Original documents:", len(documents))
    print("Generated chunks:", len(chunks))
    print("Successfully indexed:", success)
    print("Failed:", len(errors))

    return success, errors
