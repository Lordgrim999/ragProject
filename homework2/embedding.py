
from sentence_transformers import SentenceTransformer
from chunking import create_chunks

MODEL_NAME = "Snowflake/snowflake-arctic-embed-s"


class EmbeddingGenerator:

    def __init__(self, batch_size=128):

        self.model  = SentenceTransformer( MODEL_NAME,device="mps", model_kwargs={"torch_dtype": "float16"})
        self.batch_size = batch_size

    def generate_embeddings(self, documents: list[dict]):

        # Step 1: Generate chunks using existing chunking module
        chunks = create_chunks(documents)

        if not chunks:
            return []

        # Step 2: Extract chunk content
        texts = [chunk["content"] for chunk in chunks]

        # Step 3: Validate actual model input length
        for chunk, text in zip(chunks, texts):

            token_count = len(
                self.model.tokenizer.encode(
                    text,
                    add_special_tokens=True
                )
            )

            assert token_count <= self.model.max_seq_length, (
                f"Chunk exceeds model limit: "
                f"{chunk['filename']} - {chunk['chunkId']}"
            )

        # Step 4: Generate embeddings in batches
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        # Step 5: Attach embeddings to original chunk records
        embedded_chunks = []

        for chunk, vector in zip(chunks, vectors):

            embedded_chunks.append({
                **chunk,
                "embedding": vector.tolist()
            })

        return embedded_chunks
