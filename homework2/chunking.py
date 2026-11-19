
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

MODEL_NAME = "Snowflake/snowflake-arctic-embed-s"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def count_tokens(text: str) -> int:
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False
        )
    )


splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=60,
    length_function=count_tokens,
    separators=[
        "\n## ",
        "\n### ",
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


def create_chunks(documents):

    chunks = []

    for doc in documents:

        text_chunks = splitter.split_text(doc["content"])

        for chunk_id, chunk in enumerate(text_chunks):

            # Validate actual model input length
            actual_tokens = len(
                tokenizer.encode(
                    chunk,
                    add_special_tokens=True
                )
            )

            assert actual_tokens <= 512, (
                f"Chunk exceeds model limit: {actual_tokens}"
            )

            chunks.append({
                "filename": doc["filename"],
                "chunkId": chunk_id,
                "content": chunk
            })

    return chunks
