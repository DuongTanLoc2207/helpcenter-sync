import logging
import math
import os

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

ARTICLES_DIR = "articles"
VECTOR_STORE_NAME = "OptiSigns Help Center"
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 400
TOKEN_ENCODING = "cl100k_base"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_markdown_files():
    if not os.path.isdir(ARTICLES_DIR):
        raise FileNotFoundError(f"Articles directory not found: {ARTICLES_DIR}")
    filenames = sorted(f for f in os.listdir(ARTICLES_DIR) if f.endswith(".md"))
    return [os.path.join(ARTICLES_DIR, f) for f in filenames]


def estimate_chunks_for_tokens(token_count, chunk_size=CHUNK_SIZE_TOKENS, overlap=CHUNK_OVERLAP_TOKENS):
    if token_count <= 0:
        return 0
    if token_count <= chunk_size:
        return 1
    step = chunk_size - overlap
    return 1 + math.ceil((token_count - chunk_size) / step)


def estimate_total_chunks(filepaths):
    """Estimate chunk count per file (chunk_size=800, overlap=400 tokens) using tiktoken."""
    encoding = tiktoken.get_encoding(TOKEN_ENCODING)
    total_tokens = 0
    total_chunks = 0
    for path in filepaths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = len(encoding.encode(text))
        total_tokens += tokens
        total_chunks += estimate_chunks_for_tokens(tokens)
    return total_tokens, total_chunks


def get_or_create_vector_store(client):
    for vector_store in client.vector_stores.list(limit=100).data:
        if vector_store.name == VECTOR_STORE_NAME:
            logger.info("Reusing existing vector store: %s (%s)", vector_store.name, vector_store.id)
            return vector_store

    vector_store = client.vector_stores.create(name=VECTOR_STORE_NAME)
    logger.info("Created new vector store: %s (%s)", vector_store.name, vector_store.id)
    return vector_store


def upload_files(client, vector_store_id, filepaths):
    file_streams = [open(path, "rb") for path in filepaths]
    try:
        batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id,
            files=file_streams,
        )
    finally:
        for stream in file_streams:
            stream.close()
    return batch


def attach_vector_store_to_assistant(client, assistant_id, vector_store_id):
    """Attach the vector store via file_search without dropping the assistant's other tools."""
    assistant = client.beta.assistants.retrieve(assistant_id)
    tools = [tool.model_dump() for tool in assistant.tools]
    if not any(tool["type"] == "file_search" for tool in tools):
        tools.append({"type": "file_search"})

    client.beta.assistants.update(
        assistant_id,
        tools=tools,
        tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
    )
    logger.info("Attached vector store %s to assistant %s", vector_store_id, assistant_id)


def main():
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    assistant_id = os.environ.get("OPENAI_ASSISTANT_ID")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    if not assistant_id:
        raise RuntimeError("OPENAI_ASSISTANT_ID is not set")

    client = OpenAI(api_key=api_key)

    filepaths = get_markdown_files()
    if not filepaths:
        logger.warning("No .md files found in %s", ARTICLES_DIR)
        return
    logger.info("Found %d markdown file(s) in %s", len(filepaths), ARTICLES_DIR)

    vector_store = get_or_create_vector_store(client)
    batch = upload_files(client, vector_store.id, filepaths)

    logger.info(
        "Upload finished (status=%s): %d completed, %d failed, %d in progress",
        batch.status,
        batch.file_counts.completed,
        batch.file_counts.failed,
        batch.file_counts.in_progress,
    )

    total_tokens, estimated_chunks = estimate_total_chunks(filepaths)
    logger.info(
        "Estimated chunks: %d (from %d total tokens, chunk_size=%d, overlap=%d)",
        estimated_chunks,
        total_tokens,
        CHUNK_SIZE_TOKENS,
        CHUNK_OVERLAP_TOKENS,
    )

    attach_vector_store_to_assistant(client, assistant_id, vector_store.id)

    logger.info(
        "Done: %d file(s) uploaded to vector store '%s' (%s) and attached to assistant %s",
        batch.file_counts.completed,
        vector_store.name,
        vector_store.id,
        assistant_id,
    )


if __name__ == "__main__":
    main()
