import json
import logging
import math
import os
from datetime import datetime

import tiktoken
import yaml
from dotenv import load_dotenv
from openai import OpenAI

ARTICLES_DIR = "articles"
VECTOR_STORE_NAME = "OptiSigns Help Center"
STATE_FILE = "upload_state.json"
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


def parse_frontmatter(path):
    """Extract the YAML frontmatter block from a markdown file, if present."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if not content.startswith("---"):
        return {}
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        data = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as e:
        logger.warning("Failed to parse frontmatter for %s: %s", path, e)
        return {}
    return data if isinstance(data, dict) else {}


def parse_timestamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def is_newer(new_value, old_value):
    """True if new_value's timestamp is strictly after old_value's (falls back to inequality)."""
    new_dt = parse_timestamp(new_value)
    old_dt = parse_timestamp(old_value)
    if new_dt is None or old_dt is None:
        return new_value != old_value
    return new_dt > old_dt


def load_state():
    if not os.path.isfile(STATE_FILE):
        return {"vector_store_id": None, "files": {}}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read %s (%s); starting with empty state", STATE_FILE, e)
        return {"vector_store_id": None, "files": {}}
    state.setdefault("vector_store_id", None)
    state.setdefault("files", {})
    return state


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        f.write("\n")


def build_delta(filepaths, state):
    """Classify each markdown file as added, updated, or skipped based on frontmatter updated_at."""
    known_files = state.get("files", {})
    added, updated, skipped = [], [], []
    for path in filepaths:
        filename = os.path.basename(path)
        frontmatter = parse_frontmatter(path)
        updated_at = frontmatter.get("updated_at")
        prior = known_files.get(filename)
        if prior is None:
            added.append((path, filename, updated_at))
        elif is_newer(updated_at, prior.get("updated_at")):
            updated.append((path, filename, updated_at, prior.get("file_id")))
        else:
            skipped.append(filename)
    return added, updated, skipped


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


def remove_stale_file(client, vector_store_id, file_id):
    """Detach and delete a previously uploaded file version before re-uploading an update."""
    try:
        client.vector_stores.files.delete(file_id=file_id, vector_store_id=vector_store_id)
    except Exception as e:
        logger.warning("Could not remove old vector store file %s: %s", file_id, e)
    try:
        client.files.delete(file_id)
    except Exception as e:
        logger.warning("Could not delete old file object %s: %s", file_id, e)


def upload_files(client, vector_store_id, to_process):
    """Upload each (path, filename, updated_at) file individually so we can track its file_id, then batch-attach."""
    file_ids = []
    file_id_by_filename = {}
    for path, filename, _updated_at in to_process:
        with open(path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="assistants")
        file_ids.append(file_obj.id)
        file_id_by_filename[filename] = file_obj.id

    batch = client.vector_stores.file_batches.create_and_poll(
        vector_store_id=vector_store_id,
        file_ids=file_ids,
    )
    return batch, file_id_by_filename


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

    state = load_state()
    added, updated, skipped = build_delta(filepaths, state)

    vector_store = get_or_create_vector_store(client)
    state["vector_store_id"] = vector_store.id

    for _path, filename, _updated_at, old_file_id in updated:
        if old_file_id:
            remove_stale_file(client, vector_store.id, old_file_id)
        else:
            logger.warning("No prior file_id recorded for %s; skipping stale cleanup", filename)

    to_process = [(path, filename, updated_at) for path, filename, updated_at in added]
    to_process += [(path, filename, updated_at) for path, filename, updated_at, _old_id in updated]

    if to_process:
        batch, file_id_by_filename = upload_files(client, vector_store.id, to_process)
        logger.info(
            "Upload finished (status=%s): %d completed, %d failed, %d in progress",
            batch.status,
            batch.file_counts.completed,
            batch.file_counts.failed,
            batch.file_counts.in_progress,
        )

        for _path, filename, updated_at in to_process:
            state["files"][filename] = {
                "updated_at": updated_at,
                "file_id": file_id_by_filename[filename],
            }

        total_tokens, estimated_chunks = estimate_total_chunks([path for path, _f, _u in to_process])
        logger.info(
            "Estimated chunks: %d (from %d total tokens, chunk_size=%d, overlap=%d)",
            estimated_chunks,
            total_tokens,
            CHUNK_SIZE_TOKENS,
            CHUNK_OVERLAP_TOKENS,
        )
    else:
        logger.info("No new or updated files to upload")

    save_state(state)
    attach_vector_store_to_assistant(client, assistant_id, vector_store.id)

    logger.info(
        "Delta summary: %d added, %d updated, %d skipped (vector store '%s', %s)",
        len(added),
        len(updated),
        len(skipped),
        vector_store.name,
        vector_store.id,
    )


if __name__ == "__main__":
    main()
