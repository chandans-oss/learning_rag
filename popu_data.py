import argparse
import os
import shutil
import logging
from pathlib import Path
import re

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma

from embedding_helper import get_embedding_function


# ---------------------------
# CONFIG
# ---------------------------
CHROMA_DB_PATH = "chroma"
FILE_DATA_PATH = "data"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------
# UNIFIED LOADER
# ---------------------------
def load_file_unified(path: Path, chunk_size: int = 800, chunk_overlap: int = 50) -> list[Document]:
    """
    Unified loader for PDFs, feature files, and text files.
    - .pdf: uses PyPDFLoader
    - .feature: split scenario-wise; long scenarios are chunked
    - .txt: plain text split
    """
    ext = path.suffix.lower()
    documents = []

    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
        documents = loader.load()

    elif ext == ".feature":
        text = Path(path).read_text(encoding="utf-8")
        # Split by Scenario or Background
        scenarios = re.split(r"(?=Scenario:|Background:)", text)
        for i, scenario in enumerate(scenarios):
            if not scenario.strip():
                continue
            doc = Document(page_content=scenario.strip(), metadata={"source": str(path), "scenario": i})
            documents.append(doc)

    elif ext == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
        documents = loader.load()

    else:
        logger.warning(f"Skipping unsupported file type: {path}")

    return split_documents(documents, chunk_size, chunk_overlap)


def load_documents() -> list[Document]:
    """Load all supported documents from the data folder."""
    all_docs = []
    for file in Path(FILE_DATA_PATH).glob("**/*"):
        if file.suffix.lower() in [".pdf", ".feature", ".txt"]:
            docs = load_file_unified(file)
            all_docs.extend(docs)
    return all_docs


# ---------------------------
# SPLITTING
# ---------------------------
def split_documents(documents: list[Document], chunk_size=800, chunk_overlap=80) -> list[Document]:
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


# ---------------------------
# CHUNK IDS
# ---------------------------
def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    """Assign unique IDs to each document chunk."""
    for i, chunk in enumerate(chunks):
        chunk.metadata["id"] = f"{chunk.metadata.get('source', 'doc')}_{i}"
    return chunks


# ---------------------------
# CLEAN METADATA
# ---------------------------
def clean_metadata(metadata: dict) -> dict:
    """Ensure metadata is JSON-serializable (convert lists/objects to strings)."""
    cleaned = {}
    for k, v in metadata.items():
        if isinstance(v, (list, dict)):
            cleaned[k] = str(v)  # convert complex objects to string
        else:
            cleaned[k] = v
    return cleaned


# ---------------------------
# CHROMA FUNCTIONS
# ---------------------------
def add_to_chroma(chunks: list[Document]):
    """Add document chunks into Chroma DB, avoiding duplicates."""
    db = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    logger.info(f"Number of existing document chunks in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if not isinstance(chunk.metadata, dict):
            chunk.metadata = {"id": str(chunk.metadata)}

        chunk_id = chunk.metadata.get("id")
        if chunk_id not in existing_ids:
            new_chunks.append(
                Document(
                    page_content=chunk.page_content,
                    metadata=clean_metadata(chunk.metadata)
                )
            )

    if new_chunks:
        logger.info(f"Adding new document chunks: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata.get("id") for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        logger.info("No new document chunks to add")


def clear_database():
    """Delete Chroma database folder."""
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
        logger.info("Clearing Database")


# ---------------------------
# MAIN
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Clear DB before running")
    args = parser.parse_args()

    if args.reset:
        clear_database()

    documents = load_documents()
    logger.info(f"Loaded {len(documents)} chunks from files")
    add_to_chroma(documents)


if __name__ == "__main__":
    main()
