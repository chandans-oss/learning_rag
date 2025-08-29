import argparse
import json
import os
import re
import shutil
import sys
import time
import hashlib
import logging
from pathlib import Path

from typing import Iterable, List, Dict, Generator, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma

from embedding_helper import get_embedding_function

# ---------------------------
# CONFIG / LOGGING
# ---------------------------
DEFAULT_DB_DIR = "chroma2"
DEFAULT_DATA_DIR = "data2"
DEFAULT_MANIFEST = "manifest.json"
SUPPORTED_EXTS = {".pdf", ".feature", ".txt"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")


# ---------------------------
# UTILS
# ---------------------------
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def clean_metadata(md: Dict) -> Dict:
    """Ensure metadata values are primitives (str, int, float, bool, None)."""
    out = {}
    for k, v in md.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def batched(iterable: Iterable, n: int) -> Generator[list, None, None]:
    """Yield lists of size n from iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------
# LOADING & CHUNKING
# ---------------------------
def load_file(path: Path) -> List[Document]:
    """Load a single file into one or more Documents (pre-chunk)."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
        return loader.load()  # page-wise docs with metadata (source, page)
    elif ext == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
        return loader.load()
    elif ext == ".feature":
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Split by scenario/background; keep headers
        parts = re.split(r"(?=^\s*(Scenario:|Background:))", text, flags=re.MULTILINE)
        if not parts:
            return []
        # Stitch pattern splits back into full sections starting with the keyword
        sections = []
        buf = ""
        for p in parts:
            if re.match(r"^\s*(Scenario:|Background:)", p, flags=re.IGNORECASE):
                if buf.strip():
                    sections.append(buf)
                buf = p
            else:
                buf += p
        if buf.strip():
            sections.append(buf)

        docs = []
        for idx, sec in enumerate(sections):
            docs.append(
                Document(
                    page_content=sec.strip(),
                    metadata={"source": str(path), "section_index": idx},
                )
            )
        return docs
    else:
        logger.debug(f"Skipping unsupported file: {path}")
        return []


def split_docs(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    return splitter.split_documents(docs)


def iter_all_chunks(
    data_dir: Path, exts: set, chunk_size: int, chunk_overlap: int
) -> Generator[Tuple[Path, str, Document], None, None]:
    """
    Yield (filepath, file_sha256, chunk_doc) for all supported files.
    Chunks are produced per-file to keep memory low.
    """
    for file in sorted(data_dir.rglob("*")):
        if file.is_dir():
            continue
        if file.suffix.lower() not in exts:
            continue
        try:
            fhash = sha256_file(file)
            raw_docs = load_file(file)
            if not raw_docs:
                continue
            chunks = split_docs(raw_docs, chunk_size, chunk_overlap)
            # Attach source + stable file hash
            for ch in chunks:
                meta = dict(ch.metadata or {})
                meta["source"] = meta.get("source", str(file))
                meta["file_sha256"] = fhash
                ch.metadata = meta
                yield file, fhash, ch
        except Exception as e:
            logger.exception(f"Failed to process {file}: {e}")


# ---------------------------
# MANIFEST (RESUME SUPPORT)
# ---------------------------
def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"files": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Manifest is corrupt; starting fresh.")
        return {"files": {}}


def save_manifest(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def needs_reindex(manifest: dict, file: Path, sha: str) -> bool:
    rec = manifest["files"].get(str(file))
    if not rec:
        return True
    return rec.get("sha256") != sha


def update_manifest_record(manifest: dict, file: Path, sha: str, done: bool, chunks: int):
    manifest["files"][str(file)] = {
        "sha256": sha,
        "chunks_indexed": chunks,
        "completed": done,
        "mtime": os.path.getmtime(file),
        "updated_at": int(time.time()),
    }


# ---------------------------
# CHROMA INDEXING
# ---------------------------
def delete_file_from_chroma(db: Chroma, file_path: Path):
    """Delete old chunks for a file by source path."""
    try:
        db.delete(where={"source": str(file_path)})
        logger.info(f"Deleted old entries for {file_path}")
    except Exception as e:
        logger.warning(f"Delete failed for {file_path}: {e}")


def ensure_chunk_ids(chunks: List[Document]) -> None:
    """
    Ensure each chunk has a stable unique id = {file_sha256}:{seq}.
    If Chroma already has ids, we still overwrite with our id scheme.
    """
    counters = {}
    for ch in chunks:
        fhash = ch.metadata.get("file_sha256", "nofilehash")
        counters.setdefault(fhash, 0)
        seq = counters[fhash]
        ch.metadata["id"] = f"{fhash}:{seq}"
        counters[fhash] = seq + 1


def add_batches_to_chroma(
    db: Chroma,
    chunk_iter: Iterable[Document],
    batch_size: int,
    persist_every: int,
) -> int:
    """Add chunks to Chroma in batches; return number of chunks added."""
    total = 0
    for i, batch in enumerate(batched(chunk_iter, batch_size), start=1):
        ensure_chunk_ids(batch)
        safe_docs = [
            Document(page_content=doc.page_content, metadata=clean_metadata(doc.metadata))
            for doc in batch
        ]
        ids = [d.metadata["id"] for d in safe_docs]
        db.add_documents(safe_docs, ids=ids)
        total += len(safe_docs)

        if i % persist_every == 0:
            # with langchain_chroma, persistence is automatic
            logger.info(f"Auto-persist handled after {i} batches ({total} chunks).")

    # no explicit db.persist() call needed
    return total



# ---------------------------
# MAIN PIPELINE
# ---------------------------
def ingest(
    data_dir: str,
    db_dir: str,
    manifest_path: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    persist_every: int,
    exts: List[str],
    reset: bool,
    max_files: int | None,
):
    data_path = Path(data_dir)
    db_path = Path(db_dir)
    manifest_file = Path(manifest_path)
    exts_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}

    if reset and db_path.exists():
        shutil.rmtree(db_path)
        logger.info("Cleared Chroma database directory.")

    manifest = load_manifest(manifest_file)

    db = Chroma(
        persist_directory=str(db_path),
        embedding_function=get_embedding_function(),
    )

    # Plan files to process
    files = [p for p in sorted(data_path.rglob("*")) if p.is_file() and p.suffix.lower() in exts_set]
    if max_files:
        files = files[:max_files]
    logger.info(f"Found {len(files)} file(s) to consider.")

    processed = 0
    total_added = 0

    for file in files:
        fhash = sha256_file(file)
        reindex = needs_reindex(manifest, file, fhash)

        if not reindex and manifest["files"][str(file)].get("completed"):
            logger.info(f"✔ Skipping up-to-date file: {file}")
            continue

        # If file changed, delete prior rows for that file
        delete_file_from_chroma(db, file)

        # Stream chunks for this file and index
        def file_chunk_iter() -> Generator[Document, None, None]:
            raw_docs = load_file(file)
            if not raw_docs:
                return
            chunks = split_docs(raw_docs, chunk_size, chunk_overlap)
            for ch in chunks:
                meta = dict(ch.metadata or {})
                meta["source"] = meta.get("source", str(file))
                meta["file_sha256"] = fhash
                ch.metadata = meta
                yield ch

        added = add_batches_to_chroma(
            db=db,
            chunk_iter=file_chunk_iter(),
            batch_size=batch_size,
            persist_every=persist_every,
        )
        total_added += added
        processed += 1

        update_manifest_record(
            manifest, file, fhash, done=True, chunks=added
        )
        save_manifest(manifest_file, manifest)
        logger.info(f"Indexed {added} chunks from {file}")

    logger.info(f"✅ Done. Files processed: {processed}. Chunks added: {total_added}.")


# ---------------------------
# CLI
# ---------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Batch & resumable RAG ingestion for PDFs / .feature / .txt")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Folder with documents")
    p.add_argument("--db-dir", default=DEFAULT_DB_DIR, help="Chroma persistence directory")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Path to manifest.json (resume checkpoints)")
    p.add_argument("--batch-size", type=int, default=1000, help="Chunks per batch")
    p.add_argument("--persist-every", type=int, default=1, help="Persist after N batches")
    p.add_argument("--chunk-size", type=int, default=1000, help="Characters per chunk")
    p.add_argument("--chunk-overlap", type=int, default=150, help="Characters overlap")
    p.add_argument("--ext", action="append", default=list(SUPPORTED_EXTS),
                   help="File extension to include (repeatable). Defaults to pdf/feature/txt")
    p.add_argument("--reset", action="store_true", help="Clear DB before indexing")
    p.add_argument("--max-files", type=int, default=None, help="Limit number of files for a dry run")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ingest(
        data_dir=args.data_dir,
        db_dir=args.db_dir,
        manifest_path=args.manifest,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        persist_every=args.persist_every,
        exts=args.ext,
        reset=args.reset,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
