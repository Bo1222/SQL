"""Per-database metadata store: build, load, and BM25 index."""
import json
import re
from collections import Counter
from pathlib import Path

from retrieval.catalog_parser import attach_descriptions, parse_descriptions
from retrieval.schema_extractor import extract_schema

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {"the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "by", "with", "that", "this",
             "is", "are", "was", "were", "what", "which", "how", "many", "much", "list", "give", "me",
             "please", "show", "find", "get", "all", "each", "per", "their", "its", "it", "as", "at"}


def words(text):
    """Tokenize identifiers and prose: split snake_case and camelCase."""
    text = text or ""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ").lower()
    return [w for w in TOKEN_RE.findall(text) if w not in STOPWORDS and len(w) > 1]


def _column_doc(record):
    parts = [record["table"], record["column"], record.get("original_column_name", ""),
             record.get("description", ""), record.get("value_description", "")]
    text = " ".join(p for p in parts if p)
    tokens = words(text)
    return text, tokens


def build_retrieval_index(catalog):
    docs, df = [], Counter()
    for record in catalog:
        text, tokens = _column_doc(record)
        docs.append({"table": record["table"], "column": record["column"], "text": text, "tokens": tokens})
        for token in set(tokens):
            df[token] += 1
    avgdl = sum(len(d["tokens"]) for d in docs) / max(len(docs), 1)
    return {"docs": docs, "df": dict(df), "n_docs": len(docs), "avgdl": avgdl}


def build_store(db_path, out_root, store_dirs=None):
    """Build one db store and return its manifest."""
    db_path = Path(db_path).resolve()
    db_id = db_path.stem
    schema = extract_schema(db_path)
    description_dir = db_path.parent / "database_description"
    descriptions, parse_stats = parse_descriptions(description_dir) if description_dir.is_dir() else ({}, {})
    catalog, unmatched = attach_descriptions(schema, descriptions)
    index = build_retrieval_index(catalog)
    store = Path(out_root) / db_id
    store.mkdir(parents=True, exist_ok=True)
    (store / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=1), encoding="utf-8")
    (store / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    (store / "relationships.json").write_text(
        json.dumps(schema["relationships"], ensure_ascii=False, indent=1), encoding="utf-8")
    (store / "retrieval_index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "db_id": db_id,
        "db_path": str(db_path),
        "tables": len(schema["tables"]),
        "columns": len(catalog),
        "descriptions_matched": sum(1 for c in catalog if c["source"]),
        "descriptions_unmatched": len(unmatched),
        "relationships": len(schema["relationships"]),
        "catalog_match_rate": round(sum(1 for c in catalog if c["source"]) / max(len(catalog), 1), 4),
        "description_files": parse_stats.get("files", 0),
        "description_rows": parse_stats.get("rows", 0),
        "unmapped_headers": parse_stats.get("unmapped_headers", []),
        "unmatched_descriptions": unmatched[:50],
        "store_version": "v1",
    }
    (store / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


class CatalogStore:
    """Loaded view of one db store (schema + catalog + relationships + index)."""

    def __init__(self, store_dir):
        self.dir = Path(store_dir)
        self.schema = json.loads((self.dir / "schema.json").read_text(encoding="utf-8"))
        self.catalog = json.loads((self.dir / "catalog.json").read_text(encoding="utf-8"))
        self.relationships = json.loads((self.dir / "relationships.json").read_text(encoding="utf-8"))
        self.index = json.loads((self.dir / "retrieval_index.json").read_text(encoding="utf-8"))
        self.manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        self.table_names = [t["name"] for t in self.schema["tables"]]
        self.column_types = {(c["table"], c["column"]): c["data_type"] for c in self.catalog}
        self.catalog_by_column = {(c["table"], c["column"]): c for c in self.catalog}
        self.fk_by_table = {t["name"]: t["foreign_keys"] for t in self.schema["tables"]}
        self._adjacency = None

    @property
    def adjacency(self):
        """Undirected join graph keyed by table."""
        if self._adjacency is None:
            adj = {name: [] for name in self.table_names}
            for rel in self.relationships:
                adj.setdefault(rel["from_table"], []).append(rel)
                adj.setdefault(rel["to_table"], []).append({
                    "from_table": rel["to_table"], "from_column": rel["to_column"],
                    "to_table": rel["from_table"], "to_column": rel["from_column"],
                    "source": rel.get("source", "sqlite_fk"), "confidence": rel.get("confidence", 1.0)})
            self._adjacency = adj
        return self._adjacency


def load_store(store_dir):
    return CatalogStore(store_dir)


def build_all(db_root, out_root):
    """Build stores for every <db_id>/<db_id>.sqlite under db_root."""
    manifests = []
    for sqlite_path in sorted(Path(db_root).rglob("*.sqlite")):
        if "__MACOSX" in str(sqlite_path):
            continue
        manifests.append(build_store(sqlite_path, out_root))
    return manifests
