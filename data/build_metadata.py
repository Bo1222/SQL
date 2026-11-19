"""Build per-database metadata stores (schema + catalog + relationships + index)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.catalog_store import build_store  # noqa: E402


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_databases(sources):
    """Return {db_id: db_path} from every dataset file given."""
    found = {}
    for source in sources:
        path = Path(source)
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            db_id, db_path = row.get("db_id"), row.get("db_path")
            if db_id and db_path:
                found.setdefault(db_id, db_path)
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=[
        "data/processed/bird_agent/rl.jsonl",
        "data/processed/bird_agent/val.jsonl",
        "data/processed/bird_agent/test.jsonl",
        "data/processed/bird_agent/full_sft_20261026/full_tool.jsonl",
    ])
    p.add_argument("--out", default="artifacts/catalog")
    p.add_argument("--manifest", default="artifacts/manifests/catalog_manifest.json")
    a = p.parse_args()

    databases = collect_databases(a.data)
    manifests, missing = [], []
    for db_id, db_path in sorted(databases.items()):
        if not Path(db_path).is_file():
            missing.append({"db_id": db_id, "db_path": db_path})
            continue
        manifests.append(build_store(db_path, a.out))

    summary = {
        "stores": len(manifests),
        "missing_databases": missing,
        "tables": sum(m["tables"] for m in manifests),
        "columns": sum(m["columns"] for m in manifests),
        "relationships": sum(m["relationships"] for m in manifests),
        "descriptions_matched": sum(m["descriptions_matched"] for m in manifests),
        "catalog_match_rate": round(
            sum(m["descriptions_matched"] for m in manifests) / max(sum(m["columns"] for m in manifests), 1), 4),
        "stores_detail": manifests,
    }
    out = Path(a.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("stores", "tables", "columns", "relationships",
                                              "catalog_match_rate", "missing_databases")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
