"""Prepare local BIRD archives, audited metadata, and DB-disjoint agent splits."""
import argparse
import csv
import hashlib
import io
import json
import math
import random
import shutil
import sqlite3
import stat
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent.common import write_json, write_jsonl
from agent.tools import run_sql
from data.build_bird import PROMPT


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            result.update(block)
    return result.hexdigest()


def extract_archive(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if "__MACOSX" in member.filename or Path(member.filename).name.startswith("._"):
                continue
            path = (destination / member.filename).resolve()
            if not path.is_relative_to(destination) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Unsafe zip member: " + member.filename)
            if member.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            elif not path.is_file() or path.stat().st_size != member.file_size:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name + ".extracting")
                with bundle.open(member) as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 8 * 1024 ** 2)
                temporary.replace(path)


def read_descriptions(directory):
    descriptions = {}
    for path in sorted(Path(directory).glob("*.csv")):
        raw = path.read_bytes()
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                content = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        for row in csv.DictReader(io.StringIO(content)):
            name = (row.get("original_column_name") or row.get("column_name") or "").strip()
            # Some official CSVs put value explanations in unnamed/overflow columns.
            values = []
            for key, value in row.items():
                if key == "original_column_name":
                    continue
                for item in value if isinstance(value, list) else [value]:
                    if item and str(item).strip():
                        values.append(str(item).strip())
            descriptions[(path.stem.lower(), name.lower())] = {
                "description": " | ".join(dict.fromkeys(values)),
                "source": str(path.resolve()), "encoding": encoding}
    return descriptions


def build_metadata(db_path):
    db_path = Path(db_path)
    descriptions = read_descriptions(db_path.parent / "database_description")
    con = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    tables, columns = [], []
    quote = lambda value: '"' + value.replace('"', '""') + '"'
    try:
        for name, ddl in con.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"):
            info = list(con.execute(f"PRAGMA table_info({quote(name)})"))
            fks = list(con.execute(f"PRAGMA foreign_key_list({quote(name)})"))
            tables.append({"name": name, "ddl": ddl, "columns": [r[1] for r in info],
                           "primary_keys": [r[1] for r in info if r[5]],
                           "foreign_keys": [{"column": r[3], "target_table": r[2], "target_column": r[4]} for r in fks]})
            for _, column, dtype, _, _, pk in info:
                columns.append({"table": name, "name": column, "type": dtype, "primary_key": bool(pk),
                                **descriptions.get((name.lower(), column.lower()), {})})
    finally:
        con.close()
    return {"metadata_version": 1, "db_id": db_path.stem, "tables": tables, "columns": columns}


def split_rows(rows, seed=0, val_fraction=.1, sft_fraction=.7):
    dbs = sorted({r["db_id"] for r in rows})
    if len(dbs) < 2:
        raise ValueError("Need at least two databases for DB-disjoint validation")
    val_dbs = set(random.Random(seed).sample(dbs, min(len(dbs) - 1, max(1, math.ceil(len(dbs) * val_fraction)))))
    grouped = defaultdict(list)
    for row in rows:
        # Keep exact-SQL paraphrases together, preserving literals and SQL case.
        grouped[(row["db_id"], row["gold_sql"].strip())].append(row)
    splits = {"sft": [], "rl": [], "val": []}
    for key in sorted(grouped):
        digest = hashlib.sha256((str(seed) + repr(key)).encode()).digest()
        role = "val" if key[0] in val_dbs else "sft" if int.from_bytes(digest[:8], "big") / 2 ** 64 < sft_fraction else "rl"
        splits[role].extend({**r, "split": role} for r in grouped[key])
    return splits


def download_filtered(destination):
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 502, 503, 504])))
    info = session.get("https://huggingface.co/api/datasets/birdsql/bird23-train-filtered", timeout=30)
    info.raise_for_status()
    info = info.json()
    candidates = [r["rfilename"] for r in info["siblings"] if r["rfilename"].endswith((".jsonl", ".json"))]
    candidates = [p for p in candidates if "train" in p.lower() and "meaning" not in p.lower()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one train annotation file, found {candidates}; supply --filtered explicitly")
    url = f"https://huggingface.co/datasets/birdsql/bird23-train-filtered/resolve/{info['sha']}/{candidates[0]}"
    response = session.get(url, timeout=60)
    response.raise_for_status()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    write_json(str(destination) + ".source.json", {"url": url, "revision": info["sha"], "sha256": sha256(destination)})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/raw/train.zip")
    parser.add_argument("--extract-root", default="data/raw/bird_agent_train")
    parser.add_argument("--filtered", default="data/raw/bird23_train_filtered.jsonl")
    parser.add_argument("--download-filtered", action="store_true")
    parser.add_argument("--out", default="data/processed/bird_agent")
    parser.add_argument("--validate-gold", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dev-root", default="data/raw/dev_extract/dev_20240627")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    filtered = Path(args.filtered)
    if args.download_filtered and not filtered.exists():
        download_filtered(filtered)
    if not filtered.is_file():
        parser.error("Official filtered annotations required; pass --download-filtered or --filtered PATH")
    extract_root = Path(args.extract_root)
    print("[prepare] extracting local outer archive", flush=True)
    extract_archive(args.archive, extract_root)
    nested = list(extract_root.rglob("train_databases.zip"))
    if len(nested) != 1:
        raise ValueError("Expected one nested train_databases.zip")
    print("[prepare] extracting local databases", flush=True)
    db_root = extract_root / "databases"
    extract_archive(nested[0], db_root)
    dbs = {p.stem: p.resolve() for p in db_root.rglob("*.sqlite") if "__MACOSX" not in str(p)}
    raw = filtered.read_text(encoding="utf-8")
    annotations = json.loads(raw) if raw.lstrip().startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
    if isinstance(annotations, dict):
        annotations = annotations.get("train", annotations.get("data"))
    out = Path(args.out).resolve()
    metas = {}
    for db_id in sorted({r["db_id"] for r in annotations}):
        if db_id not in dbs:
            raise ValueError("Database missing: " + db_id)
        meta = build_metadata(dbs[db_id])
        path = out / "metadata" / f"{db_id}.json"
        write_json(path, meta)
        metas[db_id] = (meta, path)
    rows, excluded, seen = [], [], set()
    for item in annotations:
        key = (item["db_id"], " ".join(item["question"].lower().split()))
        if key in seen or not item.get("SQL", "").strip() or not item["question"].strip():
            excluded.append({"db_id": item["db_id"], "question": item["question"], "reason": "duplicate_or_empty"})
            continue
        seen.add(key)
        evidence = item.get("evidence") or ""
        metadata, path = metas[item["db_id"]]
        row_id = hashlib.sha256(json.dumps([item["db_id"], item["question"], item["SQL"]], ensure_ascii=False).encode()).hexdigest()[:20]
        rows.append({"id": "bird_train_" + row_id, "db_id": item["db_id"], "db_path": str(dbs[item["db_id"]]),
                     "question": item["question"], "evidence": evidence, "gold_sql": item["SQL"],
                     "metadata_path": str(path), "dataset": "bird23-filtered",
                     "prompt": PROMPT.format(schema="\n".join(t["ddl"] for t in metadata["tables"]), question=item["question"],
                                               evidence=f"\n### Evidence\n{evidence}\n" if evidence else "")})
    if args.validate_gold:
        def validate(row):
            return run_sql(row["db_path"], row["gold_sql"], operation="validate", timeout=30)
        kept = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, (row, result) in enumerate(zip(rows, pool.map(validate, rows)), 1):
                if result["ok"]:
                    kept.append(row)
                else:
                    excluded.append({"id": row["id"], "reason": "gold_failure", "result": result})
                if i % 100 == 0:
                    print(f"[gold] {i}/{len(rows)} kept={len(kept)}", flush=True)
        rows = kept
    splits = split_rows(rows, args.seed)
    for role, data in splits.items():
        write_jsonl(out / f"{role}.jsonl", data)
    dev_root = Path(args.dev_root)
    if (dev_root / "dev.json").is_file():
        dev_rows = []
        for item in json.loads((dev_root / "dev.json").read_text()):
            db_id = item["db_id"]
            db = (dev_root / "dev_databases" / db_id / f"{db_id}.sqlite").resolve()
            path = out / "dev_metadata" / f"{db_id}.json"
            if not path.exists():
                write_json(path, build_metadata(db))
            dev_rows.append({"id": f"bird_dev_{item['question_id']}", "db_id": db_id,
                             "db_path": str(db), "metadata_path": str(path), "question": item["question"],
                             "evidence": item.get("evidence", ""), "gold_sql": item["SQL"],
                             "difficulty": item.get("difficulty", ""), "split": "test", "dataset": "bird-dev"})
        write_jsonl(out / "test.jsonl", dev_rows)
    manifest = {"version": 1, "archive": str(Path(args.archive).resolve()), "archive_sha256": sha256(args.archive),
                "annotations_sha256": sha256(filtered), "annotations": str(filtered.resolve()),
                "annotations_source": json.loads(Path(str(filtered) + ".source.json").read_text()) if Path(str(filtered) + ".source.json").exists() else None,
                "seed": args.seed, "gold_validated": args.validate_gold, "total": len(annotations),
                "counts": {k: len(v) for k, v in splits.items()},
                "dbs": {k: sorted({r["db_id"] for r in v}) for k, v in splits.items()}, "excluded": excluded}
    write_json(out / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ["gold_validated", "total", "counts"]}), flush=True)


if __name__ == "__main__":
    main()
