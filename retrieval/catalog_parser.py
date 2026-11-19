"""Parse BIRD database_description CSVs into catalog records.

The CSV header differs across BIRD versions, so the parser reads the real
header and maps aliases instead of assuming a fixed schema.
"""
import csv
import io
from pathlib import Path

COLUMN_ALIASES = ["original_column_name", "column_name", "column"]
DESCRIPTION_ALIASES = ["column_description", "description", "column_meaning"]
VALUE_ALIASES = ["value_description", "value_meaning", "values"]
TABLE_ALIASES = ["table_name", "table"]


def _decode(raw):
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace"), "utf-8-replace"


def _pick(row, aliases):
    for key in aliases:
        if key in row and row[key] is not None:
            value = str(row[key]).strip()
            if value:
                return value
    return ""


def parse_descriptions(db_dir):
    """Return ({ (table_lower, column_lower): record }, stats)."""
    records = {}
    stats = {"files": 0, "rows": 0, "unmapped_headers": [], "encodings": []}
    for path in sorted(Path(db_dir).glob("*.csv")):
        content, encoding = _decode(path.read_bytes())
        stats["encodings"].append({"file": path.name, "encoding": encoding})
        reader = csv.DictReader(io.StringIO(content))
        header = [h.strip().lower() for h in (reader.fieldnames or [])]
        column_key = next((h for h in header if h in COLUMN_ALIASES), None)
        if column_key is None:
            stats["unmapped_headers"].append({"file": path.name, "header": header})
            continue
        stats["files"] += 1
        for row in reader:
            row = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
            column = _pick(row, COLUMN_ALIASES)
            if not column:
                continue
            table = _pick(row, TABLE_ALIASES) or path.stem
            description = _pick(row, DESCRIPTION_ALIASES)
            value = _pick(row, VALUE_ALIASES)
            key = (table.lower(), column.lower())
            if key in records:
                continue
            records[key] = {
                "table": table,
                "column": column,
                "original_column_name": column,
                "description": description,
                "value_description": value,
                "source": f"bird_database_description/{path.name}",
            }
            stats["rows"] += 1
    return records, stats


def attach_descriptions(schema, descriptions):
    """Attach catalog text to real schema columns; return (catalog, unmatched)."""
    catalog, matched = [], set()
    for table in schema["tables"]:
        for column in table["columns"]:
            key = (table["name"].lower(), column["name"].lower())
            record = descriptions.get(key)
            if record:
                matched.add(key)
                catalog.append({
                    "table": table["name"],
                    "column": column["name"],
                    "original_column_name": record["original_column_name"],
                    "description": record["description"],
                    "value_description": record["value_description"],
                    "data_type": column["type"],
                    "primary_key": column["primary_key"],
                    "source": record["source"],
                })
            else:
                catalog.append({
                    "table": table["name"],
                    "column": column["name"],
                    "original_column_name": column["name"],
                    "description": "",
                    "value_description": "",
                    "data_type": column["type"],
                    "primary_key": column["primary_key"],
                    "source": "",
                })
    unmatched = [{"table": t, "column": c} for (t, c) in descriptions if (t, c) not in matched]
    return catalog, unmatched
