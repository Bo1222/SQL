"""Evidence in actual encoded Observation text, never retrieval_metadata.

Offsets address the decoded JSON `observation` field in the named segment;
raw_segment_sha256 binds the evidence to the exact model-visible wrapper.
"""
import json
from e1_alignment.common import sha256_text


def payload(text):
    if not text.startswith("<observation>") or "</observation>" not in text:
        return {}
    try:
        return json.loads(text[len("<observation>"):text.rindex("</observation>")])
    except (ValueError, TypeError):
        return {}


def visible_evidence(segments, before, schema):
    found = {"tables": {}, "columns": {}, "joins": {}}
    for index, segment in enumerate(segments[:before]):
        if segment.get("kind") != "observation":
            continue
        raw = segment.get("text", "")
        body = payload(raw)
        text = body.get("observation", "")
        if body.get("status") != "ok" or not isinstance(text, str):
            continue
        table, section, offset = None, None, 0
        for line in text.splitlines(keepends=True):
            value = line.rstrip("\r\n")
            if value in ("Tables:", "Relationships:", "Relevant Values:"):
                section = value; table = None
            evidence = {"segment": index, "field": "observation", "start": offset,
                        "end": offset + len(value), "text": value,
                        "raw_segment_sha256": sha256_text(raw)}
            if section == "Tables:":
                if value.startswith("- "):
                    name = value[2:].strip().lower()
                    table = name if name in schema else None
                    if table:
                        found["tables"].setdefault(table, []).append(evidence)
                elif table and value.startswith("  - "):
                    rest = value[4:].lower()
                    hits = [c for c in schema[table] if rest == c or rest.startswith(c + " ")]
                    if hits:
                        column = max(hits, key=len)
                        found["columns"].setdefault(f"{table}.{column}", []).append(evidence)
            elif section == "Relationships:" and value.startswith("- "):
                # Match actual, schema-known endpoints, including quoted/space names.
                relation = value[2:].split("   [", 1)[0].lower().strip()
                if " = " in relation:
                    a, b = [x.strip() for x in relation.split(" = ", 1)]
                    if any(a == f"{t}.{c}" for t in schema for c in schema[t]) and any(
                            b == f"{t}.{c}" for t in schema for c in schema[t]):
                        key = "=".join(sorted([a, b]))
                        evidence = {**evidence, "source": value.split("   [", 1)[-1].rstrip("]")
                                    if "   [" in value else "unspecified"}
                        found["joins"].setdefault(key, []).append(evidence)
            offset += len(line)
    return found


def coverage(requirements, segments, before, schema):
    evidence = visible_evidence(segments, before, schema)
    missing = {"tables": [], "columns": [], "joins": []}
    for field in ("tables", "columns"):
        missing[field] = [item for item in requirements.get(field, []) if item not in evidence[field]]
    required_joins = []
    for join in requirements.get("joins", []):
        a, b = join["left"], join["right"]
        key = "=".join(sorted([f"{a['table']}.{a['column']}", f"{b['table']}.{b['column']}"]))
        required_joins.append(key)
        if key not in evidence["joins"]:
            missing["joins"].append(key)
    return {"ok": requirements.get("ok", False) and not any(missing.values()),
            "before_segment": before, "missing": missing,
            "evidence": {field: {key: evidence[field][key] for key in keys if key in evidence[field]}
                         for field, keys in (("tables", requirements.get("tables", [])),
                                             ("columns", requirements.get("columns", [])),
                                             ("joins", required_joins))}}
