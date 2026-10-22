"""SQL extraction, result comparison, and audit flags."""
import re
from dataclasses import dataclass, field

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
SQL_RE = re.compile(r"<sql>(.*?)</sql>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
FROM_RE = re.compile(r"\bfrom\b", re.IGNORECASE)
TAG_RE = re.compile(r"</?(?:answer|sql)>", re.IGNORECASE)


def _clean_sql(text: str) -> str:
    """Remove stray protocol tags that the model may append without an opening tag."""
    return TAG_RE.sub(" ", text or "").strip()


@dataclass
class Verdict:
    correct: bool
    flags: list = field(default_factory=list)
    reason: str = ""
    order_sensitive: bool = False


def extract_sql(text: str) -> tuple[str | None, str]:
    """Return (sql, method). Handles <answer>, <sql>, unclosed tags, fences."""
    text = text or ""
    for tag, pattern in (("answer", ANSWER_RE), ("sql", SQL_RE)):
        matches = pattern.findall(text)
        if matches:
            return _clean_sql(matches[-1]), f"{tag}_tag"
    for tag in ("answer", "sql"):
        match = re.search(rf"<{tag}>(.*)$", text, re.DOTALL | re.IGNORECASE)
        if match:
            candidate = re.split(r"</(?:answer|sql)>", match.group(1), flags=re.IGNORECASE)[0]
            candidate = _clean_sql(candidate)
            if candidate:
                return candidate, f"{tag}_unclosed"
    fences = FENCE_RE.findall(text)
    if fences:
        return _clean_sql(fences[-1]), "fenced"
    stripped = _clean_sql(text.strip())
    if stripped.lower().startswith(("select", "with")):
        return stripped, "raw"
    return None, "missing"


def normalize_cell(value):
    if value is None:
        return "<null>"
    if isinstance(value, float):
        if value != value:
            return "<nan>"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value).strip()


def normalize_rows(rows) -> list[tuple]:
    return [tuple(normalize_cell(c) for c in row) for row in rows]


def is_order_sensitive(sql: str) -> bool:
    return "order by" in (sql or "").lower()


def compare(gold_rows, cand_rows, gold_sql: str) -> tuple[bool, str]:
    gold = normalize_rows(gold_rows)
    cand = normalize_rows(cand_rows)
    if len(gold) != len(cand):
        return False, "row_count"
    if gold and cand and len(gold[0]) != len(cand[0]):
        return False, "column_count"
    if is_order_sensitive(gold_sql):
        return (gold == cand), "ordered"
    return (sorted(gold) == sorted(cand)), "unordered"


def audit_flags(cand_sql: str | None, gold_sql: str, gold_rows, cand_rows) -> tuple[list, list]:
    """Return (hard_flags, info_flags).

    hard: likely reward hacking, excluded from ex_clean.
    info: coincidence-prone but legitimate (e.g. scalar answers), reported only.
    """
    hard, info = [], []
    if cand_sql and not FROM_RE.search(cand_sql) and FROM_RE.search(gold_sql or ""):
        hard.append("constant_no_from")
    if not gold_rows and not cand_rows:
        info.append("empty_both")
    if len(gold_rows) == 1 and len(cand_rows) == 1:
        info.append("single_row_both")
    return hard, info


def error_type(message: str | None) -> str:
    if not message:
        return "none"
    text = message.lower()
    if "interrupt" in text:
        return "timeout"
    if "one statement at a time" in text:
        return "multi_statement"
    if "no such table" in text:
        return "no_such_table"
    if "no such column" in text:
        return "no_such_column"
    if "no such function" in text:
        return "no_such_function"
    if "ambiguous column" in text:
        return "ambiguous_column"
    if "syntax error" in text:
        return "syntax_error"
    if "misuse" in text:
        return "misuse"
    if "datatype mismatch" in text:
        return "datatype_mismatch"
    return "other"
