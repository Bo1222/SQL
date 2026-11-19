"""Verify frozen runtime files/DBs/tokenizer without freezing pipeline fixes."""
import json
from pathlib import Path
from e1_alignment.common import sha256_file
from e1_alignment.runtime import RuntimeContract

FINGERPRINT = Path("reports/m6/e1_alignment/runtime_fingerprint.json")


def verify_contract(check_databases=True):
    frozen = json.loads(FINGERPRINT.read_text())
    if RuntimeContract().as_dict() != frozen["contract"]:
        raise ValueError("runtime_config_changed")
    entries = frozen["code"]["runtime_files"] + frozen["model"]["files"]
    if check_databases:
        entries += frozen["database"]["databases"]
    for entry in entries:
        if entry["exists"] and sha256_file(entry["path"]) != entry["sha256"]:
            raise ValueError("frozen_contract_file_changed:" + entry["path"])
    if sha256_file("artifacts/manifests/catalog_manifest.json") != frozen["database"]["catalog_manifest_sha256"]:
        raise ValueError("catalog_manifest_changed")
    return frozen["fingerprint_sha256"]
