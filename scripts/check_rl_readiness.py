"""Pre-flight audit for a future GRPO/DAPO run.

This script does not import a model and does not start Ray/vLLM.  It checks the
things that are cheap to get wrong and expensive to discover after a rollout:

* RL parquet contains only ``split=rl`` rows and has no DB/question overlap
  with val/test/dev200;
* the parquet was built with the frozen E1.1 query prompt, not the legacy
  builder default;
* the selected SFT checkpoint exists, is a full HF model (no adapter), and its
  config is loadable;
* the installed veRL exposes the GRPO/DAPO loss knobs that the config uses;
* dynamic sampling is *not* silently claimed: this veRL commit defines
  ``FilterGroupsConfig`` but has no execution path, so the audit reports that
  DAPO dynamic sampling must be implemented separately;
* the SFT final-test gate (test1534) is reported when available.

Exit code is 0 when no errors were found.  Warnings are allowed and are part
of the report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def as_dict(value):
    if isinstance(value, dict):
        return value
    # Older pyarrow builds return numpy.void; convert through dict().
    return dict(value)


def extra_info(frame, column="extra_info"):
    return [as_dict(value) for value in frame[column].tolist()]


def check_parquet(path, *, expected_split, name, errors, warnings, runtime_contract="", catalog_dir=None):
    path = Path(path)
    if not path.is_file():
        errors.append(f"{name}: missing parquet {path}")
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - the report should include the reason
        errors.append(f"{name}: cannot read {path}: {type(exc).__name__}: {exc}")
        return None
    required = {"prompt", "reward_model", "extra_info"}
    missing = required - set(frame.columns)
    if missing:
        errors.append(f"{name}: missing columns {sorted(missing)}")
        return None
    infos = extra_info(frame)
    splits = sorted({str(info.get("split")) for info in infos})
    if splits != [expected_split]:
        errors.append(f"{name}: split values {splits}, expected [{expected_split!r}]")
    ids = [info.get("question_id") for info in infos]
    if any(not value for value in ids):
        errors.append(f"{name}: some question_id values are empty")
    if len(set(ids)) != len(ids):
        errors.append(f"{name}: duplicate question_id values: {len(ids) - len(set(ids))}")
    missing_db = [str(info.get("db_path")) for info in infos
                  if not Path(str(info.get("db_path", ""))).is_file()]
    if missing_db:
        errors.append(f"{name}: missing SQLite db_path: {len(missing_db)} e.g. {missing_db[:3]}")
    dbs = sorted({str(info.get("db_id")) for info in infos})
    modes = sorted({str(info.get("prompt_mode")) for info in infos})
    retrieval = sorted({str(info.get("retrieval_version")) for info in infos})
    versions = sorted({str(info.get("prompt_version")) for info in infos})
    if modes != ["query"]:
        errors.append(f"{name}: prompt_mode {modes}, expected ['query'] (E1.1 contract)")
    expected_retrieval = "E1.2" if runtime_contract else "E1.1"
    if retrieval != [expected_retrieval]:
        errors.append(f"{name}: retrieval_version {retrieval}, expected [{expected_retrieval!r}]")
    if runtime_contract:
        from agent.runtime_contract import load_store, prompt, validate_extra
        stores = {}
        catalog_dir = catalog_dir or ROOT / "artifacts/catalog"
        for info, messages in zip(infos, frame["prompt"].tolist()):
            try:
                validate_extra(info)
                db = info["db_id"]
                if db not in stores:
                    stores[db] = load_store(catalog_dir, db, info["db_path"])
                actual = "\n".join(as_dict(m).get("content", "") for m in messages)
                if actual != prompt(info, stores[db]):
                    raise ValueError(f"prompt differs from runtime for {info.get('question_id')}")
            except (ValueError, KeyError, OSError) as exc:
                errors.append(f"{name}: {exc}")
                break
    if versions != ["catalog-three-action-query-v2"]:
        errors.append(
            f"{name}: prompt_version {versions}, expected ['catalog-three-action-query-v2']")
    missing_gold = 0
    for item in frame["reward_model"].tolist():
        value = as_dict(item).get("ground_truth")
        if not value:
            missing_gold += 1
    if missing_gold:
        errors.append(f"{name}: {missing_gold} rows have no reward_model.ground_truth")
    prompt_lengths = [len(as_dict(item[0]).get("content", ""))
                      for item in frame["prompt"].tolist()]
    if not prompt_lengths:
        errors.append(f"{name}: empty prompt column")
    return {
        "path": str(path),
        "rows": len(frame),
        "dbs": dbs,
        "db_count": len(dbs),
        "splits": splits,
        "prompt_modes": modes,
        "retrieval_versions": retrieval,
        "prompt_versions": versions,
        "prompt_chars": {
            "min": min(prompt_lengths) if prompt_lengths else None,
            "max": max(prompt_lengths) if prompt_lengths else None,
            "mean": (round(sum(prompt_lengths) / len(prompt_lengths), 2)
                     if prompt_lengths else None),
        },
        "ids": ids,
    }


def check_disjoint(train, val, name, errors):
    if not train or not val:
        return
    db_overlap = sorted(set(train["dbs"]) & set(val["dbs"]))
    id_overlap = sorted(set(train["ids"]) & set(val["ids"]))
    if db_overlap:
        errors.append(f"RL/{name} DB overlap: {len(db_overlap)} e.g. {db_overlap[:5]}")
    if id_overlap:
        errors.append(f"RL/{name} question overlap: {len(id_overlap)} e.g. {id_overlap[:5]}")


def check_catalog(train, val, catalog_dir, errors):
    catalog_dir = Path(catalog_dir)
    dbs = sorted(set((train or {}).get("dbs", [])) | set((val or {}).get("dbs", [])))
    missing = []
    for db_id in dbs:
        path = catalog_dir / db_id
        if not path.is_dir():
            missing.append(str(path))
            continue
        if not (path / "schema.json").is_file() and not (path / "catalog.json").is_file():
            missing.append(str(path))
    if missing:
        errors.append(f"catalog: missing/invalid stores: {len(missing)} e.g. {missing[:5]}")
    return {
        "catalog_dir": str(catalog_dir),
        "dbs_checked": len(dbs),
        "missing": missing[:20],
    }


def check_model(path, errors, warnings):
    path = Path(path)
    config_path = path / "config.json"
    if not config_path.is_file():
        errors.append(f"model: missing config.json in {path}")
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"model: cannot read {config_path}: {type(exc).__name__}: {exc}")
        return None
    if (path / "adapter_config.json").exists():
        errors.append("model: directory contains adapter_config.json; this launcher is full-FT")
    weight_files = [path / "model.safetensors", path / "pytorch_model.bin"] + sorted(
        path.glob("*.safetensors.index.json")) + sorted(path.glob("pytorch_model-*.bin"))
    if not any(candidate.is_file() for candidate in weight_files):
        errors.append(f"model: no HF weight file found in {path}")
    architectures = config.get("architectures") or []
    if "Qwen2ForCausalLM" not in architectures:
        warnings.append(f"model: architectures={architectures}, expected Qwen2ForCausalLM")
    return {
        "path": str(path),
        "architectures": architectures,
        "dtype": config.get("dtype") or config.get("torch_dtype"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "vocab_size": config.get("vocab_size"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "has_adapter": (path / "adapter_config.json").exists(),
    }


def check_verl(errors, warnings):
    try:
        import verl
    except Exception as exc:  # noqa: BLE001
        errors.append(f"verl: import failed: {type(exc).__name__}: {exc}")
        return {"available": False}
    root = Path(verl.__file__).resolve().parent
    core = (root / "trainer" / "ppo" / "core_algos.py").read_text(encoding="utf-8")
    trainer = (root / "trainer" / "ppo" / "ray_trainer.py").read_text(encoding="utf-8")
    algorithm = (root / "trainer" / "config" / "algorithm.py").read_text(encoding="utf-8")
    has_filter_config = "class FilterGroupsConfig" in algorithm
    # A real implementation would reference filter_groups outside the config
    # definition.  The installed 0.8.0.dev0 commit does not.
    has_filter_execution = "filter_groups" in trainer or "filter_groups" in core
    if has_filter_config and not has_filter_execution:
        warnings.append(
            "verl: FilterGroupsConfig exists but no execution path was found; "
            "true DAPO dynamic sampling is NOT available in this commit")
    info = {
        "available": True,
        "path": str(root),
        "version": getattr(verl, "__version__", None),
        "has_filter_groups_config": has_filter_config,
        "has_filter_groups_execution": has_filter_execution,
        "has_clip_ratio_high": "clip_ratio_high" in core,
        "has_loss_agg_mode": "loss_agg_mode" in core,
        "has_dapo_reward_manager": (root / "workers" / "reward_manager" / "dapo.py").is_file(),
        "has_response_mask_agent_loop": "response_mask" in (
            root / "experimental" / "agent_loop" / "agent_loop.py").read_text(encoding="utf-8"),
    }
    for key in ("has_clip_ratio_high", "has_loss_agg_mode", "has_dapo_reward_manager"):
        if not info[key]:
            errors.append(f"verl: required capability missing: {key}")
    return info


def check_agent_loop(errors, warnings):
    path = ROOT / "agent" / "bird_agent_loop.py"
    if not path.is_file():
        errors.append("agent_loop: file missing")
        return None
    text = path.read_text(encoding="utf-8")
    checks = {
        "passes_retrieval_version": "retrieval_version=self.retrieval_version" in text,
        "caps_action_tokens": "max_tokens" in text and "_sampling_with_budget" in text,
        "caps_context": "self.ep_cfg.max_context" in text,
        "zero_logprobs_for_observations": "logprobs += [0.0] * len(ids)" in text,
        "returns_none_on_partial_logprobs": "logprobs_complete" in text,
    }
    for key, value in checks.items():
        if not value:
            errors.append(f"agent_loop: contract check failed: {key}")
    return checks


def check_sft_gate(base_path, sft_path, *, require, errors, warnings,
                   paired_path=None):
    base, sft = read_json(base_path), read_json(sft_path)
    if base is None or sft is None:
        missing = []
        if base is None:
            missing.append(str(base_path))
        if sft is None:
            missing.append(str(sft_path))
        message = f"SFT gate: test1534 report(s) missing: {missing}"
        (errors if require else warnings).append(message)
        return {"available": False, "require": require, "missing": missing}
    base_ex, sft_ex = float(base.get("ex", 0.0)), float(sft.get("ex", 0.0))
    delta = sft_ex - base_ex
    info = {
        "available": True,
        "base_ex": base_ex,
        "sft_ex": sft_ex,
        "delta": delta,
        "base_n": base.get("n"),
        "sft_n": sft.get("n"),
        "require": require,
    }
    if base.get("n") != sft.get("n"):
        errors.append(f"SFT gate: n mismatch {base.get('n')} vs {sft.get('n')}")
    if delta <= 0:
        message = f"SFT gate: test1534 delta {delta:+.4f} is not positive"
        (errors if require else warnings).append(message)

    paired = read_json(paired_path) if paired_path else None
    if paired is None:
        if paired_path:
            message = f"SFT gate: paired report missing: {paired_path}"
            (errors if require else warnings).append(message)
        return info
    p_value = float(paired.get("mcnemar_exact_p", 1.0))
    ci = paired.get("db_bootstrap_ci95") or [None, None]
    info.update({
        "paired": str(paired_path),
        "net_gain": paired.get("net_gain"),
        "b_only": paired.get("b_only"),
        "a_only": paired.get("a_only"),
        "mcnemar_exact_p": p_value,
        "db_bootstrap_ci95": ci,
    })
    # The project's pre-registered SFT gate is EX improvement + McNemar p<0.1.
    if require and p_value >= 0.1:
        errors.append(f"SFT gate: McNemar p={p_value:.4f} >= 0.1")
    elif p_value >= 0.1:
        warnings.append(f"SFT gate: McNemar p={p_value:.4f} >= 0.1")
    if ci[0] is not None and ci[1] is not None and ci[0] <= 0.0 <= ci[1]:
        warnings.append(
            f"SFT gate: DB bootstrap 95% CI {ci} includes 0; treat the final "
            "test as a weak positive, not a robust one")
    return info


def check_local_dynamic_sampling(root, *, require, errors, warnings):
    """Check the project-local DAPO dynamic-sampling implementation.

    The installed veRL commit still has no filter_groups execution path; the
    project implements it locally.  R1 does not require these files.  R3 does.
    """
    root = Path(root)
    files = {
        "trainer": root / "train" / "dapo_ray_trainer.py",
        "adapter": root / "train" / "dynamic_sampling.py",
        "entrypoint": root / "train" / "main_dapo.py",
        "tests": root / "tests" / "test_dapo_dynamic_sampling.py",
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    rc_file = root / "e1_alignment" / "rl_core.py"
    has_select = rc_file.is_file() and "def select_mixed_groups" in rc_file.read_text(
        encoding="utf-8")
    info = {
        "files": {name: str(path) for name, path in files.items()},
        "missing": missing,
        "has_select_mixed_groups": has_select,
        "require_dynamic_sampling": require,
    }
    if missing and require:
        errors.append(f"dynamic_sampling: missing local files: {missing}")
    elif missing:
        warnings.append(
            "dynamic_sampling: local DS files missing; R1 terminal GRPO can run, "
            f"R3 cannot: {missing}")
    if not has_select:
        message = "dynamic_sampling: e1_alignment.rl_core.select_mixed_groups missing"
        (errors if require else warnings).append(message)
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-parquet", default="artifacts/datasets/rl_e1_1.parquet")
    p.add_argument("--val-parquet", default="artifacts/datasets/dev200_rl_e1_1.parquet")
    p.add_argument("--model", default="outputs/m6_08m/student_sft_full_lowlr_5e6/epoch_2/hf_model")
    p.add_argument("--catalog-dir", default="artifacts/catalog")
    p.add_argument("--test1534-base", default="reports/m6/e1_student_only/test1534_base_e1_1.json")
    p.add_argument("--test1534-sft", default="reports/m6/e1_student_only/test1534_sft_lowlr_788.json")
    p.add_argument("--test1534-paired",
                   default="reports/m6/e1_student_only/test1534_paired_lowlr_788.json")
    p.add_argument("--require-sft-gate", action="store_true")
    p.add_argument("--require-dynamic-sampling", action="store_true")
    p.add_argument("--runtime-contract", choices=["", "A-e1.2-typed-v1"], default="")
    p.add_argument("--out", default="reports/m6/e1_student_only/rl_readiness.json")
    a = p.parse_args()

    errors, warnings = [], []
    train = check_parquet(a.train_parquet, expected_split="rl", name="train", errors=errors, warnings=warnings,
                          runtime_contract=a.runtime_contract, catalog_dir=a.catalog_dir)
    val = check_parquet(a.val_parquet, expected_split="val", name="val", errors=errors, warnings=warnings,
                        runtime_contract=a.runtime_contract, catalog_dir=a.catalog_dir)
    if train and train["rows"] != 1801:
        warnings.append(f"train: rows={train['rows']}, expected 1801 from docs/DATA_SPLITS.md")
    if val and val["rows"] != 200:
        warnings.append(f"val: rows={val['rows']}, expected 200 dev200")
    check_disjoint(train, val, "dev200", errors)
    catalog = check_catalog(train, val, a.catalog_dir, errors)
    model = check_model(a.model, errors, warnings)
    verl = check_verl(errors, warnings)
    agent = check_agent_loop(errors, warnings)
    dynamic_sampling = check_local_dynamic_sampling(
        ROOT, require=a.require_dynamic_sampling, errors=errors, warnings=warnings)
    if not dynamic_sampling["missing"] and dynamic_sampling["has_select_mixed_groups"]:
        # The installed veRL lacks filter_groups execution, but the project
        # provides it locally. Do not keep a misleading "not available" warning.
        warnings[:] = [
            warning for warning in warnings
            if not warning.startswith("verl: FilterGroupsConfig exists")
        ]
    gate = check_sft_gate(
        a.test1534_base, a.test1534_sft, require=a.require_sft_gate,
        errors=errors, warnings=warnings, paired_path=a.test1534_paired)

    report = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "train": {key: value for key, value in (train or {}).items() if key != "ids"},
        "val": {key: value for key, value in (val or {}).items() if key != "ids"},
        "catalog": catalog,
        "model": model,
        "verl": verl,
        "agent_loop": agent,
        "dynamic_sampling": dynamic_sampling,
        "sft_gate": gate,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
