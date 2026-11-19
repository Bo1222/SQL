"""Smallest feasible reviewed dataset; auxiliary binary variables count coverage."""
from collections import Counter, defaultdict
import random
from e1_alignment.common import canonical_hash
from e1_alignment.revision.quality import review_errors


def features(row):
    actions = [s for s in row["segments"] if s.get("kind") == "action"]
    repair = row.get("repair_evidence") or {}
    boundary = repair.get("failure_action_count", len(actions))
    return {"gold": int(row["construction_source"] == "gold_assisted"),
            "tokens": sum(row["loss_mask"][1:]),
            "repair": int(row.get("has_verified_failure_repair") is True),
            "repair_tokens": sum(sum(s["loss_mask"]) for s in actions[boundary:]) if repair.get("verified") else 0,
            "initial": actions[0]["action"] if row["start_mode"] == "question_start" and actions and any(actions[0]["loss_mask"]) else None,
            "direct": int(not any(s["action"] != "answer" for s in actions)),
            "error": repair.get("error_class", "none")}


def learning_value(row):
    f = features(row)
    # Source labels and paraphrased SQL are deliberately excluded.
    # A full episode with an unsupervised failed first action and a continuation
    # of that same failure teach the same suffix. Merely changing start_mode
    # cannot create a second value when no initial decision is supervised.
    mode = "initial_decision" if f["initial"] else ("failure_repair" if f["repair"] else "history_completion")
    return canonical_hash([mode, f["initial"], f["error"] if f["repair"] else "none"])


def distribution(rows):
    fs = [features(r) for r in rows]
    return {"n": len(rows), "unique_questions": len({r["id"] for r in rows}),
            "databases": len({r["db_id"] for r in rows}),
            "by_db": dict(Counter(r["db_id"] for r in rows)),
            "gold_records": sum(f["gold"] for f in fs),
            "supervised_tokens": sum(f["tokens"] for f in fs),
            "gold_supervised_tokens": sum(f["gold"] * f["tokens"] for f in fs),
            "repair_records": sum(f["repair"] for f in fs),
            "repair_tokens": sum(f["repair_tokens"] for f in fs),
            "repair_questions": len({r["id"] for r, f in zip(rows, fs) if f["repair"]}),
            "repair_questions_by_error": {e: len({r["id"] for r, f in zip(rows, fs) if f["repair"] and f["error"] == e})
                                           for e in ("column_alias", "table_join", "syntax", "semantic", "other")},
            "initial_records": sum(f["initial"] is not None for f in fs),
            "initial_actions": dict(Counter(f["initial"] for f in fs if f["initial"])),
            "direct_answers": sum(f["direct"] for f in fs)}


def quota_errors(rows, p):
    d = distribution(rows)
    n, t, init = d["n"], d["supervised_tokens"], d["initial_records"]
    errors = []
    def require(ok, name):
        if not ok:
            errors.append(name)
    require(p["min_records"] <= n <= p["max_records"], "record_count")
    require(d["gold_records"] <= p["gold_record_max"] * n, "gold_record_ratio")
    require(t > 0 and d["gold_supervised_tokens"] <= p["gold_token_max"] * t, "gold_token_ratio")
    require(d["unique_questions"] >= p["unique_question_min"] * n, "unique_question_ratio")
    require(d["databases"] >= p["min_databases"], "database_coverage")
    require(d["repair_records"] >= p["repair_record_min"] * n, "repair_record_ratio")
    require(t > 0 and d["repair_tokens"] >= p["repair_token_min"] * t, "repair_token_ratio")
    require(d["repair_questions"] >= p["min_repair_questions"], "repair_question_coverage")
    for e in p["repair_errors"]:
        require(d["repair_questions_by_error"].get(e, 0) >= p["min_questions_per_error"], "repair_error_coverage:" + e)
    require(init >= p["initial_record_min"] * n and init > 0, "initial_record_ratio")
    for action, threshold in (("schema_lookup", p["initial_lookup_min"]), ("sql_execute", p["initial_execute_min"])):
        require(init > 0 and d["initial_actions"].get(action, 0) >= threshold * init, "initial_ratio:" + action)
    require(d["initial_actions"].get("answer", 0) <= p["initial_answer_max"] * init, "initial_answer_ratio")
    require(d["direct_answers"] <= p["direct_answer_max"] * n, "direct_answer_ratio")
    require(max(Counter(r["id"] for r in rows).values(), default=0) <= p["question_cap"], "question_cap")
    require(len({(r["id"], learning_value(r)) for r in rows}) == n, "duplicate_learning_value")
    return errors


def select(rows, reviews, policy, quarantined=()):
    eligible = [r for r in rows if r["id"] not in quarantined and
                not review_errors(r, reviews.get(r["trajectory_id"]), policy)]
    return feasible(eligible, policy)


def feasible(eligible, p, preferred_ids=()):
    """Program-only calls provide capacity evidence, never semantic approval."""
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import lil_matrix
    buckets = defaultdict(list)
    for r in eligible:
        buckets[(r["db_id"], features(r)["error"])].append(r)
    rng = random.Random(p["seed"])
    for key in sorted(buckets):
        buckets[key].sort(key=lambda r: r["trajectory_id"])
        rng.shuffle(buckets[key])
    rows = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                rows.append(buckets[key].pop())
    n = len(rows)
    report = {"eligible": n, "pool_distribution": distribution(rows), "seed": p["seed"]}
    if n < p["min_records"]:
        return [], {**report, "status": "infeasible", "reasons": quota_errors(rows, p)}
    fs = [features(r) for r in rows]
    constraints = []
    size = n
    def add(weights, lo=-np.inf, hi=np.inf):
        constraints.append((dict(weights), lo, hi))
    def vec(vals):
        return dict(enumerate(vals))
    add(vec([1] * n), p["min_records"], p["max_records"])
    add(vec([f["gold"] - p["gold_record_max"] for f in fs]), hi=0)
    add(vec([(f["gold"] - p["gold_token_max"]) * f["tokens"] for f in fs]), hi=0)
    add(vec([f["repair"] - p["repair_record_min"] for f in fs]), lo=0)
    add(vec([f["repair_tokens"] - p["repair_token_min"] * f["tokens"] for f in fs]), lo=0)
    add(vec([int(f["initial"] is not None) - p["initial_record_min"] for f in fs]), lo=0)
    for action, threshold, minimum in (("schema_lookup", p["initial_lookup_min"], True),
                                       ("sql_execute", p["initial_execute_min"], True),
                                       ("answer", p["initial_answer_max"], False)):
        weights = vec([int(f["initial"] == action) - threshold * int(f["initial"] is not None) for f in fs])
        add(weights, lo=0) if minimum else add(weights, hi=0)
    add(vec([f["direct"] - p["direct_answer_max"] for f in fs]), hi=0)
    def presence(groups):
        nonlocal size
        indices = []
        for key in sorted(groups):
            group = groups[key]
            y = size; size += 1; indices.append(y)
            add({**{i: -1 for i in group}, y: 1}, hi=0)
            for i in group:
                add({i: 1, y: -1}, hi=0)
        return indices
    questions, databases, repairs, errors, values = (defaultdict(list) for _ in range(5))
    for i, (r, f) in enumerate(zip(rows, fs)):
        questions[r["id"]].append(i); databases[r["db_id"]].append(i)
        values[(r["id"], learning_value(r))].append(i)
        if f["repair"]:
            repairs[r["id"]].append(i); errors[(f["error"], r["id"])].append(i)
    for group in questions.values():
        add({i: 1 for i in group}, hi=p["question_cap"])
    for group in values.values():
        add({i: 1 for i in group}, hi=1)
    q = presence(questions)
    add({**{i: -p["unique_question_min"] for i in range(n)}, **{i: 1 for i in q}}, lo=0)
    add({i: 1 for i in presence(databases)}, lo=p["min_databases"])
    add({i: 1 for i in presence(repairs)}, lo=p["min_repair_questions"])
    for error in p["repair_errors"]:
        groups = {q: indices for (e, q), indices in errors.items() if e == error}
        add({i: 1 for i in presence(groups)}, lo=p["min_questions_per_error"])
    matrix = lil_matrix((len(constraints), size), dtype=float)
    for j, (weights, _, _) in enumerate(constraints):
        for i, value in weights.items():
            matrix[j, i] = value
    objective = np.zeros(size)
    # Strict objective order: minimum size, most independent questions, then
    # seeded DB/error rotation. Review proposals may prefer already-read rows
    # within those first two objectives; final selection needs no such hint.
    objective[:n] = 1 + np.arange(n) / (32 * max(len(q), 1) * n * n)
    objective[q] = -.25 / max(len(q), 1)
    preferred = set(preferred_ids)
    if preferred:
        objective[:n] -= np.array([r["trajectory_id"] in preferred for r in rows]) / (16 * max(len(q), 1) * n)
    solved = milp(objective, integrality=np.ones(size), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix.tocsc(), [c[1] for c in constraints], [c[2] for c in constraints]),
                  options={"time_limit": 55, "mip_rel_gap": 0.0})
    if solved.status != 0 or solved.x is None:
        return [], {**report, "status": "infeasible" if solved.status == 2 else "solver_unresolved", "solver": solved.message}
    selected = [r for r, flag in zip(rows, solved.x[:n]) if flag > .5]
    failures = quota_errors(selected, p)
    return ([] if failures else selected), {**report, "status": "rejected" if failures else "selected", "reasons": failures,
                                           "distribution": distribution(selected), "solver": solved.message}
