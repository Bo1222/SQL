"""CPU review of the 114 evidence_conflict questions.

This script does not modify labels.  It records a per-question review
disposition next to the original audit row so the training/eval gate can be
decided explicitly:

``no_semantic_conflict``
    The heuristic flag is an alias/encoding/definition artefact; the question,
    Evidence and Gold do not actually disagree.
``evidence_defect``
    Evidence contradicts the question and Gold follows the question.
``gold_defect_suspected``
    Gold omits or contradicts a condition the question/Evidence requires.
``needs_manual_review``
    The conflict cannot be resolved from question/Evidence/Gold alone.

The dispositions are review proposals, not label edits.  Gold/Evidence are
left byte-identical to the audit input.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Reviewed 2026-11-26, in the order of labels_evidence_conflict.jsonl.
REVIEW = {
    0: ("gold_defect_suspected", "Evidence requires month<6; Gold has no month predicate."),
    1: ("gold_defect_suspected", "Evidence requires result='Nominee'; Gold omits result and uses birth_region for birth_place."),
    2: ("needs_manual_review", "Evidence adds professor=1; Gold relies on hasPosition only; possible redundancy."),
    3: ("no_semantic_conflict", "Method_100k vs Method is a table-name alias; Id=2 predicate is present."),
    4: ("no_semantic_conflict", "state.name='New York' is encoded as state.abbreviation='NY' in Gold."),
    5: ("no_semantic_conflict", "Evidence typo STAR for START; Gold orders by START."),
    6: ("needs_manual_review", "Evidence adds lgID='NBA'; Gold filters year only; league may be implicit."),
    7: ("no_semantic_conflict", "character_name vs `Character Name` is a naming alias."),
    8: ("no_semantic_conflict", "'system' is stored as Italian 'sistema'; Gold filters the equivalent word id."),
    9: ("no_semantic_conflict", "Arizona encoded as state='AZ'."),
    10: ("evidence_defect", "Evidence says paid; question asks free; Gold uses Type='Free' matching the question."),
    11: ("no_semantic_conflict", "Evidence lists both d_quality=5 and 0; question asks worst, Gold uses 0."),
    12: ("evidence_defect", "Evidence uses FirstBrewedYear; the question's 2012 refers to review date, which Gold uses."),
    13: ("no_semantic_conflict", "Description='No Discount' encoded as UnitPriceDiscount=0."),
    14: ("no_semantic_conflict", "divID='WE' encoded as conference='West'."),
    15: ("evidence_defect", "Evidence says city New York; question says state; Gold uses State='New York'."),
    16: ("gold_defect_suspected", "Question says won; Evidence says result='Winner'; Gold filters result='Nominee'."),
    17: ("no_semantic_conflict", "city 'mountainview' vs 'mountain view' is spacing."),
    18: ("gold_defect_suspected", "Evidence requires type='sen'; Gold has no chamber/type predicate."),
    19: ("evidence_defect", "Question says Net 30; Evidence says Net 60; Gold uses Net 30."),
    20: ("no_semantic_conflict", "mpaa_rating vs `MPAA Rating` is a naming alias."),
    21: ("no_semantic_conflict", "medal_name='Gold' encoded as medal.id='1'."),
    22: ("gold_defect_suspected", "Question says thumbnail; Evidence uses ThumbnailPhotoFileName; Gold uses LargePhotoFileName."),
    23: ("gold_defect_suspected", "Question/Evidence require ExternalDebtReportingStatus='Estimate'; Gold omits it."),
    24: ("gold_defect_suspected", "Question/Evidence say star score >5; Gold filters votes >5."),
    25: ("needs_manual_review", "Gold uses string comparison closing_time > '8PM'; Evidence enumerates 9/10/11/12PM; 12PM ordering is suspect."),
    26: ("evidence_defect", "Question says 1952; Evidence says 1977; Gold uses 1952."),
    27: ("needs_manual_review", "Evidence references Business.review_count; Gold counts 5-star reviews; review_count may be a different aggregate."),
    28: ("no_semantic_conflict", "rating > 7.0 vs > 7 is numeric formatting."),
    29: ("no_semantic_conflict", "ser_time > '00:15:00' encoded as minutes > '15'."),
    30: ("gold_defect_suspected", "Evidence requires type='rep'; Gold has no chamber/type predicate."),
    31: ("no_semantic_conflict", "Currency string cast to numeric is equivalent."),
    32: ("no_semantic_conflict", "median_female_age vs female_median_age is a column-name alias."),
    33: ("no_semantic_conflict", "school_country vs school_county is a typo; the value matches."),
    34: ("gold_defect_suspected", "Question asks assistant location managers; Evidence role matches; Gold filters 'president of NBC West Coast'."),
    35: ("evidence_defect", "Question says range 10000-50000; Evidence says hirange<12000; Gold uses 50000."),
    36: ("gold_defect_suspected", "Evidence requires type='rep'; Gold has no chamber/type predicate."),
    37: ("no_semantic_conflict", "Furnitures vs Furniture is spelling."),
    38: ("evidence_defect", "Question says business 11825; Evidence says 12476; Gold uses 11825."),
    39: ("no_semantic_conflict", "rae vs race is a typo; Gold uses race='X'."),
    40: ("no_semantic_conflict", "reace vs race is a typo; Gold uses race='X'."),
    41: ("no_semantic_conflict", "ser_time < '00:10:00' encoded as minutes < '10'."),
    42: ("needs_manual_review", "Evidence adds lgID='NBA'; Gold relies on award='Finals MVP'."),
    43: ("no_semantic_conflict", "Evidence lists both year=4 and year=1; question asks oldest, Gold uses 1."),
    44: ("no_semantic_conflict", "Derived Profit column is computed inline by Gold."),
    45: ("no_semantic_conflict", "Derived age is computed inline by Gold."),
    46: ("gold_defect_suspected", "Question asks cities in Philippines; Evidence says CountryCode='PHL'; Gold filters City.Name='PHL'."),
    47: ("no_semantic_conflict", "Derived Profit column is computed inline by Gold."),
    48: ("no_semantic_conflict", "OCCURENCES vs OCCURRENCES is a typo; Gold uses the real column."),
    49: ("no_semantic_conflict", "waiter_services vs waiter_service is a spelling variant."),
    50: ("needs_manual_review", "Evidence adds tip_length='short'; question asks about one tip; Gold omits tip_length."),
    51: ("no_semantic_conflict", "rumoured vs Rumored is spelling."),
    52: ("no_semantic_conflict", "state.Region vs district.division is a modeling alias; the value matches."),
    53: ("gold_defect_suspected", "Question separates monthly basis and hourly rate>50; Gold multiplies rate by PayFrequency instead of filtering both."),
    54: ("no_semantic_conflict", "AddressTypeID=5 encoded as name='Shipping'."),
    55: ("no_semantic_conflict", "CommentsXML vs CommentIsXml is a naming alias."),
    56: ("no_semantic_conflict", "LIKE '%Pass%' covers Pass and Pass w/Conditions."),
    57: ("no_semantic_conflict", "CANCELLED=1 is implied by CANCELLATION_CODE='A'."),
    58: ("gold_defect_suspected", "Question asks SuperSport Park/Centurion; Gold filters St George."),
    59: ("no_semantic_conflict", "teachability vs teachingability is a column-name alias."),
    60: ("evidence_defect", "Question says 2008; Evidence says 2018; Gold uses 2008."),
    61: ("no_semantic_conflict", "Positive sentiment encoded as polarity > 0."),
    62: ("needs_manual_review", "Question says slowest and least expensive; Gold orders by cost only; Standard method not enforced."),
    63: ("no_semantic_conflict", "Evidence lists both payment_method values; Gold returns the value."),
    64: ("evidence_defect", "Question says 1960; Evidence says 1970; Gold uses 1960."),
    65: ("gold_defect_suspected", "Question/Evidence ask weight; Gold filters description='Body Height'."),
    66: ("no_semantic_conflict", "Heuristic parsed SQL fragments from Evidence; Gold uses gender='F' and the event name."),
    67: ("no_semantic_conflict", "CountryName is reached through the Country join; Gold uses ShortName."),
    68: ("no_semantic_conflict", "Question asks whether credited; Gold returns the credited value without a filter."),
    69: ("no_semantic_conflict", "result!='Winner' is equivalent to Nominee for this field."),
    70: ("needs_manual_review", "Question says 1997-2000 season; Evidence says the same; Gold uses 1999-2000."),
    71: ("evidence_defect", "Question asks city Indianapolis and 2016-10-07; Evidence says state Indianapolis and 2013-04-04; Gold matches the question."),
    72: ("no_semantic_conflict", "Evidence lists two code definitions; Gold returns PreferredVendorStatus."),
    73: ("evidence_defect", "Question says 6 June 2018; Evidence says 6/5/18; Gold uses 6/6/18."),
    74: ("no_semantic_conflict", "year vs season_id is a column-name alias."),
    75: ("evidence_defect", "Evidence's 'Bothell' is a copy/paste error; Gold uses Arabic."),
    76: ("no_semantic_conflict", "CANCELLATION_CODE='C' implies cancelled and non-null."),
    77: ("no_semantic_conflict", "Rating=5 is equivalent to the subquery max in this DB."),
    78: ("no_semantic_conflict", "florida encoded as FL."),
    79: ("no_semantic_conflict", "11/30/2008 vs 2008-11-30 is date formatting."),
    80: ("no_semantic_conflict", "Evidence lists position-code definitions; Gold returns pos."),
    81: ("no_semantic_conflict", "Evidence lists both a_quality extremes; question asks worst, Gold uses 0."),
    82: ("evidence_defect", "Evidence says 'Low middle income'; Gold uses the actual label 'Lower middle income'."),
    83: ("no_semantic_conflict", "Evidence's Style='M' is irrelevant to the question; Gold uses Class='H'."),
    84: ("gold_defect_suspected", "Question asks oldest production technician; Evidence requires JobTitle; Gold filters only HireDate."),
    85: ("no_semantic_conflict", "character_name vs `Character Name` is a naming alias."),
    86: ("no_semantic_conflict", "Heuristic parsed fragments from Evidence; Gold applies race, cohort, and year."),
    87: ("needs_manual_review", "Evidence person_id nm007064 vs Gold nm0007064 (typo); credited='' vs Gold 'false' needs schema-semantic check."),
    88: ("no_semantic_conflict", "Evidence lists rating-code definitions; Gold returns the rating."),
    89: ("evidence_defect", "Question says 2018-09-11; Evidence says 2017-07-22; Gold matches the question."),
    90: ("gold_defect_suspected", "Question/Evidence say Arlington; Gold filters Kemptville."),
    91: ("needs_manual_review", "Evidence adds lgID='NBA'; Gold omits the league filter."),
    92: ("gold_defect_suspected", "Question asks payment due; Evidence says bool='pos'; Gold filters bool='neg'."),
    93: ("gold_defect_suspected", "Question/Evidence require result='Nominee'; Gold omits result."),
    94: ("needs_manual_review", "Evidence adds cohort '4y other','4y bach'; Gold omits cohort."),
    95: ("evidence_defect", "Question asks Sales person; Evidence says PersonType='SC'; Gold uses 'SP'."),
    96: ("needs_manual_review", "Evidence's indicator name differs from Gold's but may be the same series; needs catalog check."),
    97: ("gold_defect_suspected", "Question requires United States; Gold omits the country filter."),
    98: ("needs_manual_review", "Evidence references players_teams.ftAttempted; Gold uses player_allstar; table choice needs check."),
    99: ("evidence_defect", "Evidence says paid; question asks free; Gold uses Type='Free'. Gold also has a SUBSTR bug on a missing column."),
    100: ("gold_defect_suspected", "Question/Evidence say World Computer Congress; Gold filters International Congress Series."),
    101: ("gold_defect_suspected", "Question asks highest pay frequency; Evidence says PayFrequency=2; Gold orders by Rate."),
    102: ("needs_manual_review", "Evidence adds professor=1; Gold uses hasPosition<>0; possible redundancy."),
    103: ("no_semantic_conflict", "Evidence lists transaction-type definitions; Gold returns the type."),
    104: ("no_semantic_conflict", ">2011 encoded as >=2012-01-01."),
    105: ("evidence_defect", "Evidence adds category Food; question/Gold only require BYOB, high review count, active."),
    106: ("gold_defect_suspected", "Question/Evidence require FinishedGoodsFlag=1 (salable); Gold omits it."),
    107: ("needs_manual_review", "Question says not credited; Evidence says credited=''; Gold uses 'false'; schema semantics need check."),
    108: ("evidence_defect", "Question says after October 2012; Evidence says May 2012; Gold uses October."),
    109: ("no_semantic_conflict", "lgID='NBA' is implied by award='NBA Coach of the Year'."),
    110: ("needs_manual_review", "Question asks status; Gold returns counts from no_payment_due without filtering bool; semantics need check."),
    111: ("gold_defect_suspected", "Question/Evidence require work orders (TransactionType='W'); Gold omits it."),
    112: ("no_semantic_conflict", ">1960-01-01 encoded as >=1961-01-01 for date rows."),
    113: ("evidence_defect", "Question says medium class women's; Evidence says high class men's; Gold follows the question."),
}

POLICY = {
    "no_semantic_conflict": "re-admit candidate; no label edit needed (user approval still required)",
    "evidence_defect": "keep eval-only; register Evidence repair proposal",
    "gold_defect_suspected": "keep eval-only; register Gold repair proposal",
    "needs_manual_review": "keep eval-only until manual review",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conflicts", default=(
        "reports/m6/sft_first_20261120/step0_step1/labels_evidence_conflict.jsonl"))
    parser.add_argument("--out", default=(
        "reports/m6/sft_first_20261120/step0_step1/conflict_review"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.conflicts).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len(REVIEW):
        raise SystemExit(f"review covers {len(REVIEW)} rows but input has {len(rows)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reviewed = []
    counts = Counter()
    for index, row in enumerate(rows):
        disposition, note = REVIEW[index]
        counts[disposition] += 1
        reviewed.append({
            "index": index,
            "question_id": row["question_id"],
            "db_id": row["db_id"],
            "category": row["category"],
            "question": row["question"],
            "evidence": row["evidence"],
            "gold_sql": row["gold_sql"],
            "flags": row["label"]["evidence_conflicts"],
            "disposition": disposition,
            "review_note": note,
            "recommended_policy": POLICY[disposition],
            "original_label_modified": False,
        })

    with (out / "conflict_review.jsonl").open("w", encoding="utf-8") as handle:
        for row in reviewed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "total": len(reviewed),
        "counts": dict(counts),
        "policy": POLICY,
        "original_label_modified": False,
        "reviewed_by": "CPU manual review of question/evidence/Gold text",
        "reviewed_at": "2026-11-26",
        "caveat": ("Dispositions are review proposals. Evidence/Gold were not changed; "
                   "re-admission or repair still requires user approval under D2."),
    }
    (out / "conflict_review_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Evidence-conflict review (114 questions)",
        "",
        "This is a CPU text review of the questions already flagged as "
        "`evidence_conflict`. No label, Evidence, or Gold SQL was modified.",
        "",
        "| disposition | questions | recommended policy |",
        "|---|---:|---|",
    ]
    for disposition in ("no_semantic_conflict", "evidence_defect",
                        "gold_defect_suspected", "needs_manual_review"):
        lines.append(f"| {disposition} | {counts.get(disposition, 0)} | "
                     f"{POLICY[disposition]} |")
    lines += ["", "## Reading", "",
              "- `no_semantic_conflict` rows are heuristic artefacts (column aliases, "
              "value encodings, definitions that list both sides, date/number formatting).",
              "- `evidence_defect` means the Evidence contradicts the question and Gold "
              "follows the question; these are not Gold defects.",
              "- `gold_defect_suspected` means Gold appears to omit or contradict a "
              "condition the question/Evidence requires.",
              "- `needs_manual_review` cannot be decided from text and DB metadata alone.",
              "", "## Per-question review", "",
              "| # | db | disposition | note | question |",
              "|---:|---|---|---|---|"]
    for row in reviewed:
        question = row["question"].replace("|", "\\|")
        lines.append(f"| {row['index']} | {row['db_id']} | {row['disposition']} | "
                     f"{row['review_note']} | {question} |")
    lines += ["", "Full Evidence/Gold text and flags are in `conflict_review.jsonl`.", ""]
    (out / "CONFLICT_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
