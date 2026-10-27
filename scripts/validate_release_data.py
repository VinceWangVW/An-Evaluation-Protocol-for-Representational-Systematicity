#!/usr/bin/env python3
"""Check release integrity, exact pairs and split semantics; optionally load Croissant."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_release_data import build_payloads, CANONICAL, CSV_FIELDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--mlcroissant", action="store_true", help="Also validate metadata and load all local CSV records using mlcroissant.")
    args = parser.parse_args()
    root = args.root
    for name, expected in build_payloads().items():
        assert (root / name).read_bytes() == expected, f"Non-deterministic or modified release file: {name}"
    manifest = json.loads((root / "data/release/manifest.json").read_text())
    for name, facts in manifest["files"].items():
        data = (root / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == facts["sha256"], name
        assert len(data) == facts["bytes"], name
    rows = [json.loads(line) for line in (root / CANONICAL).read_text().splitlines()]
    assert len(rows) == 672
    assert Counter(row["split"] for row in rows) == {"train": 448, "dev": 128, "gen": 96}
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["metadata"]["counterpart_family_id"]][row["metadata"]["counterpart_type"]] = row
    assert len(grouped) == 96
    required = {"rescue_donor", "rescue_target", "role_swap", "agent_swap", "theme_swap", "filler_swap", "modifier_swap"}
    assert all(set(members) == required for members in grouped.values())
    for members in grouped.values():
        assert members["rescue_donor"]["target_form"] == members["rescue_target"]["target_form"]
        assert members["rescue_donor"]["metadata"]["voice"] == "active"
        assert members["rescue_donor"]["metadata"]["embedding_depth"] == 0
        train = any(row["split"] == "train" for row in members.values())
        assert not train or all(row["split"] == "train" for row in members.values()), "Family leakage from train"
        if members["rescue_target"]["generalization_case"] == "passive_depth1":
            assert not train, "Held-out construction in training"
    families = [json.loads(line) for line in (root / "data/release/matched_sets.jsonl").read_text().splitlines()]
    by_id = {row["item_id"]: row for row in rows}
    assert sum(len(f["m2_pairs"]) for f in families) == 192
    assert len(families) == 96
    for family in families:
        assert set(family["members"].values()) <= set(by_id)
        for pair in family["m2_pairs"]:
            before = by_id[pair["source_item_id"]]["role_signature"]
            after = by_id[pair["counterpart_item_id"]]["role_signature"]
            role = "agent" if pair["relation"] == "agent_swap" else "theme"
            other = "theme" if role == "agent" else "agent"
            assert before[role] != after[role] and before[other] == after[other]
    with (root / "data/release/items.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(CSV_FIELDS)
        csv_rows = list(reader)
    assert len(csv_rows) == 672
    assert {r["item_id"] for r in csv_rows} == set(by_id)
    croissant = json.loads((root / "croissant.json").read_text())
    assert croissant["license"] == "https://spdx.org/licenses/MIT.html"
    for field in ("dataLimitations", "dataBiases", "personalSensitiveInformation", "dataUseCases", "dataSocialImpact", "hasSyntheticData"):
        assert "rai:" + field in croissant, field
    assert "prov:wasDerivedFrom" in croissant and "prov:wasGeneratedBy" in croissant
    if args.mlcroissant:
        import mlcroissant as mlc
        # Use official content URLs as metadata, with a local mapping for offline load.
        dataset = mlc.Dataset(
            jsonld=croissant,
            mapping={resource["@id"]: root / resource["@id"] for resource in croissant["distribution"]},
        )
        records = list(dataset.records(record_set="items"))
        assert len(records) == 672, f"Croissant loaded {len(records)} records"
        print("mlcroissant metadata validation and complete 672-record local load passed.")
    print("Data validation passed: deterministic bytes/checksums, 96 complete sets, 672 items, splits, 192 interchange pairs, 96 repair pairs, Croissant/RAI fields.")


if __name__ == "__main__":
    main()
