#!/usr/bin/env python3
"""Materialize the seed-13 paper testbed without network access or ML packages."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.systematicity_schema import build_counterpart_families, generate_schema_records
from data.surface_templates import render_template_candidates
from data.surface_validation import select_primary_candidates, validate_surface_candidate

BENCHMARK = "phase11_systematicity_v2"
CANONICAL = "data/processed/phase11_systematicity_v2.jsonl"
REPOSITORY = "https://github.com/VinceWangVW/An-Evaluation-Protocol-for-Representational-Systematicity"
RAW = "https://raw.githubusercontent.com/VinceWangVW/An-Evaluation-Protocol-for-Representational-Systematicity/main/"
CSV_FIELDS = (
    "item_id", "matched_set_id", "release_index_id", "split", "family_split",
    "counterpart_type", "construction_family", "voice", "embedding_depth",
    "input_text", "target_form", "predicate", "agent", "theme",
    "agent_modifier", "theme_modifier",
)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode()


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows).encode()


def build_payloads(seed: int = 13) -> dict[str, bytes]:
    schema = generate_schema_records(
        train_families=64, dev_families=16, gen_families=16,
        seed=seed, benchmark_name=BENCHMARK,
    )
    validated = [
        validate_surface_candidate(record, candidate)
        for record in schema for candidate in render_template_candidates(record, num_candidates=3)
    ]
    selected, examples, validation = select_primary_candidates(
        schema_records=schema, validated_candidates=validated, selection_strategy="shortest",
    )
    canonical = [example.to_dict() for example in examples]
    families = build_counterpart_families(schema)
    order = list(dict.fromkeys(record.counterpart_family_id for record in schema))
    # These are release indexing aliases only, not recovered historical walkthrough IDs.
    aliases = {family_id: f"GEN-{index:03d}" for index, family_id in enumerate(order)}
    for family in families:
        family["release_index_id"] = aliases[family["counterpart_family_id"]]
        family["release_index_provenance"] = "new zero-based generator-order alias; historical MS mapping unavailable"
    by_schema = {record.item_id: record for record in schema}
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in canonical:
        record = by_schema[row["item_id"]]
        writer.writerow({
            "item_id": record.item_id,
            "matched_set_id": record.counterpart_family_id,
            "release_index_id": aliases[record.counterpart_family_id],
            "split": record.split,
            "family_split": record.metadata["family_split"],
            "counterpart_type": record.counterpart_type,
            "construction_family": record.construction_family,
            "voice": record.voice,
            "embedding_depth": record.embedding_depth,
            "input_text": row["input_text"],
            "target_form": record.target_form,
            "predicate": record.predicate_lemma,
            "agent": record.agent,
            "theme": record.theme,
            "agent_modifier": record.agent_modifier,
            "theme_modifier": record.theme_modifier,
        })
    nontrivial_repairs = sum(
        next(row["input_text"] for row in canonical if row["item_id"] == family["m4_pair"]["target_item_id"])
        != next(row["input_text"] for row in canonical if row["item_id"] == family["m4_pair"]["donor_item_id"])
        for family in families
    )
    validation.update({
        "seed": seed,
        "split_counts": dict(Counter(record.split for record in schema)),
        "training_construction_counts": dict(Counter(
            record.construction_family for record in schema
            if record.counterpart_type == "rescue_target" and record.split == "train"
        )),
        "nonidentical_repair_pair_count": nontrivial_repairs,
        "artifact_status": "deterministically regenerated; canonical rows match pre-release repository data",
    })
    payloads = {
        CANONICAL: jsonl_bytes(canonical),
        "data/release/items.csv": output.getvalue().encode(),
        "data/release/matched_sets.jsonl": jsonl_bytes(families),
        "data/release/schema.jsonl": jsonl_bytes([record.to_dict() for record in schema]),
        "data/release/validation.json": json_bytes(validation),
    }
    manifest = {
        "dataset": "BOUNDED-RS controlled semantic-parsing testbed",
        "version": "1.0.0", "seed": seed, "license": "MIT",
        "creators": ["Boyu Cao", "Wenshuo Wang"],
        "generator": "scripts/build_release_data.py",
        "generation_command": "python3 scripts/build_release_data.py --seed 13",
        "provenance": "Existing public phase11_systematicity_v2 rows regenerated using the original schema/template code; not recovered model evidence.",
        "release_index_ids": "GEN-000 through GEN-095 are new zero-based generator-order aliases, not a historical MS-047 identification.",
        "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                  for name, data in payloads.items()},
    }
    payloads["data/release/manifest.json"] = json_bytes(manifest)
    payloads["croissant.json"] = json_bytes(build_croissant(payloads))
    return payloads


def build_croissant(payloads: dict[str, bytes]) -> dict:
    context = {
        "@language": "en", "@vocab": "https://schema.org/",
        "sc": "https://schema.org/", "cr": "http://mlcommons.org/croissant/",
        "dct": "http://purl.org/dc/terms/", "rai": "http://mlcommons.org/croissant/RAI/",
        "prov": "http://www.w3.org/ns/prov#",
        "conformsTo": "dct:conformsTo", "citeAs": "cr:citeAs",
        "recordSet": "cr:recordSet", "field": "cr:field",
        "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
        "source": "cr:source", "fileObject": "cr:fileObject", "extract": "cr:extract",
        "column": "cr:column", "key": "cr:key",
        "data": {"@id": "cr:data", "@type": "@json"},
        "examples": {"@id": "cr:examples", "@type": "@json"},
    }
    for key in ("equivalentProperty", "fileProperty", "fileSet", "format", "includes",
                "isLiveDataset", "jsonPath", "md5", "parentField", "path", "references",
                "regex", "repeated", "replace", "samplingRate", "separator", "subField", "transform"):
        context[key] = "cr:" + key
    descriptions = {
        "item_id": "Unique canonical item identifier.",
        "matched_set_id": "Original counterpart-family identifier and bootstrap cluster key.",
        "release_index_id": "New zero-based generator-order alias; not a historical walkthrough identifier.",
        "split": "Item-level train, dev or gen split; gen easy donors are in dev.",
        "family_split": "Family assignment before gen easy donors are moved to dev.",
        "counterpart_type": "rescue_donor, rescue_target, agent_swap, theme_swap, role_swap, filler_swap or modifier_swap.",
        "construction_family": "Hard-target active/passive voice crossed with relative-clause depth 0/1.",
        "voice": "Actual item voice, active or passive.",
        "embedding_depth": "Actual item relative-clause depth, 0 or 1.",
        "input_text": "Canonical shortest validated English template realization.",
        "target_form": "Gold logical form containing predicate, role fillers and modifier attachments.",
        "predicate": "Predicate lemma.", "agent": "Synthetic agent filler.",
        "theme": "Synthetic theme filler.", "agent_modifier": "Agent-attached adjective.",
        "theme_modifier": "Theme-attached adjective.",
    }
    resources = []
    for name, mime in (("data/release/items.csv", "text/csv"), (CANONICAL, "application/jsonlines")):
        resources.append({
            "@type": "cr:FileObject", "@id": name, "name": Path(name).name,
            "description": "Canonical 672-item testbed, seed 13.",
            "contentUrl": RAW + name, "encodingFormat": mime,
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
            "contentSize": f"{len(payloads[name])} B",
        })
    return {
        "@context": context, "@type": "sc:Dataset",
        "name": "BOUNDED-RS controlled semantic-parsing testbed",
        "description": "96 synthetic seven-item matched sets (672 rows), with exact role/filler/modifier contrasts and repair pairs for bounded representational-systematicity audits. See docs/DATA_CARD.md for construction and limits.",
        "url": REPOSITORY + "/tree/main/data/release", "version": "1.0.0",
        "license": "https://spdx.org/licenses/MIT.html",
        "datePublished": "2026-10-27",
        "creator": [{"@type": "sc:Person", "name": name} for name in ("Boyu Cao", "Wenshuo Wang")],
        "publisher": {"@type": "sc:Organization", "name": "BOUNDED-RS authors"},
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "keywords": ["systematicity", "semantic parsing", "synthetic data", "mechanistic interpretability"],
        "citeAs": "@misc{boundedrs2026, author={Boyu Cao and Wenshuo Wang}, title={BOUNDED-RS: A Bounded Evaluation Protocol and Testbed for Representational Systematicity}, year={2026}, url={" + REPOSITORY + "}}",
        "distribution": resources,
        "recordSet": [{
            "@type": "cr:RecordSet", "@id": "items", "name": "items",
            "description": "672 canonical items; matched_set_id defines the 96 clusters.",
            "key": [{"@id": "items/item_id"}],
            "field": [{
                "@type": "cr:Field", "@id": "items/" + name, "name": name,
                "description": descriptions[name],
                "dataType": "sc:Integer" if name == "embedding_depth" else "sc:Text",
                "source": {"fileObject": {"@id": "data/release/items.csv"}, "extract": {"column": name}},
            } for name in CSV_FIELDS],
        }],
        "rai:dataLimitations": "Finite English templates, 16 invented name tokens, 5 main predicates and a role-filler logical-form schema. Not representative of natural language, all geometric mechanisms, other architectures or general model ability. Historical model evidence is separate from this deterministic dataset.",
        "rai:dataBiases": "The construction explicitly privileges exact discrete role/filler contrasts; fixed names, adjectives, syntactic templates and omitted phenomena can favor the tested Binding operations. No population sampling or demographic balance is claimed.",
        "rai:personalSensitiveInformation": "No personal or sensitive information. Names are synthetic filler tokens and do not identify human subjects.",
        "rai:dataUseCases": "Intended for matched behavioral contrasts, generation-time intervention/repair development and bounded evidence-to-claim audits. Not validated for model ranking, fairness/safety certification or naturalistic language performance.",
        "rai:dataSocialImpact": "Can support more qualified causal claims and transparent negative results. Overgeneralizing finite template results or unvalidated detectors could mislead; keep family, task, architecture and closure-basis limits explicit.",
        "rai:hasSyntheticData": True,
        "prov:wasDerivedFrom": RAW + "src/data/systematicity_schema.py",
        "prov:wasGeneratedBy": {
            "@type": "prov:Activity",
            "description": "Programmatic semantic designs with random.Random(13), 64/16/16 train/dev/gen families, three deterministic templates each, rule-based validation, shortest-valid selection. Labels come directly from semantic designs; no human annotation or language-model generation. Command: python3 scripts/build_release_data.py --seed 13. See data/release/manifest.json and docs/DATA_CARD.md.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-root", type=Path, default=ROOT,
                        help="Write the same repository-relative layout under this directory.")
    parser.add_argument("--check", action="store_true", help="Compare deterministic bytes without writing.")
    args = parser.parse_args()
    if args.seed != 13:
        parser.error("This release is fixed at seed 13; use the Python generator API for other experiments.")
    payloads = build_payloads(args.seed)
    failures = []
    for name, content in payloads.items():
        path = args.output_root / name
        if args.check:
            if not path.exists() or path.read_bytes() != content:
                failures.append(name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    if failures:
        raise SystemExit("Regeneration mismatch: " + ", ".join(failures))
    print(f"{'Verified' if args.check else 'Wrote'} {len(payloads)} deterministic release files; 672 items, 96 sets, seed 13.")


if __name__ == "__main__":
    main()
