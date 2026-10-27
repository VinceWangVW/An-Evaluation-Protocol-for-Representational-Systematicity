"""Regression checks for exact contrasts, held-out construction and tamper detection."""
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run_builder(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_release_data.py"), *arguments],
        cwd=ROOT, text=True, capture_output=True,
    )


def test_release_rejects_tampering_without_overwriting_it(tmp_path):
    generated = run_builder("--output-root", str(tmp_path))
    assert generated.returncode == 0, generated.stderr
    csv_path = tmp_path / "data/release/items.csv"
    tampered = csv_path.read_text().replace("Freya", "UnknownPerson", 1)
    csv_path.write_text(tampered)
    checked = run_builder("--output-root", str(tmp_path), "--check")
    assert checked.returncode != 0
    assert "data/release/items.csv" in checked.stderr
    assert csv_path.read_text() == tampered


def test_training_excludes_held_out_construction_and_role_changes_are_exact():
    records = [json.loads(line) for line in (ROOT / "data/release/schema.jsonl").read_text().splitlines()]
    families = defaultdict(dict)
    for record in records:
        families[record["counterpart_family_id"]][record["counterpart_type"]] = record
    semantic_fields = ("predicate_lemma", "agent", "theme", "agent_modifier", "theme_modifier")
    for family in families.values():
        target = family["rescue_target"]
        if target["metadata"]["family_split"] == "train":
            assert not (target["voice"] == "passive" and target["embedding_depth"] == 1)
        for variant, expected_changes in (
            ("agent_swap", {"agent"}), ("theme_swap", {"theme"}),
            ("filler_swap", {"agent", "theme"}),
            ("modifier_swap", {"agent_modifier", "theme_modifier"}),
        ):
            changed = {field for field in semantic_fields if family[variant][field] != target[field]}
            assert changed == expected_changes
        donor = family["rescue_donor"]
        assert donor["target_form"] == target["target_form"]
        assert donor["voice"] == "active" and donor["embedding_depth"] == 0


def test_release_cannot_be_silently_relabelled_as_another_seed(tmp_path):
    result = run_builder("--seed", "14", "--output-root", str(tmp_path))
    assert result.returncode != 0
    assert "fixed at seed 13" in result.stderr
    assert not (tmp_path / "data").exists()
