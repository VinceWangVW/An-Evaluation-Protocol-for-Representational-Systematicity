"""Build and run the MS-047 matched-set walkthrough.

The panel is generated from an explicit semantic design. Supply --checkpoint
for generation-time interventions or --evidence to score measured L1 signals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.systematicity_schema import (
    REQUIRED_COUNTERPART_TYPES,
    SystematicitySchemaRecord,
    build_counterpart_families,
    build_target_form,
)
from data.surface_templates import render_template_candidates
from data.surface_validation import select_primary_candidates, validate_surface_candidate
from release_analysis import adjudicate

DEFAULT_CONFIG = ROOT / "configs/experiments/ms047.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n" for row in rows).encode()


def _load_design(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    design = cfg["semantic_design"]
    for key in ("agent", "theme", "alternate_agent", "alternate_theme", "agent_modifier", "theme_modifier"):
        if not isinstance(design.get(key), str) or not design[key].strip():
            raise ValueError(f"Semantic design requires a nonempty {key}")
    fillers = [design[key].lower() for key in ("agent", "theme", "alternate_agent", "alternate_theme")]
    if len(set(fillers)) != 4:
        raise ValueError("The four role fillers must be distinct")
    if design["agent_modifier"] == design["theme_modifier"]:
        raise ValueError("Modifier-swap requires two distinct modifiers")
    for key in ("lemma", "past", "participle"):
        if not design.get("predicate", {}).get(key):
            raise ValueError(f"Predicate requires {key}")
    if design["target_voice"] not in ("active", "passive") or design["target_embedding_depth"] not in (0, 1):
        raise ValueError("Target construction requires supported voice and depth")
    if not cfg.get("trace_id") or not cfg.get("benchmark_name"):
        raise ValueError("Walkthrough requires trace_id and benchmark_name")
    return cfg


def build_panel(config_path: Path = DEFAULT_CONFIG) -> dict:
    """Render seven controlled inputs and exact intervention/repair relations."""
    cfg = _load_design(config_path)
    design = cfg["semantic_design"]
    agent, theme = design["agent"], design["theme"]
    alt_agent, alt_theme = design["alternate_agent"], design["alternate_theme"]
    amod, tmod = design["agent_modifier"], design["theme_modifier"]
    voice, depth = design["target_voice"], design["target_embedding_depth"]
    construction = f"{voice}_depth{depth}"
    predicate = design["predicate"]
    context = design["context"]
    base = dict(agent=agent, theme=theme, agent_modifier=amod, theme_modifier=tmod, voice=voice, depth=depth)
    specs = {
        "rescue_donor": {**base, "voice": "active", "depth": 0},
        "rescue_target": dict(base),
        "agent_swap": {**base, "agent": alt_agent},
        "theme_swap": {**base, "theme": alt_theme},
        "role_swap": {**base, "agent": theme, "theme": agent, "agent_modifier": tmod, "theme_modifier": amod},
        "filler_swap": {**base, "agent": alt_agent, "theme": alt_theme},
        "modifier_swap": {**base, "agent_modifier": tmod, "theme_modifier": amod},
    }
    schema = []
    for counterpart_type in REQUIRED_COUNTERPART_TYPES:
        spec = specs[counterpart_type]
        item_id = f"{cfg['trace_id']}:{counterpart_type}"
        schema.append(SystematicitySchemaRecord(
            item_id=item_id, benchmark_name=cfg["benchmark_name"],
            split="dev" if counterpart_type == "rescue_donor" else "gen",
            raw_split="walkthrough", schema_id=item_id,
            counterpart_family_id=cfg["trace_id"], counterpart_type=counterpart_type,
            construction_family=construction, voice=spec["voice"], embedding_depth=spec["depth"],
            predicate_lemma=predicate["lemma"], predicate_past=predicate["past"], predicate_participle=predicate["participle"],
            context_predicate_lemma=context["predicate_lemma"], context_predicate_past=context["predicate_past"],
            agent=spec["agent"], theme=spec["theme"], agent_modifier=spec["agent_modifier"], theme_modifier=spec["theme_modifier"],
            context_filler=context["filler"],
            target_form=build_target_form(predicate_lemma=predicate["lemma"], agent=spec["agent"], theme=spec["theme"],
                                          agent_modifier=spec["agent_modifier"], theme_modifier=spec["theme_modifier"]),
            metadata={"family_split": "walkthrough", "allowed_delta": counterpart_type,
                      "provenance": "programmatic semantic design", "trace_id": cfg["trace_id"], "no_forced_h2": True},
        ))
    candidates = [validate_surface_candidate(record, candidate) for record in schema
                  for candidate in render_template_candidates(record, num_candidates=3)]
    _, examples, validation = select_primary_candidates(schema_records=schema, validated_candidates=candidates)
    rows = [example.to_dict() for example in examples]
    if len(rows) != 7 or any(not candidate["valid"] for candidate in candidates):
        raise ValueError("Every walkthrough counterpart must have valid template realizations")
    family = build_counterpart_families(schema)[0]
    return {"config": cfg, "rows": rows, "schema": [r.to_dict() for r in schema],
            "pairs": family, "validation": validation, "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}


def write_panel(output: Path, config_path: Path = DEFAULT_CONFIG) -> tuple[dict, dict[str, str]]:
    panel = build_panel(config_path)
    output.mkdir(parents=True, exist_ok=True)
    trace_id = panel["config"]["trace_id"]
    payloads = {
        f"{trace_id}_inputs.jsonl": _jsonl_bytes(panel["rows"]),
        f"{trace_id}_schema.jsonl": _jsonl_bytes(panel["schema"]),
        f"{trace_id}_pairs.json": _json_bytes(panel["pairs"]),
    }
    manifest = {
        "trace_id": trace_id, "version": panel["config"]["version"], "license": "MIT",
        "provenance": "programmatic semantic design",
        "semantic_design": panel["config"]["semantic_design"], "config_sha256": panel["config_sha256"],
        "item_count": len(panel["rows"]), "validation": panel["validation"],
        "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} for name, data in payloads.items()},
    }
    payloads[f"{trace_id}_manifest.json"] = _json_bytes(manifest)
    paths = {}
    for name, data in payloads.items():
        path = output / name
        path.write_bytes(data)
        paths[name] = str(path)
    return panel, paths


def build_walkthrough(output: Path, evidence_path: Path | None = None, *, config_path: Path = DEFAULT_CONFIG,
                      checkpoint: str | None = None, device: str | None = None, layer_index: int = -1,
                      max_new_tokens: int = 192, permutations: int = 199, seed: int = 13,
                      profile: dict | None = None, adapter: Any = None) -> dict:
    if evidence_path is not None and (checkpoint is not None or adapter is not None):
        raise ValueError("Choose a checkpoint audit or supplied evidence, not both")
    panel, files = write_panel(output, config_path)
    result = {"trace_id": panel["config"]["trace_id"], "status": "prepared", "item_count": 7,
              "files": files, "operations": ["interchange", "necessity", "repair"],
              "model_checkpoint": checkpoint,
              "command": "python3 scripts/run_walkthrough.py --checkpoint CHECKPOINT --output-dir outputs/walkthrough"}
    if checkpoint is not None or adapter is not None:
        from experiments.matched_audit import audit_matched_sets, write_audit
        if adapter is None:
            from models.adapters import build_model_adapter
            adapter = build_model_adapter(checkpoint, family="seq2seq_lm", device=device, prompt_style="raw")
        payload = audit_matched_sets(panel["rows"], adapter, layer_index=layer_index, max_new_tokens=max_new_tokens,
                                     seed=seed, permutations=permutations, profile=profile)
        result["audit_files"] = write_audit(output, payload)
        result["status"] = "executed"
        result["summary"] = payload["summary"]
    elif evidence_path is not None:
        raw = json.loads(evidence_path.read_text())
        signals = raw.get("signals", raw)
        decision = adjudicate(signals, profile)
        measured_l1 = {"trace_id": result["trace_id"], "source": str(evidence_path), "signals": signals}
        (output / "L1_signals.json").write_bytes(_json_bytes(measured_l1))
        (output / "L2_decision.json").write_bytes(_json_bytes(decision))
        result.update(status="scored", selected_state=decision["selected_state"],
                      evidence_files={"L1": str(output / "L1_signals.json"), "L2": str(output / "L2_decision.json")})
    (output / "walkthrough.json").write_bytes(_json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/walkthrough"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", help="T5 checkpoint or Hugging Face model identifier")
    source.add_argument("--evidence", type=Path, help="Measured L1 signals JSON")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--layer-index", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--profile", type=Path, help="Decision-profile JSON")
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text()) if args.profile else None
    result = build_walkthrough(args.output_dir, args.evidence, config_path=args.config, checkpoint=args.checkpoint,
                               device=args.device, layer_index=args.layer_index, max_new_tokens=args.max_new_tokens,
                               seed=args.seed, permutations=args.permutations, profile=profile)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
