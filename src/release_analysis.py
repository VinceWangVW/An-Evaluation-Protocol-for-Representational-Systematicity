"""Scoring, statistical analyses and model audit workflows for BOUNDED-RS.

Decision profiles, input contracts and report schemas are documented in
docs/ANALYSIS_GUIDE.md.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

STATES = ("Unsupported", "Present-only", "Causal-use", "Undetermined")
DEFAULT_PROFILE = {
    "version": "bounded-rs-v1",
    "provenance": "Versioned decision profile; see docs/ANALYSIS_GUIDE.md for scope and formulas.",
    "features": {
        "b0_gen_clause_f1": {"boundary": 0.570312, "scale": 0.10, "direction": 1},
        "m1_semantic_role_accuracy": {"boundary": 0.914062, "scale": 0.05, "direction": 1},
        "m1_surface_control_gap": {"boundary": -0.085938, "scale": 0.05, "direction": 1},
        "m2_advantage_mean": {"boundary": 0.245002, "scale": 0.05, "direction": 1},
        "m2_permutation_p_value": {"boundary": 0.05, "scale": 0.01, "direction": -1},
    },
    "weights": {"presence_role": 0.60, "use_effect": 0.60, "behavior": 0.04},
    "gates": {"realism": 0.60, "linkage": 0.50, "coverage": 0.80, "margin": 0.10,
              "necessity": 0.05, "repair": 0.245002},
    "closure": {"discovery_gap": 0.05, "necessity_gap": 0.05, "null_max": 0.10,
                "minimum_survivors": 8, "minimum_steps": 2, "minimum_templates": 2},
}
REQUIRED = tuple(DEFAULT_PROFILE["features"]) + (
    "m3_selective_impairment_clause_f1", "m4_advantage_mean", "realism_pass_rate",
    "linkage_pass_rate", "evidence_coverage", "audit_ready", "state_anchor_ready",
)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Expected a nonempty JSONL file of objects")
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def score_feature(value: float, boundary: float, scale: float, direction: int = 1) -> float:
    if scale <= 0 or direction not in (-1, 1) or not all(math.isfinite(x) for x in (value, boundary, scale)):
        raise ValueError("Finite values, positive scale and direction +/-1 required")
    z = direction * (value - boundary) / scale
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def adjudicate(signals: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return an inspectable decision, abstaining on missing required evidence.

    Signals use the existing repository names plus explicit linkage/coverage rates.
    Numeric zero is observed evidence; None, booleans and nonfinite numeric metrics
    are unavailable. Strict gates follow the camera-ready text's 'exceed' wording.
    """
    cfg = config or DEFAULT_PROFILE
    missing = [key for key in REQUIRED if key not in signals or signals[key] is None]
    for key in REQUIRED:
        if key in ("audit_ready", "state_anchor_ready"):
            if key in signals and not isinstance(signals[key], bool) and key not in missing:
                missing.append(key)
        elif key in signals and not _number(signals[key]) and key not in missing:
            missing.append(key)
    base = {"profile_version": cfg["version"], "selected_state": "Undetermined",
            "missing_features": missing, "feature_transforms": {}, "group_scores": {},
            "state_scores": {}, "margin": None, "gates": {}}
    if missing:
        return base
    for key in ("b0_gen_clause_f1", "m1_semantic_role_accuracy", "m2_permutation_p_value",
                "realism_pass_rate", "linkage_pass_rate", "evidence_coverage"):
        if not 0 <= signals[key] <= 1:
            raise ValueError(f"{key} must lie in [0,1]")
    scores = {}
    for key, spec in cfg["features"].items():
        value = signals[key]
        score = score_feature(value, **spec)
        scores[key] = score
        base["feature_transforms"][key] = {"raw": value, **spec, "score": score}
    w = cfg["weights"]
    if not all(0 <= v <= 1 for v in w.values()):
        raise ValueError("All mixture weights must lie in [0,1]")
    behavior = scores["b0_gen_clause_f1"]
    presence = w["presence_role"] * scores["m1_semantic_role_accuracy"] + (1-w["presence_role"]) * scores["m1_surface_control_gap"]
    use = w["use_effect"] * scores["m2_advantage_mean"] + (1-w["use_effect"]) * scores["m2_permutation_p_value"]
    p = w["behavior"]
    states = {
        "Causal-use": p * behavior + (1-p) * (presence + use) / 2,
        "Present-only": p * behavior + (1-p) * (presence + 1-use) / 2,
        "Unsupported": p * (1-behavior) + (1-p) * (2-presence-use) / 2,
    }
    ranked = sorted(states, key=states.get, reverse=True)
    margin = states[ranked[0]] - states[ranked[1]]
    gates = {
        "audit_ready": signals["audit_ready"] and signals["state_anchor_ready"],
        "coverage": signals["evidence_coverage"] > cfg["gates"]["coverage"],
        "margin": margin > cfg["gates"]["margin"],
        "realism": signals["realism_pass_rate"] > cfg["gates"]["realism"],
        "linkage": signals["linkage_pass_rate"] > cfg["gates"]["linkage"],
        "necessity": signals["m3_selective_impairment_clause_f1"] > cfg["gates"]["necessity"],
        "repair": signals["m4_advantage_mean"] > cfg["gates"]["repair"],
    }
    selected = ranked[0]
    if not all(gates[k] for k in ("audit_ready", "coverage", "margin")):
        selected = "Undetermined"
    elif selected == "Causal-use" and (presence <= .5 or use <= .5 or not all(gates.values())):
        selected = "Undetermined"
    elif selected == "Present-only" and (presence <= .5 or use >= .5):
        selected = "Undetermined"
    elif selected == "Unsupported" and (presence >= .5 or use >= .5):
        selected = "Undetermined"
    base.update(selected_state=selected, group_scores={"behavior": behavior, "presence": presence, "use": use},
                state_scores={**states, "Undetermined": 1-max(states.values())}, margin=margin, gates=gates)
    return base


def _patch(cfg: dict, path: str, value: Any) -> None:
    keys = path.split(".")
    target = cfg
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            raise ValueError(f"Unknown configuration path {path}")
        target = target[key]
    if keys[-1] not in target:
        raise ValueError(f"Unknown configuration path {path}")
    target[keys[-1]] = value


def closure_decision(objects: list[dict], cfg: dict, *, sealed: bool, complete: bool,
                     present_route_complete: bool, present_route_found: bool,
                     frozen_basis: list[dict] | None = None) -> dict:
    """Require exhaustive frozen coverage and completed direct/slice presence tests."""
    if not all(isinstance(v, bool) for v in (sealed,complete,present_route_complete,present_route_found)):
        raise ValueError("Closure context requires explicit booleans")
    if not all((sealed, complete, present_route_complete)) or not objects or not frozen_basis:
        return {"selected_state": "Undetermined", "accepted_objects": [], "reason": "closure_incomplete"}
    expected = {o["object_id"]:o for o in frozen_basis}
    if len(expected) != len(frozen_basis) or len({o["object_id"] for o in objects}) != len(objects):
        raise ValueError("Frozen basis and observed object IDs must be unique")
    if {o["object_id"] for o in objects} != set(expected):
        return {"selected_state": "Undetermined", "accepted_objects": [], "reason": "frozen_basis_coverage_incomplete"}
    classes = defaultdict(list)
    for obj in objects:
        for key in ("object_id", "object_class", "discovery_gap", "necessity_gap", "null_effect",
                    "realism_pass_rate", "step", "template"):
            if key not in obj or obj[key] is None:
                raise ValueError(f"Closure object missing {key}")
        for key in ("object_class", "step", "template"):
            if obj[key] != expected[obj["object_id"]].get(key):
                raise ValueError(f"Observed {key} differs from frozen basis")
        if not all(_number(obj[k]) for k in ("discovery_gap","necessity_gap","null_effect","realism_pass_rate")):
            raise ValueError("Closure metrics must be finite numbers")
        if not 0 <= obj["realism_pass_rate"] <= 1:
            raise ValueError("Closure realism must lie in [0,1]")
        classes[obj["object_class"]].append(obj)
    accepted = []
    incomplete_candidates = []
    object_present_only = False
    c = cfg["closure"]
    for rows in classes.values():
        survivors = [x for x in rows if x["discovery_gap"] > c["discovery_gap"]]
        class_ok = (len(survivors) >= c["minimum_survivors"] and
                    len({x["step"] for x in survivors}) >= c["minimum_steps"] and
                    len({x["template"] for x in survivors}) >= c["minimum_templates"])
        if class_ok:
            for obj in survivors:
                necessity_ok = (obj["necessity_gap"] > c["necessity_gap"] and
                                abs(obj["null_effect"]) < c["null_max"] and
                                obj["realism_pass_rate"] > cfg["gates"]["realism"])
                if necessity_ok:
                    decision = adjudicate(obj.get("signals",{}), cfg)
                    if decision["selected_state"] == "Causal-use":
                        accepted.append(obj["object_id"])
                    elif decision["selected_state"] == "Undetermined":
                        incomplete_candidates.append(obj["object_id"])
                    elif decision["selected_state"] == "Present-only":
                        object_present_only = True
    state = ("Causal-use" if accepted else "Undetermined" if incomplete_candidates else
             "Present-only" if present_route_found or object_present_only else "Unsupported")
    return {"selected_state": state, "accepted_objects": accepted, "incomplete_candidates":incomplete_candidates}


def sensitivity(panel: dict, grid: dict, cfg: dict | None = None) -> dict:
    cfg = cfg or DEFAULT_PROFILE
    blocks = grid["blocks"]
    if len(blocks) != 6 or any(len(b["levels"]) != 3 for b in blocks):
        raise ValueError("Sensitivity requires six blocks with three levels each")
    if not panel.get("calibration") or not panel.get("targets"):
        raise ValueError("Nonempty labeled calibration and separate target panels required")
    calibration_ids = {r["id"] for r in panel["calibration"]}
    target_ids = {r["id"] for r in panel["targets"]}
    if calibration_ids & target_ids:
        raise ValueError("Calibration and targets must be disjoint")
    for row in panel["calibration"]:
        if row.get("expected_state") not in STATES[:3]:
            raise ValueError("Calibration requires an intended evidential state")
    results = []
    for levels in itertools.product(range(3), repeat=6):
        trial = copy.deepcopy(cfg)
        for block, level in zip(blocks, levels):
            for path, value in block["levels"][level].items():
                _patch(trial, path, value)
        calibration = [adjudicate(r["signals"], trial)["selected_state"] for r in panel["calibration"]]
        compatible = all(p == r["expected_state"] for p, r in zip(calibration, panel["calibration"]))
        row = {"levels": levels, "calibration_compatible": compatible, "calibration_states": calibration}
        if compatible:
            row["targets"] = {r["id"]: adjudicate(r["signals"], trial)["selected_state"] for r in panel["targets"]}
            if "closure" in panel:
                row["closure"] = closure_decision(cfg=trial, **panel["closure"])
        results.append(row)
    return {"status": "executed_on_supplied_records", "profile_version": cfg["version"],
            "grid_provenance": grid.get("provenance"), "configurations": len(results),
            "compatible_count": sum(r["calibration_compatible"] for r in results), "rows": results}


def _percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    pos = fraction * (len(values)-1)
    lo = int(pos)
    return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo]) * (pos-lo)


def paired_sign_p_value(effects: list[float], rng: random.Random, permutations: int = 999) -> float:
    if not effects or not all(_number(v) for v in effects) or permutations < 19:
        raise ValueError("Finite paired effects and at least 19 random sign permutations required")
    observed = mean(effects)
    extreme = sum(mean(v * rng.choice((-1, 1)) for v in effects) >= observed for _ in range(permutations))
    return (extreme+1)/(permutations+1)


def aggregate_clusters(rows: list[dict], rng: random.Random, permutations: int = 199) -> dict:
    """Average per-set sufficient metrics, recomputing p from paired effect signs.

    Input rows are one observation per matched set, not token/item rows. A p-value
    must never be averaged across sets. Each row contains signed m2 advantage;
    bootstrap resampling preserves its pairing with other metrics.
    """
    keys = [k for k in REQUIRED if k != "m2_permutation_p_value"]
    for row in rows:
        if any(k not in row["signals"] for k in keys):
            raise ValueError("Per-set rows must contain every required sufficient metric except p-value")
    result = {}
    for key in keys:
        values = [r["signals"][key] for r in rows]
        if key in ("audit_ready", "state_anchor_ready"):
            if not all(isinstance(v, bool) for v in values):
                raise ValueError(f"{key} requires booleans")
            result[key] = all(values)
        else:
            if not all(_number(v) for v in values):
                raise ValueError(f"{key} requires finite per-set sufficient metrics")
            result[key] = mean(values)
    effects = [r["signals"]["m2_advantage_mean"] for r in rows]
    result["m2_permutation_p_value"] = paired_sign_p_value(effects, rng, permutations)
    return result


def cluster_bootstrap(rows: list[dict], *, replicates: int = 1000, seed: int = 13,
                      expected_sets: int = 96, cfg: dict | None = None, permutations: int = 199) -> dict:
    ids = [r["matched_set_id"] for r in rows]
    if len(ids) != len(set(ids)) or len(rows) != expected_sets:
        raise ValueError(f"Expected exactly {expected_sets} unique matched-set aggregate rows")
    if replicates < 1 or permutations < 19:
        raise ValueError("Positive replicates and at least 19 sign permutations required")
    rng = random.Random(seed)
    baseline = aggregate_clusters(rows, random.Random(seed), permutations)
    distributions = defaultdict(list)
    state_counts = Counter()
    samples = []
    closure_counts = Counter()
    if any("closure_objects" in row for row in rows):
        if not all("closure_objects" in row and "closure_context" in row for row in rows):
            raise ValueError("Closure bootstrap needs objects and closure_context for every matched set")
        object_ids = {x["object_id"] for x in rows[0]["closure_objects"]}
        if not object_ids or any({x["object_id"] for x in r["closure_objects"]} != object_ids or
                                 len(r["closure_objects"]) != len(object_ids) for r in rows):
            raise ValueError("Every cluster must cover the same frozen closure objects")
        descriptors = {o["object_id"]: (o["object_class"],o["step"],o["template"]) for o in rows[0]["closure_objects"]}
        for row in rows:
            for obj in row["closure_objects"]:
                if (obj["object_class"],obj["step"],obj["template"]) != descriptors[obj["object_id"]]:
                    raise ValueError("Frozen object descriptors must be stable across clusters")
    for rep in range(replicates):
        indices = [rng.randrange(len(rows)) for _ in rows]
        selected = [rows[i] for i in indices]
        aggregate = aggregate_clusters(selected, rng, permutations)
        prediction = adjudicate(aggregate, cfg)
        state_counts[prediction["selected_state"]] += 1
        for key, value in aggregate.items():
            if _number(value):
                distributions[key].append(value)
        record = {"replicate": rep, "indices": indices, "selected_state": prediction["selected_state"]}
        if closure_counts or "closure_objects" in rows[0]:
            objects = []
            for oid in sorted(object_ids):
                evidence = [next(o for o in r["closure_objects"] if o["object_id"] == oid) for r in selected]
                obj = dict(evidence[0])
                for key in ("discovery_gap", "necessity_gap", "null_effect", "realism_pass_rate"):
                    obj[key] = mean(o[key] for o in evidence)
                if all("signals" in o for o in evidence):
                    obj["signals"] = aggregate_clusters([{"signals":o["signals"]} for o in evidence],rng,permutations)
                else:
                    obj.pop("signals",None)
                objects.append(obj)
            context = rows[0]["closure_context"]
            if any(r["closure_context"] != context for r in rows):
                raise ValueError("Closure context must be identical across clusters")
            closure = closure_decision(objects, cfg or DEFAULT_PROFILE, **context)
            closure_counts[closure["selected_state"]] += 1
            record["closure"] = closure
        samples.append(record)
    return {"status": "executed_on_supplied_records", "seed": seed, "replicates": replicates,
            "matched_sets": len(rows), "unit": "matched-set cluster", "permutations_per_replicate": permutations,
            "input_provenance": sorted({str(r.get("provenance","not declared")) for r in rows}),
            "state_counts": dict(state_counts), "closure_state_counts": dict(closure_counts),
            "point_estimate": baseline,
            "intervals_95": {k: [_percentile(v,.025), _percentile(v,.975)] for k,v in distributions.items()},
            "samples": samples}


def component_rules(rows: list[dict], cfg: dict | None = None) -> dict:
    """Compare rules on the same truth-labeled rows, retaining every decision."""
    cfg = cfg or DEFAULT_PROFILE
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Component panel IDs must be unique")
    predictions = []
    for row in rows:
        if row.get("truth") not in ("causal", "noncausal") or not row.get("truth_provenance"):
            raise ValueError("Each unit needs truth=causal/noncausal and truth_provenance")
        full = adjudicate(row["signals"], cfg)
        if full["missing_features"]:
            raise ValueError("Component comparison cannot silently treat absent evidence as a negative")
        groups = full["group_scores"]
        predictions.append({"id": row["id"], "truth": row["truth"], "truth_provenance": row["truth_provenance"],
                            "recoverable_implies_causal": groups["presence"] > .5,
                            "ungated_intervention": groups["use"] > .5,
                            "full_state": full["selected_state"]})
    causal = sum(r["truth"] == "causal" for r in predictions)
    noncausal = len(rows)-causal
    stats = {}
    for name in ("recoverable_implies_causal", "ungated_intervention", "full"):
        positive = lambda r: r["full_state"] == "Causal-use" if name == "full" else r[name]
        stats[name] = {"false_causal": sum(positive(r) and r["truth"] == "noncausal" for r in predictions),
                       "noncausal_denominator": noncausal,
                       "causal_retained": sum(positive(r) and r["truth"] == "causal" for r in predictions),
                       "causal_denominator": causal,
                       "abstentions": sum(r["full_state"] == "Undetermined" for r in predictions) if name == "full" else 0}
    return {"status": "executed_on_supplied_truth_records", "units": len(rows), "statistics": stats, "predictions": predictions}


def transfer_compare(calibration: list[dict], targets: list[dict], cfg: dict | None = None) -> dict:
    """Compare frozen and control-only recalibrated profiles on a scale/task port.

    Recalibration uses feature midpoint boundaries between intended low/high
    controls, retaining scales, mixture weights and gates. Targets are never used.
    The fitted profile records the calibration IDs and complete parameters.
    """
    cfg = cfg or DEFAULT_PROFILE
    if not calibration or not targets:
        raise ValueError("Nonempty calibration and target panels required")
    if {r["id"] for r in calibration} & {r["id"] for r in targets}:
        raise ValueError("Transfer calibration and target IDs must be disjoint")
    if any(r.get("expected_state") not in STATES[:3] for r in calibration):
        raise ValueError("Transfer controls require intended evidential states")
    recalibrated = copy.deepcopy(cfg)
    recalibrated["version"] += "-control-only-recalibration"
    for key, descriptor in recalibrated["features"].items():
        low=[]; high=[]
        for row in calibration:
            value = row["signals"].get(key)
            if not _number(value):
                raise ValueError(f"Control missing finite feature {key}")
            present = row["expected_state"] != "Unsupported"
            causal = row["expected_state"] == "Causal-use"
            positive = present if key.startswith("m1_") else causal
            # Low-direction features support a state when raw values are small.
            (high if positive else low).append(value)
        if not low or not high:
            raise ValueError(f"Control panel does not bracket {key}")
        descriptor["boundary"] = (mean(low)+mean(high))/2
    rows = []
    for row in targets:
        rows.append({"id":row["id"], "frozen":adjudicate(row["signals"],cfg),
                     "recalibrated":adjudicate(row["signals"],recalibrated)})
    return {"status":"executed_on_supplied_transfer_evidence", "recalibrated_profile":recalibrated,
            "calibration_ids":[r["id"] for r in calibration], "targets":rows,
            "decision_agreement":sum(r["frozen"]["selected_state"] == r["recalibrated"]["selected_state"] for r in rows),
            "calibration_only":True}


def engineered_truth_controls(seeds: list[int], *, candidates: int = 12, observations: int = 128) -> dict:
    """Execute an engineered structural-model truth-control experiment.

    A distributed bridge stores a signed binary variable as z=a*x over a random
    unit vector. A paired intervention swaps x through the bridge; causal models
    decode z, present-only models instead read an independent unpatched route.
    Discovery uses disjoint observations and localizes by recoverability.
    """
    if len(seeds) != len(set(seeds)) or not seeds or candidates < 2 or observations < 20:
        raise ValueError("Unique seeds, >=2 candidates and >=20 observations required")
    rows = []
    for family in ("binding", "cross_step_geometry"):
        for seed in seeds:
            for condition in ("causal", "present_only", "null"):
                rng = random.Random(seed)
                planted = rng.randrange(candidates)
                direction = [1.0] if family == "binding" else [rng.uniform(-1,1) for _ in range(8)]
                norm = math.sqrt(sum(x*x for x in direction))
                direction = [x/norm for x in direction]
                def sample(n: int) -> tuple[list[int], list[list[list[float]]]]:
                    labels = [rng.choice((-1,1)) for _ in range(n)]
                    states = []
                    for label in labels:
                        objects = []
                        for index in range(candidates):
                            x = label if index == planted and condition != "null" else rng.choice((-1,1))
                            objects.append([v*x for v in direction])
                        states.append(objects)
                    return labels, states
                labels, discovery = sample(observations)
                correlations = [abs(mean(label * sum(v*d for v,d in zip(state[j],direction)) for label,state in zip(labels,discovery))) for j in range(candidates)]
                selected = max(range(candidates), key=lambda j: correlations[j])
                labels, confirmation = sample(observations)
                readout = mean((sum(v*d for v,d in zip(s[selected],direction)) > 0) == (x > 0) for x,s in zip(labels,confirmation))
                # Decode actual states after swapping/removing/restoring the chosen object.
                interchange = []; necessity = []; repair = []
                for x,state in zip(labels,confirmation):
                    decoded = lambda vector: 1 if sum(v*d for v,d in zip(vector,direction)) > 0 else -1
                    output = lambda vector: decoded(vector) if condition == "causal" else x
                    original = state[selected]
                    swapped = [-v for v in original]
                    removed = [0.0 for _ in original]
                    interchange.append(float(output(swapped) == -x) - float(output(original) == -x))
                    necessity.append(float(output(original) == x) - float(output(removed) == x))
                    repair.append(float(output(original) == x) - float(output(swapped) == x))
                signals = {"b0_gen_clause_f1": 1.0, "m1_semantic_role_accuracy": readout,
                    "m1_surface_control_gap": readout-.5, "m2_advantage_mean": mean(interchange),
                    "m2_permutation_p_value": paired_sign_p_value(interchange, rng),
                    "m3_selective_impairment_clause_f1": mean(necessity), "m4_advantage_mean": mean(repair),
                    "realism_pass_rate": 1.0, "linkage_pass_rate": 1.0, "evidence_coverage": 1.0,
                    "audit_ready": True, "state_anchor_ready": True}
                # In null models the readout is chance; no claimed recoverability.
                expected = {"causal":"Causal-use", "present_only":"Present-only", "null":"Unsupported"}[condition]
                rows.append({"id": f"{family}-{seed}-{condition}", "family": family, "seed": seed,
                    "condition": condition, "planted_object": planted, "selected_object": selected,
                    "discovery_correlations": correlations, "independent_confirmation_n": observations,
                    "expected_state": expected, "signals": signals,
                    "prediction": adjudicate(signals), "truth": "causal" if condition == "causal" else "noncausal",
                    "truth_provenance": "engineered structural model with declared signal-use equation"})
    return {"status": "executed_computational_control", "experiment_context": "engineered_structural_model",
            "design": "paired seeds, blind candidate selection, disjoint confirmation, explicit swap/removal/repair",
            "rows": rows, "correct": sum(r["expected_state"] == r["prediction"]["selected_state"] for r in rows),
            "component_rules": component_rules(rows)}


COMPONENT_CASE_COUNTS = {
    "causal": 7,
    "present_only": 6,
    "generic_decoder_gain": 6,
    "wrong_role": 6,
    "off_manifold": 6,
    "aggregate_unlinked": 5,
    "weak_shortcut": 5,
}


def component_failure_panel(*, seed: int = 13, observations: int = 128,
                            cfg: dict | None = None) -> dict:
    """Execute a prespecified 41-case structural causal-model experiment.

    Seven models route their prediction through the declared signal. The 34
    noncausal models route natural predictions through a separate bypass; their
    intervention operators instantiate recoverable-only, generic-gain,
    wrong-role, off-manifold, unlinked, and weak-shortcut failure mechanisms.
    Every score is calculated from recorded predictions and matched controls.
    This computational panel is identified by its equations and parameters.
    """
    if observations < 20:
        raise ValueError("At least 20 observations per case are required")
    cfg = cfg or DEFAULT_PROFILE
    rows = []
    definitions = []
    for mechanism,count in COMPONENT_CASE_COUNTS.items():
        for variant in range(count):
            case_seed = seed + len(rows)*1009
            rng = random.Random(case_seed)
            family = "binding" if variant % 2 == 0 else "cross_step_geometry"
            width = 1 if family == "binding" else 8
            direction = [rng.uniform(-1,1) for _ in range(width)]
            length = math.sqrt(sum(x*x for x in direction))
            direction = [x/length for x in direction]
            causal = mechanism == "causal"
            records = []
            for index in range(observations):
                label = rng.choice((-1,1))
                stored = rng.choice((-1,1)) if mechanism == "weak_shortcut" else label
                vector = [stored*x for x in direction]
                decode = lambda z: 1 if sum(a*b for a,b in zip(z,direction)) >= 0 else -1
                baseline = decode(vector) if causal else label
                swap = [-x for x in vector]
                zero = [0.0]*width
                # A noncausal model's natural equation is y=bypass. Some
                # intervention operators additionally touch a decoder route.
                side_effect = mechanism in ("generic_decoder_gain","wrong_role","off_manifold","aggregate_unlinked")
                side_effect = side_effect or (mechanism == "weak_shortcut" and index % 5 == 0)
                intervened = decode(swap) if causal or side_effect else label
                control_effect = mechanism in ("generic_decoder_gain","wrong_role")
                controlled = decode(swap) if control_effect else label
                removed = decode(zero) if causal or side_effect else label
                control_removed = decode(zero) if control_effect else label
                corrupted = decode(swap) if causal or side_effect else label
                repaired = decode(vector) if causal or side_effect else label
                control_repaired = decode(vector) if control_effect else corrupted
                natural_state = vector+[0.0]
                intervention_state = swap+[10.0 if mechanism == "off_manifold" else 0.0]
                natural_norm = math.sqrt(sum(x*x for x in natural_state))
                intervention_norm = math.sqrt(sum(x*x for x in intervention_state))
                records.append({
                    "observation":index,"label":label,"stored_signal":stored,"baseline_prediction":baseline,
                    "counterfactual_label":-label,"interchange_prediction":intervened,
                    "interchange_control_prediction":controlled,
                    "necessity_prediction":removed,"necessity_control_prediction":control_removed,
                    "corrupted_prediction":corrupted,"repair_prediction":repaired,
                    "repair_control_prediction":control_repaired,
                    "readout_correct":decode(vector) == label,
                    "surface_control_correct":rng.choice((-1,1)) == label,
                    "filler_control_correct":rng.choice((-1,1)) == label,
                    "predicate_control_correct":rng.choice((-1,1)) == label,
                    "natural_state":natural_state,"intervention_state":intervention_state,
                    "natural_state_norm":natural_norm,"intervention_state_norm":intervention_norm,
                    "realism_pass":intervention_norm <= natural_norm + 1e-9,
                    "intended_item_role":mechanism != "aggregate_unlinked",
                })
            effects = [float(r["interchange_prediction"] == r["counterfactual_label"])
                       -float(r["interchange_control_prediction"] == r["counterfactual_label"]) for r in records]
            necessity = [float(r["necessity_control_prediction"] == r["label"])
                         -float(r["necessity_prediction"] == r["label"]) for r in records]
            repair = [float(r["repair_prediction"] == r["label"])
                      -float(r["repair_control_prediction"] == r["label"]) for r in records]
            readout = mean(r["readout_correct"] for r in records)
            signals = {"b0_gen_clause_f1":mean(r["baseline_prediction"] == r["label"] for r in records),
                "m1_semantic_role_accuracy":readout,
                "m1_surface_control_gap":readout-mean(r["surface_control_correct"] for r in records),
                "m1_filler_control_accuracy":mean(r["filler_control_correct"] for r in records),
                "m1_predicate_control_accuracy":mean(r["predicate_control_correct"] for r in records),
                "m2_advantage_mean":mean(effects),"m2_permutation_p_value":paired_sign_p_value(effects,rng),
                "m3_selective_impairment_clause_f1":mean(necessity),"m4_advantage_mean":mean(repair),
                "realism_pass_rate":mean(r["realism_pass"] for r in records),
                "linkage_pass_rate":mean(r["intended_item_role"] for r in records),
                "evidence_coverage":1.0,"audit_ready":True,"state_anchor_ready":True}
            case_id = f"component-{len(rows)+1:03d}"
            equation = "y=decode(z)" if causal else "y=bypass; declared z is unused in natural generation"
            definitions.append({"id":case_id,"mechanism":mechanism,"variant":variant,
                                "seed":case_seed,"natural_equation":equation,"family":family})
            rows.append({"id":case_id,"family":family,"mechanism":mechanism,"seed":case_seed,
                         "truth":"causal" if causal else "noncausal",
                         "truth_provenance":{"context":"structural_causal_model","natural_equation":equation,
                                             "intervention_mechanism":mechanism},
                         "signals":signals,"raw_observations":records,"decision":adjudicate(signals,cfg)})
    return {"status":"executed_computational_experiment","experiment_context":"structural_causal_model",
            "design_version":"component-panel-v1","seed":seed,"observations_per_case":observations,
            "case_counts":COMPONENT_CASE_COUNTS,"cases":definitions,"rows":rows,
            "comparison":component_rules(rows,cfg)}


def make_suite_plan(root: Path, output_dir: Path, configuration: dict | None = None) -> dict:
    """Prepare a complete training, controls, uncertainty and transfer workflow."""
    config = configuration or {"seeds":[0,1,2,3,4],"transfer_seeds":[0,1,2],
        "benchmark":"phase11_systematicity_v2","bootstrap_replicates":1000,
        "bootstrap_seed":13,"grid":"configs/release/sensitivity_grid.json"}
    output_dir.mkdir(parents=True,exist_ok=True)
    commands = []
    plans = []
    for name,model,seeds in (("t5small","google-t5/t5-small",config["seeds"]),
                             ("t5base","google-t5/t5-base",config["transfer_seeds"])):
        plan_dir = output_dir/name
        plan = make_training_plan(root,plan_dir,model=model,seeds=seeds,benchmark=config["benchmark"])
        commands.extend(plan["commands"])
        commands.append([sys.executable,"scripts/run_release_analysis.py","transfer-compare","--panel",
                         str(plan_dir/"binding_evidence.json"),"--output",str(plan_dir/"binding_transfer_compare.json")])
        if name == "t5small":
            commands.append([sys.executable,"scripts/run_release_analysis.py","sensitivity","--panel",
                             str(plan_dir/"binding_evidence.json"),"--grid",config["grid"],
                             "--output",str(plan_dir/"sensitivity.json")])
            if plan["audit_layout"] == "seven_item_matched_sets":
                generated_panel = load_json(plan_dir/"binding_panel.json")
                for target in generated_panel["targets"]:
                    if "target" not in target["tags"]:
                        continue
                    seed,run = target["seed"],target["run_name"]
                    commands.append([sys.executable,"scripts/run_release_analysis.py","bootstrap","--rows",
                        str(plan_dir/"audits"/run/"L1_per_set.jsonl"),"--replicates",str(config["bootstrap_replicates"]),
                        "--seed",str(config["bootstrap_seed"]),"--output",str(plan_dir/f"bootstrap_seed{seed}.json")])
        plans.append({"model":model,"seeds":seeds,"plan":str(plan_dir/"plan.json")})
    commands += [
        [sys.executable,"scripts/run_release_analysis.py","truth-controls","--seeds","0","1","2",
         "--output",str(output_dir/"truth_controls.json")],
        [sys.executable,"scripts/run_release_analysis.py","component-panel","--output",str(output_dir/"component_panel.json")],
    ]
    if config.get("slog_benchmark"):
        slog = make_training_plan(root,output_dir/"slog",model="google-t5/t5-small",
                                  seeds=config["transfer_seeds"],benchmark=config["slog_benchmark"])
        commands.extend(slog["commands"])
        plans.append({"model":"google-t5/t5-small","benchmark":config["slog_benchmark"],"plan":str(output_dir/"slog/plan.json")})
    result = {"status":"plan_only","suite_version":"bounded-rs-suite-v1","configuration":config,
              "model_plans":plans,"commands":commands}
    write_json(output_dir/"suite_plan.json",result)
    return result


def make_training_plan(root: Path, output_dir: Path, *, model: str = "google-t5/t5-small",
                       seeds: list[int] | None = None, benchmark: str = "phase11_systematicity_v2") -> dict:
    """Generate actual train -> discovery -> generation intervention commands."""
    seeds = [0,1,2,3,4] if seeds is None else seeds
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Independent seed list must be nonempty and unique")
    if set(seeds) & {seed+1000 for seed in seeds}:
        raise ValueError("Target and shifted calibration seed sets must be disjoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = model.rsplit("/",1)[-1]
    if benchmark != "phase11_systematicity_v2":
        model_tag += "_"+re.sub(r"[^a-zA-Z0-9_-]+","_",benchmark)
    panel = {"panel_name": "release_"+model_tag, "benchmark": benchmark,
             "model_name_or_path": model, "prompt_style": "raw", "layers": [-1],
             "metadata": {"operationalization_family": "binding", "provenance": "Prespecified independent training and calibration seeds."},
             "targets": []}
    for seed in seeds:
        for family in ("binding", "geometry"):
            run = f"release_{model_tag}_{family}_seed{seed}"
            base_target = {"target_id": run, "run_name": run, "phase11_prefix": run,
                "seed": seed, "num_train_epochs": 8, "learning_rate": 5e-5, "train_batch_size": 8,
                "num_beams": 1, "encoder_geometry_transform": "feature_mask" if family == "geometry" else None,
                "encoder_geometry_keep_ratio": .75 if family == "geometry" else None,
                "tags": [family,"target"], "notes": "Protocol run; checkpoints and evidence are generated under runs/."}
            panel["targets"].append(base_target)
            for regime in ("causal", "present_only", "unsupported"):
                control = copy.deepcopy(base_target)
                control_run = run+"_calibration_"+regime
                control.update(target_id=control_run, run_name=control_run, phase11_prefix=control_run,
                               seed=seed+1000, panel_regime=regime, tags=[family,"calibration"])
                if regime == "present_only":
                    control.update(reuse_checkpoint_from=run+"_calibration_causal", num_train_epochs=0,
                                   decoder_cross_attention_scale=.25)
                elif regime == "unsupported":
                    control["train_target_permutation_seed"] = seed+1000
                panel["targets"].append(control)
    panel_path = output_dir / "training_panel.json"
    write_json(panel_path, panel)
    commands = [[sys.executable, "scripts/run/train_target_families.py", "--panel-config", str(panel_path)]]
    data_path = root/"data/processed"/(benchmark+".jsonl")
    complete_sets = False
    if data_path.exists():
        members = defaultdict(set)
        for record in read_rows(data_path):
            metadata = record.get("metadata",{})
            if metadata.get("counterpart_family_id"):
                members[metadata["counterpart_family_id"]].add(metadata.get("counterpart_type"))
        required_types = {"rescue_donor","rescue_target","agent_swap","theme_swap","role_swap","filler_swap","modifier_swap"}
        complete_sets = bool(members) and all(types == required_types for types in members.values())
    # Binding audits enumerate every member of every complete matched set.
    # Geometry retains its object-specific discovery/generation operations.
    for family in ("binding", "geometry"):
        subpanel = copy.deepcopy(panel)
        subpanel["targets"] = [r for r in panel["targets"] if family in r["tags"]]
        subpanel["metadata"]["operationalization_family"] = family
        if family == "binding" and complete_sets:
            audit_outputs = {}
            for target in subpanel["targets"]:
                run = target["run_name"]
                checkpoint_name = target.get("reuse_checkpoint_from") or run
                audit_dir = output_dir / "audits" / run
                audit_outputs[run] = str(audit_dir)
                command = [sys.executable, "scripts/run_matched_audit.py", "--data",
                           f"data/processed/{benchmark}.jsonl", "--checkpoint",
                           str(root/"runs/trained_target_families"/checkpoint_name),
                           "--output-dir", str(audit_dir), "--layer-index", "-1",
                           "--seed", str(target["seed"]), "--family", "binding"]
                if target.get("decoder_cross_attention_scale") is not None:
                    command += ["--decoder-cross-attention-scale", str(target["decoder_cross_attention_scale"])]
                commands.append(command)
            subpanel["metadata"]["audit_outputs"] = audit_outputs
        subpath = output_dir / f"{family}_panel.json"
        write_json(subpath, subpanel)
        if family == "geometry" or not complete_sets:
            commands.append([sys.executable, "scripts/run/materialize_target_runs.py", "--panel-config", str(subpath)])
        commands.append([sys.executable, "scripts/run_release_analysis.py", "collect-evidence", "--panel", str(subpath),
                         "--output", str(output_dir/f"{family}_evidence.json")])
    result = {"status": "plan_only", "seed_provenance": "Prespecified seed list and disjoint seed+1000 calibration controls.",
              "seeds": seeds, "model": model, "benchmark":benchmark,
              "audit_layout":"seven_item_matched_sets" if complete_sets else "exact_contrast_pairs",
              "commands": commands,
              "next_step": "Aggregate generated evidence with collect-evidence; closure additionally needs frozen object reruns."}
    write_json(output_dir / "plan.json", result)
    return result


def execute_plan(plan: dict, root: Path) -> dict:
    logs = []
    for command in plan["commands"]:
        if not isinstance(command, list) or not command or not all(isinstance(x,str) for x in command):
            raise ValueError("Commands must be argv lists; shell commands are not accepted")
        completed = subprocess.run(command, cwd=root, check=True)
        logs.append({"argv": command, "returncode": completed.returncode})
    return {**plan, "status": "executed", "execution": logs}


def collect_evidence(root: Path, panel_path: Path, cfg: dict | None = None) -> dict:
    """Collect measured audit outputs in the common evidence and decision schema."""
    from hypotheses.audit import derive_phase11_binding_audit_signals, derive_phase11_geometry_audit_signals
    panel = load_json(panel_path)
    rows = []
    for target in panel["targets"]:
        prefix = target.get("phase11_prefix", target["run_name"])
        audit_dir = panel.get("metadata",{}).get("audit_outputs",{}).get(target["run_name"])
        if audit_dir:
            audit_path = Path(audit_dir) / "summary.json"
            if not audit_path.exists():
                rows.append({"id":prefix,"status":"missing_inputs","missing":[str(audit_path)]})
                continue
            audit = load_json(audit_path)
            signals = audit.get("signals",audit.get("summary",{}).get("signals"))
            if not isinstance(signals,dict):
                raise ValueError(f"Matched audit summary lacks signals: {audit_path}")
            role = "calibration" if "calibration" in target.get("tags",[]) else "target"
            expected = {"causal":"Causal-use","present_only":"Present-only","unsupported":"Unsupported"}.get(target.get("panel_regime"))
            rows.append({"id":prefix,"family":"binding","seed":target["seed"],"signals":signals,
                         "panel_role":role,"expected_state":expected,
                         "provenance":{"summary":str(audit_path),"per_set":str(Path(audit_dir)/"L1_per_set.jsonl")},
                         "decision":adjudicate(signals,cfg)})
            continue
        paths = {"behavior": root/"reports/behavior_runs"/(target["run_name"]+"_summary.json"),
                 "discovery": root/"reports/discovery_runs"/(prefix+"_summary.json"),
                 "protocol": root/"reports/protocol_runs"/(prefix+"_summary.json")}
        missing = [str(p.relative_to(root)) for p in paths.values() if not p.exists()]
        if missing:
            rows.append({"id": prefix, "status": "missing_inputs", "missing": missing})
            continue
        values = {k: load_json(p) for k,p in paths.items()}
        family = "geometry" if "geometry" in target.get("tags",[]) else "binding"
        derive = derive_phase11_geometry_audit_signals if family == "geometry" else derive_phase11_binding_audit_signals
        linkage_path = root/"artifacts/protocol_runs"/(prefix+"_m5_generation_item_linkage.jsonl")
        linkage = read_rows(linkage_path) if linkage_path.exists() and linkage_path.stat().st_size else []
        signals = derive(behavior_summary=values["behavior"], discovery_summary=values["discovery"],
                         protocol_summary=values["protocol"], linkage_records=linkage)
        from analysis.state_analysis import summarize_phase11_realism
        raw = {}
        for slot,suffix in (("m2","_m2_generation_interchange.jsonl"),("m4","_m4_generation_rescue.jsonl")):
            path = root/"artifacts/protocol_runs"/(prefix+suffix)
            raw[slot] = read_rows(path) if path.exists() and path.stat().st_size else []
        signals.update(summarize_phase11_realism(m2_records=raw["m2"],m4_records=raw["m4"]))
        if linkage:
            # Implementation definition: intended item-role entries with either
            # selective necessity or repair advantage strictly above zero.
            flags = []
            for record in linkage:
                effect = record.get("rescue_vs_strongest_control_advantage_clause_f1")
                necessity = record.get("role_vs_strongest_control_gap_clause_f1")
                observed = [v for v in (effect,necessity) if _number(v)]
                if observed:
                    flags.append(any(v > 0 for v in observed))
            if flags:
                signals["linkage_pass_rate"] = mean(flags)
        # Coverage is the fraction of explicitly measured numeric required slots.
        numeric_slots = [k for k in REQUIRED if k not in ("audit_ready","state_anchor_ready","evidence_coverage")]
        signals["evidence_coverage"] = mean(_number(signals.get(k)) for k in numeric_slots)
        role = "calibration" if "calibration" in target.get("tags",[]) else "target"
        expected = {"causal":"Causal-use","present_only":"Present-only","unsupported":"Unsupported"}.get(target.get("panel_regime"))
        rows.append({"id": prefix, "family": family, "seed": target["seed"], "signals": signals,
                     "panel_role": role, "expected_state": expected,
                     "provenance": {k:str(p.relative_to(root)) for k,p in paths.items()},
                     "decision": adjudicate(signals,cfg)})
    return {"status": "executed_on_available_L1_summaries", "rows": rows,
            "calibration":[r for r in rows if r.get("panel_role") == "calibration"],
            "targets":[r for r in rows if r.get("panel_role") == "target"],
            "note": "Coverage and item-role linkage are calculated from measured operation records."}


def slog_adapt(source: list[dict], selection: dict, *, expected_pairs: int = 96) -> dict:
    """Validate prespecified exact contrasts without substituting synthetic data.

    Input is normalized source SLOG rows with item_id,input_text,target_form,
    split,role_signature. Selection pins the source file hash/revision and IDs.
    Roles must be extracted from each original logical form by its adapter.
    """
    if not selection.get("source_revision") or not selection.get("source_sha256") or not selection.get("selection_rationale"):
        raise ValueError("Pin SLOG revision, source SHA256 and prespecified structural selection rationale")
    from data.canonicalize import infer_role_signature_from_target_form
    by_id = {r["item_id"]: r for r in source}
    if len(by_id) != len(source):
        raise ValueError("Source SLOG IDs must be unique")
    pairs = selection["pairs"]
    if len(pairs) != expected_pairs or len({p["pair_id"] for p in pairs}) != expected_pairs:
        raise ValueError(f"Require exactly {expected_pairs} unique selected pairs")
    endpoint_pairs = {frozenset((p["source_id"],p["target_id"])) for p in pairs}
    if len(endpoint_pairs) != len(pairs):
        raise ValueError("Selected pairs must have unique endpoints; reversed pairs are duplicates")
    output = []
    for pair in pairs:
        left, right = by_id[pair["source_id"]], by_id[pair["target_id"]]
        for record in (left,right):
            if not record.get("input_text") or not record.get("target_form") or not record.get("split"):
                raise ValueError("Incomplete normalized SLOG source row")
            if not all(record.get("role_signature",{}).get(k) for k in ("predicate","agent","theme")):
                raise ValueError("Exact-contrast adapter requires parsed predicate/agent/theme roles")
            parsed = infer_role_signature_from_target_form(record["target_form"])
            if any(parsed.get(k) != record["role_signature"][k] for k in ("predicate","agent","theme")):
                raise ValueError("Role metadata must match roles parsed from the original logical form")
            if record["split"] == "train":
                raise ValueError("Selected evaluation pairs must not include original training records")
        if left["item_id"] == right["item_id"] or left["input_text"] == right["input_text"]:
            raise ValueError("Pair must contain distinct original SLOG inputs")
        delta = {k for k in ("predicate","agent","theme") if left["role_signature"][k] != right["role_signature"][k]}
        declared = set(pair["changed_roles"])
        if delta != declared or len(delta) > 1:
            raise ValueError("Pair must have the declared exact single-role or surface-only contrast")
        if not delta and left["target_form"] != right["target_form"]:
            raise ValueError("Surface-only pairs require identical canonical logical forms")
        if delta:
            role = next(iter(delta))
            a,b = left["role_signature"][role],right["role_signature"][role]
            def abstract(text: str, filler: str) -> str:
                pattern = r"(?<!\w)"+re.escape(filler)+r"(?!\w)"
                value,count = re.subn(pattern,"__changed_filler__",text,flags=re.I)
                if count != 1:
                    raise ValueError("Single-role adapter requires one unambiguous filler occurrence in each input and LF")
                return " ".join(value.casefold().split())
            if any(abstract(left[k],a) != abstract(right[k],b) for k in ("input_text","target_form")):
                raise ValueError("Single-role contrast changes additional surface or logical-form content")
        output.append({"pair_id": pair["pair_id"], "source": left, "target": right,
                       "changed_roles": sorted(delta), "provenance": selection["source_revision"]})
    return {"status": "validated_selected_source_pairs", "experiment_context": "selected_slog_subdomain",
            "source_revision": selection["source_revision"], "selection_rationale": selection["selection_rationale"],
            "pairs": output}
