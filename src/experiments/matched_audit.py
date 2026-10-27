"""Measure the Binding evidence chain on complete seven-item matched sets.

Interchange compares role replacement against wrong-role, predicate and shuffled
donors. Necessity compares a role ablation against wrong-role and non-role
ablations; global ablation is recorded separately. Repair patches
both easy-source role vectors into every hard target, with the same controls,
self/path-only and generic decoder-gain controls. Correct baselines are included
in the repair panel; error-conditional repair eligibility is a separate count.

The readout is leave-one-item-out nearest-centroid prediction. Its surface,
filler and predicate controls use the same role-span observations. Vector-local
realism compares a replacement's distance to an independent natural role bank
against the target's distance, with a prespecified cosine margin. Linkage checks
the intended changed role and preservation of the other role on each item.

Panel decision features are equal-weight means of matched-set estimates, the
same estimator used by cluster resampling. The independent held-out behavioral
anchor gates the audit; pooled readouts remain descriptive diagnostics.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
from statistics import mean
from typing import Any

from discovery.spans import build_probe_targets
from release_analysis import DEFAULT_PROFILE, adjudicate, paired_sign_p_value

TYPES = ("rescue_donor", "rescue_target", "agent_swap", "theme_swap", "role_swap", "filler_swap", "modifier_swap")
ROLES = ("agent", "theme")


def _cosine_distance(left: list[float], right: list[float]) -> float:
    a = math.sqrt(sum(x*x for x in left)); b = math.sqrt(sum(x*x for x in right))
    if not a or not b:
        return 1.0
    return 1 - sum(x*y for x,y in zip(left,right))/(a*b)


def _parsed(text: str) -> dict[str, str] | None:
    match = re.fullmatch(r"\s*([\w.-]+)\s*\((.*)\)\s*", text)
    if not match:
        return None
    result = {"predicate": match[1].lower()}
    for part in match[2].split(","):
        if "=" not in part:
            return None
        key,value = part.split("=",1)
        if key.strip() in result:
            return None
        result[key.strip()] = value.strip().lower()
    return result


def _score(prediction: str, gold: str) -> dict[str, Any]:
    """Clause-F1 on predicate/argument/attachment atoms of functional LFs."""
    expected = _parsed(gold)
    if expected is None:
        raise ValueError("Matched audit requires functional logical forms with named role arguments")
    observed = _parsed(prediction)
    truth = set(expected.items()); guesses = set((observed or {}).items())
    overlap = len(truth & guesses)
    f1 = 2*overlap/(len(truth)+len(guesses)) if truth or guesses else 0.0
    return {"exact_match": prediction.strip() == gold.strip(), "clause_f1": f1,
            "semantic_exact_match": observed == expected, "parsed": observed}


def _centroid_readout(observations: list[dict], label: str) -> dict:
    if not observations:
        return {"accuracy":None,"count":0,"predictions":[]}
    width=len(observations[0]["vector"])
    totals=defaultdict(lambda:[0.]*width); counts=Counter()
    item_totals=defaultdict(lambda:[0.]*width); item_counts=Counter()
    for obs in observations:
        key=obs[label]; item_key=(obs["item_id"],key)
        counts[key]+=1; item_counts[item_key]+=1
        for index,value in enumerate(obs["vector"]):
            totals[key][index]+=value; item_totals[item_key][index]+=value
    predictions = []
    for obs in observations:
        # A whole input is held out, rather than its duplicate role/filler probes.
        centers={}
        for key,vector_sum in totals.items():
            item_key=(obs["item_id"],key)
            count=counts[key]-item_counts[item_key]
            if count:
                excluded=item_totals.get(item_key,[0.]*width)
                centers[key]=[(value-heldout)/count for value,heldout in zip(vector_sum,excluded)]
        if len(centers) < 2 or obs[label] not in centers:
            continue
        predicted = min(sorted(centers), key=lambda key:_cosine_distance(obs["vector"],centers[key]))
        predictions.append({"item_id":obs["item_id"],"role":obs["role"],"expected":obs[label],
                            "predicted":predicted,"correct":predicted == obs[label]})
    return {"accuracy":mean(row["correct"] for row in predictions) if predictions else None,
            "count":len(predictions),"predictions":predictions}


def _group_rows(rows: list[dict]) -> dict[str, dict[str, dict]]:
    grouped = defaultdict(dict)
    ids = set()
    for row in rows:
        for key in ("item_id","input_text","target_form","role_signature","metadata"):
            if key not in row:
                raise ValueError(f"Matched audit row lacks {key}")
        if row["item_id"] in ids:
            raise ValueError("Item IDs must be unique")
        ids.add(row["item_id"])
        family = row["metadata"]["counterpart_family_id"]
        kind = row["metadata"]["counterpart_type"]
        if kind in grouped[family]:
            raise ValueError("Duplicate counterpart type in matched set")
        grouped[family][kind] = row
        if _parsed(row["target_form"]) is None:
            raise ValueError("Expected a named-argument functional logical form")
        gold_roles = _parsed(row["target_form"])
        if any(str(row["role_signature"].get(role,"")).lower() != gold_roles.get(role)
               for role in ("predicate",*ROLES)):
            raise ValueError("Role signature must match the declared gold logical form")
    if not grouped or any(set(members) != set(TYPES) for members in grouped.values()):
        raise ValueError("Each matched set must contain all seven counterpart types")
    for family,members in grouped.items():
        if members["rescue_donor"]["target_form"] != members["rescue_target"]["target_form"]:
            raise ValueError(f"Repair source/target meaning differs in {family}")
        original = _parsed(members["rescue_target"]["target_form"])
        for role,kind in (("agent","agent_swap"),("theme","theme_swap")):
            changed = _parsed(members[kind]["target_form"])
            if {key for key in set(original)|set(changed) if original.get(key) != changed.get(key)} != {role}:
                raise ValueError(f"{kind} must change exactly its declared role")
    return dict(grouped)


def audit_matched_sets(rows: list[dict], adapter: Any, *, layer_index: int = -1,
                       max_new_tokens: int = 192, seed: int = 13, permutations: int = 199,
                       profile: dict | None = None, realism_margin: float = 0.05) -> dict:
    """Run model measurements; the adapter may be injected for offline tests."""
    if permutations < 19 or realism_margin < 0:
        raise ValueError("Use at least 19 permutations and a nonnegative realism margin")
    grouped = _group_rows(rows)
    rng = random.Random(seed)
    traces = {}; baselines = {}; observations = []; operations = []; token_pools = {}
    for family,members in grouped.items():
        for kind in TYPES:
            row = members[kind]
            probes = build_probe_targets(row)
            trace = adapter.trace_text(row["input_text"],probes=probes,layer_indices=[layer_index])
            by_key = {}
            for obs in trace["span_vectors"]:
                if obs["namespace"] == "semantic_role":
                    by_key[obs["label"]] = obs
                elif obs["namespace"] == "predicate_identity":
                    by_key["predicate"] = obs
            if not all(role in by_key for role in (*ROLES,"predicate")):
                raise ValueError(f"Unaligned role/predicate span: {row['item_id']}")
            dimensions = {len(obs["vector"]) for obs in by_key.values()}
            if len(dimensions) != 1 or not next(iter(dimensions)) or any(
                not obs.get("token_indices") or not all(isinstance(value,(int,float)) and
                    not isinstance(value,bool) and math.isfinite(value) for value in obs["vector"])
                for obs in by_key.values()
            ):
                raise ValueError("Natural span vectors need aligned tokens and equal finite dimensions")
            role_order = sorted(ROLES,key=lambda role:by_key[role]["start_char"])
            for role in ROLES:
                observations.append({"item_id":row["item_id"],"matched_set_id":family,
                    "role":role,"surface":"subject" if role == role_order[0] else "object",
                    "filler":str(row["role_signature"][role]).lower(),
                    "predicate":str(row["role_signature"]["predicate"]).lower(),
                    "vector":by_key[role]["vector"]})
            traces[row["item_id"]] = by_key
            special_tokens = {"<pad>","</s>","<s>","[PAD]","[SEP]","[CLS]"}
            token_pools[row["item_id"]] = [index for index,token in enumerate(trace.get("tokens",[]))
                                           if token not in special_tokens]
            if not token_pools[row["item_id"]]:
                token_pools[row["item_id"]] = sorted({index for obs in by_key.values() for index in obs["token_indices"]})
            predicted = adapter.generate_text(row["input_text"],max_new_tokens=max_new_tokens)
            baselines[row["item_id"]] = {"generated_text":predicted,**_score(predicted,row["target_form"])}

    anchor_rows = [row for row in rows if row.get("split") == "gen"]
    anchor_split = "gen"
    if not anchor_rows:
        anchor_rows = [row for row in rows if row.get("split") in {"dev","test"}]
        anchor_split = "held_out"
    if not anchor_rows:
        anchor_rows = rows
        anchor_split = "declared_panel"
    anchor_boundary = (profile or DEFAULT_PROFILE)["features"]["b0_gen_clause_f1"]["boundary"]
    anchor_f1 = mean(baselines[row["item_id"]]["clause_f1"] for row in anchor_rows)
    anchor = {"split":anchor_split,"item_count":len(anchor_rows),"clause_f1":anchor_f1,
        "exact_match":mean(baselines[row["item_id"]]["exact_match"] for row in anchor_rows),
        "boundary":anchor_boundary,"comparison":">","passed":anchor_f1 > anchor_boundary}

    def generate(row: dict, patches: list[dict] | None = None, ablations: list[dict] | None = None,
                 overrides: dict | None = None, gold: str | None = None) -> dict:
        output = adapter.generate_with_intervention(row["input_text"],layer_index=layer_index,
            patch_specs=patches,ablation_specs=ablations,control_overrides=overrides,
            max_new_tokens=max_new_tokens)
        text = output["generated_text"]
        return {"generated_text":text,**_score(text,gold or row["target_form"])}

    def patch(target: dict, role: str, vector: list[float]) -> dict:
        return {"token_indices":traces[target["item_id"]][role]["token_indices"],"replacement_vector":vector}

    def cardinality_matched(primary: list[int], pool: list[int], forbidden: list[int], count: int) -> list[int]:
        indices = list(dict.fromkeys(index for index in [*primary,*pool] if index not in forbidden))[:count]
        if len(indices) != count:
            raise ValueError("Natural input lacks enough distinct tokens for a cardinality-matched ablation")
        return indices

    def realism(target: dict, role: str, replacement: list[float], donor_id: str) -> dict:
        baseline = traces[target["item_id"]][role]["vector"]
        bank = [o["vector"] for o in observations if o["role"] == role and
                o["item_id"] not in {target["item_id"],donor_id}]
        original_distance = min(_cosine_distance(baseline,v) for v in bank)
        replacement_distance = min(_cosine_distance(replacement,v) for v in bank)
        return {"natural_bank_size":len(bank),"target_distance":original_distance,
                "replacement_distance":replacement_distance,"margin":realism_margin,
                "passed":replacement_distance <= original_distance+realism_margin}

    per_set = []
    for family,members in grouped.items():
        target = members["rescue_target"]; target_traces = traces[target["item_id"]]
        family_operations = []
        # Two exact single-role replacements per set.
        for role,kind in (("agent","agent_swap"),("theme","theme_swap")):
            donor = members[kind]; donor_trace = traces[donor["item_id"]]
            other_role = "theme" if role == "agent" else "agent"
            expected_filler = str(donor["role_signature"][role]).lower()
            shuffled_candidates = [row for row in rows if row["item_id"] not in {target["item_id"],donor["item_id"]}
                and str(row["role_signature"][role]).lower() != expected_filler]
            independent = [row for row in shuffled_candidates if row["metadata"]["counterpart_family_id"] != family]
            shuffled_candidates = sorted(independent or shuffled_candidates,key=lambda row:row["item_id"])
            shuffle_seed = int(hashlib.sha256(f"{seed}:{family}:{role}".encode()).hexdigest()[:16],16)
            shuffled = random.Random(shuffle_seed).choice(shuffled_candidates)
            donor_vectors = {"actual":donor_trace[role]["vector"],
                "wrong_role":donor_trace[other_role]["vector"],
                "non_role":donor_trace["predicate"]["vector"],
                "shuffled":traces[shuffled["item_id"]][role]["vector"]}
            patches = {name:[patch(target,role,vector)] for name,vector in donor_vectors.items()}
            outcomes = {name:generate(target,specs,gold=donor["target_form"])
                        for name,specs in patches.items()}
            strongest = max(outcomes[name]["clause_f1"] for name in donor_vectors if name != "actual")
            actual = outcomes["actual"]; expected = _parsed(donor["target_form"])
            observed = actual["parsed"] or {}
            linked = observed.get(role) == expected[role] and observed.get(other_role) == expected[other_role]
            record = {"matched_set_id":family,"operation":"interchange","target_item_id":target["item_id"],
                "donor_item_id":donor["item_id"],"role":role,"expected_target_form":donor["target_form"],
                "shuffled_donor_item_id":shuffled["item_id"],
                "shuffled_donor_filler":str(shuffled["role_signature"][role]).lower(),
                "outcomes":outcomes,"patches":patches,"strongest_control":strongest,"advantage":actual["clause_f1"]-strongest,
                "linked":linked,"realism":realism(target,role,donor_vectors["actual"],donor["item_id"])}
            family_operations.append(record)
        # Every item and role is measured, independent of baseline correctness.
        for kind in TYPES:
            row = members[kind]; spans = traces[row["item_id"]]; baseline = baselines[row["item_id"]]
            global_tokens = sorted({i for key in (*ROLES,"predicate") for i in spans[key]["token_indices"]})
            global_output = generate(row,ablations=[{"token_indices":global_tokens}])
            for role in ROLES:
                other_role = "theme" if role == "agent" else "agent"
                actual_tokens = spans[role]["token_indices"]
                count = len(actual_tokens)
                wrong_tokens = cardinality_matched(spans[other_role]["token_indices"],token_pools[row["item_id"]],actual_tokens,count)
                non_role_tokens = cardinality_matched(spans["predicate"]["token_indices"],token_pools[row["item_id"]],
                    actual_tokens+spans[other_role]["token_indices"],count)
                outcomes = {"actual":generate(row,ablations=[{"token_indices":actual_tokens}]),
                    "wrong_role":generate(row,ablations=[{"token_indices":wrong_tokens}]),
                    "non_role":generate(row,ablations=[{"token_indices":non_role_tokens}]),
                    "global":global_output}
                drops = {name:baseline["clause_f1"]-value["clause_f1"] for name,value in outcomes.items()}
                strongest = max(drops["wrong_role"],drops["non_role"])
                family_operations.append({"matched_set_id":family,"operation":"necessity",
                    "target_item_id":row["item_id"],"role":role,"baseline":baseline,
                    "ablation_token_indices":{"actual":actual_tokens,"wrong_role":wrong_tokens,
                        "non_role":non_role_tokens,"global":global_tokens},"matched_token_count":count,
                    "outcomes":outcomes,"drops":drops,"strongest_control":strongest,
                    "advantage":drops["actual"]-strongest})
        source = members["rescue_donor"]; source_spans = traces[source["item_id"]]
        replacement = {role:source_spans[role]["vector"] for role in ROLES}
        variants = {
            "actual":[patch(target,role,replacement[role]) for role in ROLES],
            "wrong_role":[patch(target,role,replacement["theme" if role == "agent" else "agent"]) for role in ROLES],
            "non_role":[patch(target,role,source_spans["predicate"]["vector"]) for role in ROLES],
            "shuffled":[patch(target,role,traces[members["filler_swap"]["item_id"]][role]["vector"]) for role in ROLES],
            "path_only":[patch(target,role,target_traces[role]["vector"]) for role in ROLES],
        }
        repair_outcomes = {name:generate(target,patches) for name,patches in variants.items()}
        repair_outcomes["generic_decoder_gain"] = generate(target,overrides={"decoder_cross_attention_scale":1.75})
        baseline = baselines[target["item_id"]]
        gains = {name:outcome["clause_f1"]-baseline["clause_f1"] for name,outcome in repair_outcomes.items()}
        strongest = max(gain for name,gain in gains.items() if name != "actual")
        family_operations.append({"matched_set_id":family,"operation":"repair",
            "target_item_id":target["item_id"],"donor_item_id":source["item_id"],"baseline":baseline,
            "eligible_error":not baseline["semantic_exact_match"],"outcomes":repair_outcomes,
            "patches":variants,"control_overrides":{"generic_decoder_gain":{"decoder_cross_attention_scale":1.75}},
            "gains":gains,"strongest_control":strongest,"advantage":gains["actual"]-strongest,
            "realism":[realism(target,role,replacement[role],source["item_id"]) for role in ROLES]})
        local_obs = [o for o in observations if o["matched_set_id"] == family]
        readouts = {label:_centroid_readout(local_obs,label) for label in ("role","surface","filler","predicate")}
        m2 = [o for o in family_operations if o["operation"] == "interchange"]
        m3 = [o for o in family_operations if o["operation"] == "necessity"]
        m4 = [o for o in family_operations if o["operation"] == "repair"]
        flags = [o["realism"]["passed"] for o in m2]+[r["passed"] for o in m4 for r in o["realism"]]
        role_accuracy = readouts["role"]["accuracy"]; surface_accuracy = readouts["surface"]["accuracy"]
        signals = {"b0_gen_clause_f1":mean(baselines[row["item_id"]]["clause_f1"] for row in members.values()),
            "m1_semantic_role_accuracy":role_accuracy,
            "m1_surface_control_gap":role_accuracy-surface_accuracy if role_accuracy is not None and surface_accuracy is not None else None,
            "m1_filler_control_accuracy":readouts["filler"]["accuracy"],
            "m1_predicate_control_accuracy":readouts["predicate"]["accuracy"],
            "m2_advantage_mean":mean(o["advantage"] for o in m2),
            "m2_permutation_p_value":paired_sign_p_value([o["advantage"] for o in m2],rng,permutations),
            "m3_selective_impairment_clause_f1":mean(o["advantage"] for o in m3),
            "m3_permutation_p_value":paired_sign_p_value([o["advantage"] for o in m3],rng,permutations),
            "m4_advantage_mean":mean(o["advantage"] for o in m4),
            "m4_permutation_p_value":paired_sign_p_value([o["advantage"] for o in m4],rng,permutations),
            "realism_pass_rate":mean(flags),"linkage_pass_rate":mean(o["linked"] for o in m2),
            "audit_ready":True,"state_anchor_ready":anchor["passed"]}
        # Required slots are owned by the selected scorer, including its coverage slot.
        from release_analysis import REQUIRED
        slots = [key for key in REQUIRED if key not in ("evidence_coverage","audit_ready","state_anchor_ready")]
        signals["evidence_coverage"] = mean(signals.get(key) is not None for key in slots)
        per_set.append({"matched_set_id":family,"signals":signals,"readouts":readouts,"behavioral_anchor":anchor,
            "counts":{"items":7,"interchange":len(m2),"necessity":len(m3),"repair":len(m4),
                      "eligible_repair_errors":sum(o["eligible_error"] for o in m4)},
            "decision":adjudicate(signals,profile)})
        operations.extend(family_operations)

    numeric = {key for row in per_set for key,value in row["signals"].items()
               if isinstance(value,(int,float)) and not isinstance(value,bool)}
    panel_signals = {key:mean(row["signals"][key] for row in per_set if row["signals"].get(key) is not None) for key in numeric}
    panel_signals.update(audit_ready=True,state_anchor_ready=anchor["passed"])
    # Sign flips use matched sets as the independent unit, keeping the two roles
    # and all operations within a semantic design paired.
    for metric,p_key in (("m2_advantage_mean","m2_permutation_p_value"),
                         ("m3_selective_impairment_clause_f1","m3_permutation_p_value"),
                         ("m4_advantage_mean","m4_permutation_p_value")):
        panel_signals[p_key] = paired_sign_p_value([row["signals"][metric] for row in per_set],random.Random(seed),permutations)
    panel_readouts = {label:_centroid_readout(observations,label) for label in ("role","surface","filler","predicate")}
    counts = {name:sum(row["counts"][name] for row in per_set) for name in per_set[0]["counts"]}
    summary = {"signals":panel_signals,"counts":counts,"decision":adjudicate(panel_signals,profile),
               "behavioral_anchor":anchor,"readouts":panel_readouts,
               "matched_sets":len(per_set),"items":len(rows),"baseline_exact_match":mean(value["exact_match"] for value in baselines.values()),
               "repair_error_conditional_rate":mean(o["outcomes"]["actual"]["semantic_exact_match"] for o in operations
                   if o["operation"] == "repair" and o["eligible_error"]) if counts["eligible_repair_errors"] else None}
    row_hash = hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {"status":"executed","family":"binding","checkpoint":getattr(adapter,"model_name_or_path",type(adapter).__name__),
        "config":{"layer_index":layer_index,"max_new_tokens":max_new_tokens,"seed":seed,
                  "permutations":permutations,"realism_margin":realism_margin,"input_sha256":row_hash},
        "signals":panel_signals,"summary":summary,"per_set":per_set,"operations":operations,
        "baselines":[{"item_id":key,**value} for key,value in baselines.items()],
        "span_vectors":[{"item_id":item_id,"span_key":key,**obs} for item_id,spans in traces.items() for key,obs in spans.items()]}


def write_audit(output_dir: str | Path, payload: dict) -> dict[str,str]:
    output = Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    paths = {}
    for name,rows in (("L1_per_set.jsonl",payload["per_set"]),
                      ("L2_decisions.jsonl",[{"matched_set_id":r["matched_set_id"],**r["decision"]} for r in payload["per_set"]]),
                      ("raw_operations.jsonl",payload["operations"]),("baseline_predictions.jsonl",payload["baselines"]),
                      ("span_vectors.jsonl",payload["span_vectors"])):
        path = output/name
        path.write_text("".join(json.dumps(row,allow_nan=False)+"\n" for row in rows),encoding="utf-8")
        paths[name]=str(path)
    summary = {key:value for key,value in payload.items() if key not in ("per_set","operations","baselines","span_vectors")}
    (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    paths["summary.json"]=str(output/"summary.json")
    return paths
