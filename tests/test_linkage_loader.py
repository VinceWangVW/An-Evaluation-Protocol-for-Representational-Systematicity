from hypotheses.audit import derive_generation_audit_signals


def test_current_linkage_key_is_used_and_legacy_key_still_supported():
    behavior = {"evaluation": {"overall": {}, "by_split": {
        "dev": {"exact_match": 1.0, "clause_f1": 1.0},
        "gen": {"exact_match": 1.0, "clause_f1": 1.0},
    }}}
    discovery = {"layers": {"6": {}}, "candidate_counts": {
        "selected": 6, "by_stratum": {"pass": 1, "fail": 4, "control": 1}},
        "matched_set_count": 1}
    protocol = {"layer_index": 6, "modules": {}}
    linkage = [
        {"item_id": "new", "stratum": "fail", "rescue_vs_strongest_control_advantage_clause_f1": 0.2},
        {"item_id": "legacy", "stratum": "fail", "rescue_advantage_clause_f1": 0.1},
        # An explicitly observed new zero must not fall back to an old positive.
        {"item_id": "zero", "stratum": "fail", "rescue_vs_strongest_control_advantage_clause_f1": 0.0,
         "rescue_advantage_clause_f1": 1.0},
        {"item_id": "missing", "stratum": "fail"},
    ]
    signals = derive_generation_audit_signals(behavior, discovery, protocol, linkage)
    assert signals["m5_fail_item_rescue_advantage_coverage"] == 0.5
