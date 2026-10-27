# BOUNDED-RS controlled semantic-parsing testbed

The primary testbed is `data/processed/phase11_systematicity_v2.jsonl`: **672 items in 96 complete matched sets**, already present in the original public repository. The camera-ready release regenerates those same rows from the original generator and publishes an equivalent flat CSV, explicit pairs, schema records, checksums and Croissant metadata. It does not replace the primary experiment with the earlier COGS stage or the 480-item `v1` stress test.

Creators and maintainers: Boyu Cao and Wenshuo Wang. Code and this original synthetic dataset are released under the repository's MIT license. Public hosting: [GitHub repository](https://github.com/VinceWangVW/An-Evaluation-Protocol-for-Representational-Systematicity). The data require no account, authentication or license acceptance gate. T5 and external benchmarks retain their own licenses and are not included in this data release.

## Construction and labels

`src/data/systematicity_schema.py` uses `random.Random(13)` to sample a main predicate, context predicate, four distinct synthetic name fillers, two distinct modifiers, a context noun, voice and relative-clause construction. The fixed vocabulary has 5 main predicates, 3 context predicates, 16 name tokens, 10 adjectives and 6 context nouns. Each design yields:

1. `rescue_donor`: active, non-embedded easy source;
2. `rescue_target`: the same gold meaning in the family's construction;
3. `role_swap`: agent and theme, with their modifiers, exchange roles;
4. `agent_swap`: only the agent filler changes;
5. `theme_swap`: only the theme filler changes;
6. `filler_swap`: both fillers change;
7. `modifier_swap`: modifier attachments exchange.

Each item has three deterministic English template candidates. Rule-based validation checks filler/modifier/predicate presence, active or passive order, required relative-clause context and absence of pronouns. All **2,016 candidates** pass; shortest-valid selection produces one canonical realization per item. Gold logical forms are constructed directly from the design, for example `help(agent=freya,theme=kira,agent_mod=brisk,theme_mod=careful)`. There was no human annotation, demographic sampling or language-model generation for this canonical testbed. The context relative clause serves as a surface contrast and is not added to the main-event logical form.

| Hard-target construction | Matched sets | Items |
| --- | ---: | ---: |
| Active, no relative clause | 28 | 196 |
| Passive, no relative clause | 31 | 217 |
| Active, relative clause | 31 | 217 |
| Passive, relative clause | 6 | 42 |

Family construction labels describe the hard target. Every easy source remains active and non-embedded. All 96 source–target pairs have identical gold meaning, and **68** have different surface forms. Agent and theme replacements supply **192 interchange pairs**; there are **96 repair pairs**. Active non-embedded easy-source/target duplicates are intentional controls, not accidental duplicate records.

## Splits and cluster identity

Families are assigned as 64 training, 16 development and 16 generalization families, in that generator order. Training includes 22 active non-embedded, 21 passive non-embedded and 21 active embedded sets. It contains no passive embedded item. The six passive embedded sets are held out. For generalization families, only the six non-source items are scored in `gen`; their easy sources belong to `dev`.

| Item split | Count |
| --- | ---: |
| `train` | 448 |
| `dev` | 128 |
| `gen` | 96 |

Use `metadata.counterpart_family_id` in JSONL or `matched_set_id` in CSV as the cluster key. Train families do not overlap held-out families. The intentional movement of generalization easy sources into `dev` means dev and gen share those family identities: do not mistake this for an independent dev/gen cluster split or resample individual rows when estimating matched-set uncertainty.

`data/release/matched_sets.jsonl` lists every original family ID, all seven item IDs and the intervention/repair pairs. Its `release_index_id` values `GEN-000` through `GEN-095` are **new zero-based aliases in generator order**, added only for navigation. The historical mapping of the manuscript's `MS-047` is unavailable. The new alias `GEN-047` is the Caleb/Opal/contact family, whereas the review's chef/poet/praise example is outside this generator's vocabulary. This release does not assert that those are the same example or attach the manuscript's measured scores to the regenerated rows.

## Files and reproducibility

- `data/processed/phase11_systematicity_v2.jsonl`: canonical rows consumed by existing runners, including split, text, logical form and metadata.
- `data/release/items.csv`: equivalent flat records for standard tooling and the Croissant reader.
- `data/release/schema.jsonl`: exact semantic designs behind each item.
- `data/release/matched_sets.jsonl`: exact counterpart membership and pairs.
- `data/release/validation.json`: candidate validation, counts and pair coverage.
- `data/release/manifest.json`: byte sizes, SHA-256 hashes, seed, version and provenance.
- `croissant.json`: public download URLs, file hashes, column sources, creator/license information and minimal Responsible AI metadata.

From the repository root, using Python 3.11 or later and no ML dependencies:

```bash
python3 scripts/build_release_data.py --seed 13
python3 scripts/build_release_data.py --seed 13 --check
python3 scripts/validate_release_data.py
```

To regenerate without touching the checked-in files, supply `--output-root /tmp/bounded-rs-data`. To validate and actually load all Croissant records after installing `mlcroissant`, run:

```bash
python3 scripts/validate_release_data.py --mlcroissant
```

The metadata uses Croissant 1.0 and includes the required core fields plus limitations, biases, personal/sensitive information, intended uses, social impact, synthetic-data status and generator provenance. Public raw URLs point to the repository's `main` branch; hashes detect later changes. For an immutable experiment record, retain the Git commit and manifest together. Dataset version is 1.0.0; changes to canonical rows require a new version, regenerated hashes and updated metadata. Report data issues through the repository's issue tracker.

## Intended use, limitations and social impact

The dataset supports controlled behavioral contrasts, exact signal interchange/repair implementation, and bounded audits of declared representation families. It is not a general language benchmark or a ranking of model intelligence. Validation concerns the generated templates and counterparts; it does not establish that any model exhibits a causal mechanism.

The finite English templates and discrete role–filler logical forms privilege exact role manipulation and may favor the tested Binding operations. Names are synthetic symbols; there is no represented human population, demographic balance or coverage claim. Nonlinear, higher-order, polysemantic or diffuse geometric mechanisms, naturalistic language and cross-architecture generality are outside the dataset's demonstrated scope. Causal-use claims require evidence operations and validity gates beyond dataset accuracy. Model evidence and historical aggregate findings must be tracked separately from deterministic data regeneration.

The data contain no personal or sensitive information and no human-subject records. More transparent mechanism claims are a potential benefit; overgeneralizing a finite testbed or unvalidated detector is a principal misuse risk. The dataset is not validated for safety certification, fairness auditing or consequential decisions about people. Keeping the model, task, declared operations and closure basis explicit mitigates that risk.
