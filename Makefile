PYTHON ?= python3

FAMILY_CLOSURE_CONFIG_CONTROLS := configs/experiments/family_closure_controls.json
FAMILY_CLOSURE_CONFIG_H1 := configs/experiments/family_closure_h1like.json
DETECTOR_TRUTHFULNESS_CONFIG := configs/experiments/detector_truthfulness.json
DETECTOR_COMPETITION_CONFIG := configs/experiments/detector_competition.json

.PHONY: install smoke test family-closure-observability family-closure-catalogs family-closure-h2 family-closure-h1 family-closure-finalize detector-truthfulness-build detector-truthfulness-train detector-truthfulness-run detector-competition-build detector-competition-train detector-competition-run

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

smoke:
	$(PYTHON) -m pytest -q \
		tests/test_family_closure.py \
		tests/test_detector_truthfulness.py \
		tests/test_detector_ontology_competition.py

test:
	$(PYTHON) -m pytest -q

family-closure-observability:
	$(PYTHON) scripts/run/run_observability_audit.py --config $(FAMILY_CLOSURE_CONFIG_CONTROLS)

family-closure-catalogs:
	$(PYTHON) scripts/run/build_family_closure_catalogs.py --config $(FAMILY_CLOSURE_CONFIG_CONTROLS)

family-closure-h2:
	$(PYTHON) scripts/run/gate_family_h2.py --config $(FAMILY_CLOSURE_CONFIG_CONTROLS)

family-closure-h1:
	$(PYTHON) scripts/run/gate_direct_h1.py --config $(FAMILY_CLOSURE_CONFIG_H1)

family-closure-finalize:
	$(PYTHON) scripts/run/run_family_closure.py

detector-truthfulness-build:
	$(PYTHON) scripts/run/build_truthfulness_benchmarks.py --config $(DETECTOR_TRUTHFULNESS_CONFIG)

detector-truthfulness-train:
	$(PYTHON) scripts/run/train_truthfulness_families.py --config $(DETECTOR_TRUTHFULNESS_CONFIG)

detector-truthfulness-run:
	$(PYTHON) scripts/run/run_detector_truthfulness.py --config $(DETECTOR_TRUTHFULNESS_CONFIG)

detector-competition-build:
	$(PYTHON) scripts/run/build_detector_competition_benchmarks.py --config $(DETECTOR_COMPETITION_CONFIG)

detector-competition-train:
	$(PYTHON) scripts/run/train_detector_competition.py --config $(DETECTOR_COMPETITION_CONFIG)

detector-competition-run:
	$(PYTHON) scripts/run/run_detector_competition.py --config $(DETECTOR_COMPETITION_CONFIG)
