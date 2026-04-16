PYTHON ?= .venv/bin/python

.PHONY: install build generate-train train ask eval eval-sweep analyze status test clean-reports

install:
	$(PYTHON) -m pip install -r requirements.txt

build:
	$(PYTHON) scripts/qa_cli.py build

generate-train:
	$(PYTHON) scripts/qa_cli.py generate-train

train:
	$(PYTHON) scripts/qa_cli.py train

ask:
	$(PYTHON) scripts/qa_cli.py ask "$(Q)"

eval:
	$(PYTHON) scripts/qa_cli.py eval

eval-sweep:
	$(PYTHON) scripts/qa_cli.py eval --sweep

analyze:
	$(PYTHON) scripts/qa_cli.py analyze

status:
	$(PYTHON) scripts/qa_cli.py status

test:
	$(PYTHON) -m pytest -q

clean-reports:
	rm -f artifacts/real_qa/reports/evaluation_report.json artifacts/real_qa/reports/evaluation_report.md artifacts/real_qa/reports/error_analysis.md
