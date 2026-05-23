.PHONY: install run run-no-llm compare validate api clean

install:
	pip install -r requirements.txt

run:
	python main.py --compare-chunking

run-no-llm:
	python main.py --no-llm --compare-chunking

validate:
	python validate.py

api:
	uvicorn api:app --reload --port 8000

clean:
	rm -rf artifacts llm_calls.jsonl
