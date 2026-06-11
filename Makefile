lib_name=$$(grep -oP '(?<=^name = ").*(?=")' pyproject.toml)
version=$$(grep -oP '(?<=^version = ").*(?=")' pyproject.toml)
pypi_command="curl -s https://pypi.org/rss/project/$(lib_name)/releases.xml | grep '<title>' | head -n 2 | grep -oP '(?<=title>)\d+.\d+.\d+(?=</title)'"
# Function to load .env file if it exists
define load-env
	@if [ -f .env ]; then \
		set -a && . .env && set +a; \
		echo "Loaded environment variables from .env"; \
	fi
endef

# Function to wait until a command output changes
define wait-until-changed
	@echo "cmd: $(1)"
	$(eval VAL := $(shell $(1)))
	@echo "Value: '$(VAL)'"
	@while true; do \
		sleep 3; \
		VAL_UPD=$$($(1)); \
		if [ "$(VAL)" != "$$VAL_UPD" ]; then \
			echo "Value: changed: '$(VAL)' -> '$$VAL_UPD'"; \
			exit 0; \
		fi; \
		echo "Value: still '$(VAL)'"; \
	done
endef

clean:
	rm -rf dist .mypy_cache .pytest_cache .ruff_cache

build: clean
	uv build

install: build
	uv pip install dist/*.whl

test:
	@if [ -f .env ]; then set -a && . .env && set +a && uv run pytest; \
	else uv run pytest; fi

# Run only the @pytest.mark.live tests. Requires OPENAI_API_KEY (an
# OpenRouter-compatible key works). Skipped by default in `make test`.
test-live:
	@if [ -f .env ]; then set -a && . .env && set +a && uv run pytest -m live; \
	else uv run pytest -m live; fi

# Regenerate per-topic README.md files under tests/ and examples/.
docs-topic-index:
	@uv run python scripts/generate_topic_readmes.py

# Rebuild notebooks/02_visualizations_demo.ipynb from its Python source.
notebook-build:
	@uv run python scripts/build_visualizations_notebook.py

# Refresh the cassettes consumed by the demo notebook (requires OPENAI_API_KEY).
notebook-record:
	@if [ -f .env ]; then set -a && . .env && set +a && uv run python scripts/record_notebook_cassettes.py; \
	else uv run python scripts/record_notebook_cassettes.py; fi

# Smoke-test the demo notebook (cassette mode — no API calls).
notebook-smoke:
	@uv run python scripts/run_notebook.py

# Re-execute the demo notebook and persist outputs (rendered Mermaid + text).
notebook-smoke-save:
	@uv run python scripts/run_notebook.py --save

# Regenerate notebooks/README.md from each notebook's demo_info metadata.
notebooks-readme:
	@uv run python scripts/generate_notebooks_readme.py

# Run examples with dev version from src/
# Automatically loads .env file if present
example-basic:
	@if [ -f .env ]; then set -a && . .env && set +a && PYTHONPATH=$$(pwd) uv run python examples/orchestration/basic_chain_example.py; \
	else PYTHONPATH=$$(pwd) uv run python examples/orchestration/basic_chain_example.py; fi

example-openrouter:
	@if [ -f .env ]; then set -a && . .env && set +a && PYTHONPATH=$$(pwd) uv run python examples/llm_inference/openrouter_example.py; \
	else PYTHONPATH=$$(pwd) uv run python examples/llm_inference/openrouter_example.py; fi

example-tool:
	@if [ -f .env ]; then set -a && . .env && set +a && PYTHONPATH=$$(pwd) uv run python examples/tool_calling/tool_steps_example.py; \
	else PYTHONPATH=$$(pwd) uv run python examples/tool_calling/tool_steps_example.py; fi

example-structured:
	@if [ -f .env ]; then set -a && . .env && set +a && PYTHONPATH=$$(pwd) uv run python examples/tool_calling/structured_output_example.py; \
	else PYTHONPATH=$$(pwd) uv run python examples/tool_calling/structured_output_example.py; fi

example-council:
	@if [ -f .env ]; then set -a && . .env && set +a && PYTHONPATH=$$(pwd) uv run python examples/agents/llm_council_example.py; \
	else PYTHONPATH=$$(pwd) uv run python examples/agents/llm_council_example.py; fi

example-legacy:
	@if [ -f .env ]; then set -a && . .env && set +a && PYTHONPATH=$$(pwd) uv run python examples/legacy_mmar_llm_example.py; \
	else PYTHONPATH=$$(pwd) uv run python examples/legacy_mmar_llm_example.py; fi

example-reflection:
	@if [ -f .env ]; then set -a && . .env && set +a && PYTHONPATH=$$(pwd) uv run python examples/evaluation/reflection_example.py; \
	else PYTHONPATH=$$(pwd) uv run python examples/evaluation/reflection_example.py; fi

example-replan-deterministic:
	@if [ -f .env ]; then set -a && . .env && set +a && PYTHONPATH=$$(pwd) uv run python examples/replan/replan_deterministic_example.py; \
	else PYTHONPATH=$$(pwd) uv run python examples/replan/replan_deterministic_example.py; fi

example-replan-llm:
	@if [ -f .env ]; then set -a && . .env && set +a && PYTHONPATH=$$(pwd) uv run python examples/replan/replan_llm_checker_example.py; \
	else PYTHONPATH=$$(pwd) uv run python examples/replan/replan_llm_checker_example.py; fi

example-replan-voting:
	@if [ -f .env ]; then set -a && . .env && set +a && PYTHONPATH=$$(pwd) uv run python examples/replan/replan_voting_example.py; \
	else PYTHONPATH=$$(pwd) uv run python examples/replan/replan_voting_example.py; fi

example-replan-checkpoint:
	@if [ -f .env ]; then set -a && . .env && set +a && PYTHONPATH=$$(pwd) uv run python examples/replan/replan_checkpoint_rollback_example.py; \
	else PYTHONPATH=$$(pwd) uv run python examples/replan/replan_checkpoint_rollback_example.py; fi

example-replan-budget:
	@if [ -f .env ]; then set -a && . .env && set +a && PYTHONPATH=$$(pwd) uv run python examples/replan/replan_budget_guard_example.py; \
	else PYTHONPATH=$$(pwd) uv run python examples/replan/replan_budget_guard_example.py; fi

example-execution-modes:
	@if [ -f .env ]; then set -a && . .env && set +a && PYTHONPATH=$$(pwd) uv run python examples/orchestration/execution_modes_mock_example.py; \
	else PYTHONPATH=$$(pwd) uv run python examples/orchestration/execution_modes_mock_example.py; fi
	@if [ -f .env ]; then set -a && . .env && set +a && PYTHONPATH=$$(pwd) uv run python examples/orchestration/execution_modes_pipeline_example.py; \
	else PYTHONPATH=$$(pwd) uv run python examples/orchestration/execution_modes_pipeline_example.py; fi

example-metrics:
	@if [ -f .env ]; then set -a && . .env && set +a && PYTHONPATH=$$(pwd) uv run python examples/evaluation/metrics_example.py; \
	else PYTHONPATH=$$(pwd) uv run python examples/evaluation/metrics_example.py; fi

example-dataset-evaluator:
	@if [ -f .env ]; then set -a && . .env && set +a && PYTHONPATH=$$(pwd) uv run python examples/evaluation/dataset_evaluator_example.py; \
	else PYTHONPATH=$$(pwd) uv run python examples/evaluation/dataset_evaluator_example.py; fi

example-reflection-metrics:
	@if [ -f .env ]; then set -a && . .env && set +a && PYTHONPATH=$$(pwd) uv run python examples/evaluation/reflection_metrics_example.py; \
	else PYTHONPATH=$$(pwd) uv run python examples/evaluation/reflection_metrics_example.py; fi

# AgentSkill example: document analysis with PDF → LLM reasoning → PPTX slides
# Usage:
#   make example-agent-skill               # generates a minimal test PDF automatically
#   make example-agent-skill PDF=my.pdf    # uses your own PDF
PDF ?= /tmp/carl_test_doc.pdf
example-agent-skill:
	@if [ ! -f "$(PDF)" ]; then \
		echo "No PDF found at '$(PDF)' — generating a minimal test document..."; \
		uv run python -c "import zlib; \
text = b'BT /F1 12 Tf 50 750 Td (CARL AgentSkill Test Document) Tj 0 -20 Td (This document demonstrates the CARL AgentSkill integration.) Tj 0 -20 Td (It tests the PDF and PPTX AgentSkill steps in a reasoning chain.) Tj ET'; \
stream = zlib.compress(text); \
objs = [ \
  b'1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n', \
  b'2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n', \
  b'3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n', \
  b'4 0 obj\n<< /Length ' + str(len(stream)).encode() + b' /Filter /FlateDecode >>\nstream\n' + stream + b'\nendstream\nendobj\n', \
  b'5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n', \
]; \
body = b'%PDF-1.4\n'; offsets = []; \
[offsets.append(len(body) + sum(len(o) for o in objs[:i])) or None for i in range(len(objs))]; \
body += b''.join(objs); xref_pos = len(body); \
body += b'xref\n0 6\n0000000000 65535 f \n' + b''.join(f'{o:010d} 00000 n \n'.encode() for o in offsets) + b'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n' + str(xref_pos).encode() + b'\n%%EOF\n'; \
open('$(PDF)', 'wb').write(body); print('Test PDF written to $(PDF)')"; \
	fi
	@if [ -f .env ]; then set -a && . .env && set +a && PYTHONPATH=$$(pwd) uv run python examples/agents/agent_skill_example.py --pdf "$(PDF)"; \
	else PYTHONPATH=$$(pwd) uv run python examples/agents/agent_skill_example.py --pdf "$(PDF)"; fi

# SkillResolver + LLM_AGENT mode demo (creates a temp local skill, no pre-installed skills required)
example-skill-resolver:
	@if [ -f .env ]; then set -a && . .env && set +a && PYTHONPATH=$$(pwd) uv run python examples/skills/skill_resolver_example.py; \
	else PYTHONPATH=$$(pwd) uv run python examples/skills/skill_resolver_example.py; fi

# Run all examples with summary table
examples-runner:
	@if [ -f .env ]; then set -a && . .env && set +a && PYTHONPATH=$$(pwd) uv run python examples/runner.py; \
	else PYTHONPATH=$$(pwd) uv run python examples/runner.py; fi

# Run groups of related examples
examples-replan: example-replan-deterministic example-replan-llm example-replan-voting example-replan-checkpoint example-replan-budget
examples-metrics: example-metrics example-reflection-metrics example-dataset-evaluator
examples-reflection: example-reflection example-reflection-metrics

# Run all examples
examples: example-basic example-openrouter example-tool example-structured example-council example-reflection examples-replan example-execution-modes examples-metrics

# List all installed AgentSkills (SKILL.md-based + agent-skills library)
skills:
	@PYTHONPATH=$$(pwd) uv run python -c "\
from mmar_carl.skill_loader import SkillLoader; \
loader = SkillLoader(); \
entries = loader.catalog_all(); \
print(f'Installed AgentSkills ({len(entries)} found):') if entries else print('No skills found.'); \
[print(f'  {name}: {desc[:80]}') for name, desc in entries]"

.PHONY: example-basic example-openrouter example-tool example-structured example-council example-legacy example-reflection \
	example-replan-deterministic example-replan-llm example-replan-voting example-replan-checkpoint example-replan-budget \
	example-execution-modes example-metrics example-dataset-evaluator example-reflection-metrics examples-replan examples-metrics examples-reflection examples \
	example-agent-skill example-skill-resolver skills

version:
	echo "$(lib_name)==$(version)"

inc_version:
	v0=$(version); \
	v1=$$(python -c "major, minor, micro = '$$v0'.split('.'); print(f'{major}.{minor}.{int(micro)+1}')"); \
	sed -i "s/version = \"$$v0\"/version = \"$$v1\"/g" pyproject.toml

pypi:
	eval "$(pypi_command)"


push-all:
	git pull; git push; git checkout main; git pull; git merge dev; git push; git checkout -

wait-when-published:
	$(call wait-until-changed,"$(pypi_command)") && notify-send "DONE"

v: version
v++: inc_version version
p: pypi
t: test
wwp: wait-when-published
puw: push-all wait-when-published
