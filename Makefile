# `make test` needs only the [test] extra. `make gate` is the maintainers' commit gate:
# the same run with the pytest_budget plugin loaded, which fails any test over
# TEST_BUDGET_S seconds; the `gate` tool reads GATE_BUDGET_S for the whole run. CI runs
# `pytest tests/` and loads neither.
# Slowest test: 10 s alone (setup and call).
TEST_BUDGET_S = 30
# Whole run: 21 s alone.
GATE_BUDGET_S = 90

PYTEST := python3 -m pytest -q
FAST := -n 8 --dist loadfile

.PHONY: test gate

test:
	$(PYTEST) $(FAST)

gate:
	$(PYTEST) -p pytest_budget -o test_budget_s=$(TEST_BUDGET_S) $(FAST)
