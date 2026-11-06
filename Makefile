# Test tiers (CLAUDE.md, "Tests"). `make test` needs only the [test] extra.
# `make gate` is the maintainers' commit gate: the same tier with the pytest_budget
# plugin loaded, which fails any unmarked test over TEST_BUDGET_S seconds; the
# `gate` tool reads GATE_BUDGET_S for the whole run. CI runs `pytest tests/` and
# loads neither.
TEST_BUDGET_S = 30
GATE_BUDGET_S = 120

PYTEST := python3 -m pytest -q
FAST := -n 8 --dist loadfile -m "not slow"

.PHONY: test gate slow

test:
	$(PYTEST) $(FAST)

gate:
	$(PYTEST) -p pytest_budget -o test_budget_s=$(TEST_BUDGET_S) $(FAST)

slow:
	$(PYTEST) -n 3 -m slow
