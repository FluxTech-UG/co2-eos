# `make test` needs only the [test] extra. `make gate` is the maintainers' commit gate, the
# same run; the `gate` tool reads GATE_BUDGET_S for the whole run. CI runs `pytest tests/`.
# Whole run: 21 s alone.
GATE_BUDGET_S = 90

PYTEST := python3 -m pytest -q
FAST := -n 8 --dist loadfile

.PHONY: test gate

test:
	$(PYTEST) $(FAST)

gate:
	$(PYTEST) $(FAST)
