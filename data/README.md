# Data directory

Generated data files (gitignored). To regenerate:

```bash
# Generate saturation table (requires CoolProp + scipy)
python scripts/generate_saturation_table.py
```

This creates `saturation_table.npz`, which is loaded at runtime by `co2_eos/saturation.py`.
