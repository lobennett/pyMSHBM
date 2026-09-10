# Reference validation

Read [executed upstream validation](../docs/reference-validation.md) for the
scientific scope, pinned sources, adapter boundaries, runtime, measured errors
and complete regeneration commands.

From the repository root:

```bash
uv sync --frozen --extra dev
uv run pytest tests/test_reference_oracle.py tests/test_reference_profiles.py tests/test_reference_intra_boundary.py -q
uv run python validation/compare_reference.py --case complete
uv run python validation/compare_reference.py --case missing
uv run python validation/compare_reference.py --case soft
uv run python validation/compare_reference.py --case single
uv run python validation/compare_profiles.py
```

These commands compare Python with portable NPZ fixtures produced by executed
CBIG MATLAB source in Octave. They do not generate expected values from Python.
Source, adapter and fixture hashes are checked by the tests. Regeneration
requires Octave and uses `generate_reference.py` / `generate_profile_reference.py`;
see the linked report before interpreting or modifying adapters.

`run_surface_smoke.py` generates full-size synthetic fsaverage6 BIDS GIFTIs and
runs the Python CLI. It requires prepared external reference assets and adequate
disk/RAM. This exercises the real installation and output path; it is separate
from the small numerical oracle and does not establish real-data equivalence.
