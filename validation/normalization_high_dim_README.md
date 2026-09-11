# Full-dimension profile normalization oracle

This is a separate regression for feature-order float32 accumulation at the
production D=1,175 seed dimension. It leaves the older fixture family unchanged.
Expected outputs come from the unchanged normalization statements extracted
from pinned `CBIG_MSHBM_estimate_group_priors.m`, executed in GNU Octave 6.4.0.
Both the 16-row batch and each individual row are executed independently.

Reproduce with an installed Octave:

```sh
.venv/bin/python validation/generate_normalization_high_dim.py
.venv/bin/pytest tests/test_reference_normalization_high_dim.py -q
```

To execute in an existing container, run `--prepare-only`, execute
`normalization_high_dim_oracle('/work/pyMSHBM/validation')` with
`/work/pyMSHBM/validation/runtime` on Octave's path, redirect the execution log to
`validation/fixtures/normalization_high_dim_octave.log`, then run
`--collect-existing` on the generator. The existing `pymshbm-oracle` container
was used for the checked-in fixture.

The provenance file records source, adapter, generator, raw MATLAB input/output,
compressed fixture and execution-log checksums. Raw MATLAB files are regenerated
intermediates; the compressed fixture contains their inputs and expected arrays.
Tests use exact equality for
normalization; no existing comparison tolerance was changed. This verifies the
array kernel and does not constitute another end-to-end fit or a correction for
single-subject concentration nonidentifiability.
