# Implementation plan

**Goal:** Create an installable, tested, BIDS-friendly Python MS-HBM package
with a measured compatibility path to Buckner/CBIG.

**Spec:** [design.md](design.md)

## Global constraints

- Preserve Poldrack history and attribution; private `lobennett/pyMSHBM`.
- fsaverage6 / fsaverage3 / DU15NET, 15 labels, max_iter=5 by default.
- Default workflow follows group-estimation posterior extraction, no MRF.
- No implicit denoising, resampling, seed ROI averaging or Fisher-Z conversion.
- Upstream reference execution and measured tolerances support parity claims.
- Input subjects/runs and all assets must have reproducible provenance.

## Tasks

- [x] Establish inherited baseline and pin references.
- [x] Audit numerical EM and add regression tests before corrections.
- [x] Build/run independent Octave oracle on reproducible synthetic inputs.
- [x] Implement BIDS discovery and CLI after tests for pairing/ambiguity.
- [x] Implement canonical profiles, initialization and workflow after tests.
- [x] Package with uv, document installation, scientific contract and limits.
- [x] Independently review, fix findings, run full tests and wheel checks.
- [ ] Create and push private repository; verify GitHub state and CI.

Parallel interfaces: discovery produces SurfaceRun(subject, session, task,
run, lh, rh); integration consumes it. Reference generator supplies fixed
normalized data and g_mu to estimate_group_priors(data, g_mu, settings).
Only numerical worker changes group_priors/vmf. Only integration worker
changes pyproject, types and canonical pipeline. Review findings must be
resolved before publication. Development evidence is recorded in WORK_LOG.md.
