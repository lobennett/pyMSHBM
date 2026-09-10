# Development evidence

- Base: poldrack/pyMSHBM `11b201df13c8e72caa18d3e47feda3f7d6e0f265`.
- Buckner: PrecisionNetworkMapping `988ec48cf5453c6009a6f82dbddf5d5b99eda3cd`.
- Isolated clone and feature branch; original remote renamed `poldrack`.
- Decision: use executable MATLAB script as behavioral contract because the
  Buckner README disagrees with its training script.
- Initial audit: seed averaging, Fisher-Z profiles, omitted mean centering,
  k-means initialization and extra MRF fitting diverge in the inherited pipeline.
- Baseline test run started with Python 3.14.5 and uv-resolved environment.
- Baseline completed: 186 passed, 1 skipped, 59.46 seconds.
- Added BIDS discovery/CLI, immutable run mapping, strict hemisphere/echo pairing;
  initial BIDS-specific validation 40 passed.
- Core comparison executed upstream CBIG code in Octave6.4.0. Initial standard
  and missing-session cases match all unpermuted labels and outer iterations.
- Scientific corrections include complete nested EM, session averaging, zero
  masks, literal inverse-concentration return, costs, stopping, and dtypes;
  see docs/numerical-audit.md and validation fixture reports.
- Full integrated suite at first integration: 267 passed,1 skipped,27.38sec.
- uv built wheel/sdist; isolated wheel installed with Python3.12.13, runtime
  dependencies only. Both CLIs and BIDS dry-run executed successfully.
- Initial full-size synthetic smoke exposed remaining signed-denominator
  divergence in intra convergence. Executed upstream state confirmed the bug;
  numerical worker added regression and corrected the literal denominator.
- Independent Astra review found partial-NaN medial-column imputation and a
  missing production-float32 oracle case. Whole-column correction passes its
  failing regression; oracle precision adapters are being audited against MEX C.
- Full-size variable synthetic smoke passed in10.20sec: 81924 vertices, 2 runs,
  one participant,15networks,max_iter1, actual GIFTI/asset/CLI/model/output path.
  Local evidence: .local-data/smoke-variable/smoke-report.json.
- Runtime input validation rejects changed source files and nonpositive or
  nonfinite fitted concentration parameters before publishing a derivative.
- Final independent re-review resolved both findings;72focused tests passed.
- Executed mtimesx C dtype audit established single-dominant operand conversion;
  float32 oracle and regression now exercise the actual production path.
- Final full suite before documentation refresh:322passed,1optional-data skip.
- Default full-surface run and isolated Python3.12 wheel run both completed
  five outer iterations with relative cost6.20e-6 and valid outputs.
- Private GitHub repository created at https://github.com/lobennett/pyMSHBM.
- Final reference refresh plus complete suite:323passed,1optional-data skip in28.46sec.
- Original upstream source/adapter/log whitespace is retained byte-for-byte
  for checksums; .gitattributes prevents automatic newline conversion.
- Published reviewed implementation4ae9b10 to private lobennett/pyMSHBM main.
- GitHub Actions run34529901800 succeeded:Python3.11,3.13,3.14 test matrix,
  wheel/sdist build,uv tool installation,and both CLI help commands.
- Fresh final wheel/sdist build succeeded;all local Markdown targets resolve.
- Task-created Octave container stopped after reference execution; source
  checkouts and ignored local smoke data remain available for reproduction.
