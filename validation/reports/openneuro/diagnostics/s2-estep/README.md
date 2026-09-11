# Bounded two-participant E-step diagnosis

The original comparison remains **FAIL**. `diagnosis.json` records the
mismatch, input/source/runtime hashes, arithmetic witnesses, and limitations.
The 51 selected rows are all vertices whose final posterior violates the
original tolerance. Both participants and both sessions are retained.

`input.mat` supplies final saved directions and theta. This is a new E-step
replay, **not a reconstruction of the original last E-step**: its incoming
theta was not captured. Reduced matrix shape can also change BLAS dispatch.
No whole fit, tolerance change, or frozen source edit was performed.

Run the independently extracted CBIG E-step and then the frozen Python tail:

```sh
octave --quiet --eval "addpath('.'); s2_estep_witness_tail(pwd);"
python replay.py
```

Use the recorded runtimes to reproduce exact arithmetic. `octave-tail.mat`
and `python.mat` retain observed outputs. `report.json` compares native matrix
products; `tail-and-layout-report.json` supplies identical upstream products
to the unmodified Python E-step tail and records norm-reduction layouts.
`python_estep_frozen.py` contains verbatim selected functions from pyMSHBM;
its MIT license is included. The upstream .m body and helpers retain CBIG's
included MIT license. The additional raw-product and log-normalizer outputs
are diagnostic observations; estimator algebra is unchanged.
