# Reproducing the results

Run commands from this `code/` directory after installing `requirements.txt`:

```bash
python3 run_experiments.py
python3 robustness.py
```

`run_experiments.py` generates the main simulation results and manuscript figures. The full run uses 400 replications and 1,000 Gaussian calibration draws; it takes about 85 seconds on the machine used for the revision. `--quick` runs a smaller configuration. Use `--figs-only` to regenerate figures from the checked-in `results.json` without rerunning the main simulations.

`robustness.py` generates the robustness results and writes a LaTeX table to `tab_robustness.tex`. It also accepts `--quick`. Both scripts use CPU-only NumPy, SciPy, and Matplotlib; no external data, API keys, GPU, or network connection is needed. `test_audit.py` contains the scientific regression tests.

The main script writes `results.json` and generated PDF figures to `figs/`. The robustness script writes `results_robustness.json` and its LaTeX table to `tab_robustness.tex`.
