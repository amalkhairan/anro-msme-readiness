# When Oversampling Degenerates — code and data

Reproducibility package for the manuscript

> A. Khairan, A. Fuad, F. Tempola, A. A. Qureshi, and I. Kabalmai, "When Oversampling Degenerates: Adaptive Neighborhood Repartition versus Simpler Imbalance Remedies on Small Ordinal Survey Data," submitted to the *Journal of Information Systems Engineering and Business Intelligence* (JISEBI).

The study tests whether Multiclass Neighborhood Repartition-based Oversampling (MC-NRO) works as specified on a small ordinal survey with a very rare class, shows that its repartition region is empty for every seed, proposes an adaptive variant (A-NRO), and compares 15 imbalance remedies under a leakage-free protocol. The survey data and the random-forest baseline come from

> A. Khairan, A. Fuad, Y. Muin, and S. Turuy, "Classifying digital readiness in traditional MSMEs: Random forest baseline evaluation and feature stability on imbalanced data," *TEKNIKA*, vol. 15, no. 3, pp. 573–582, 2026, doi: [10.34148/teknika.v15i3.1582](https://doi.org/10.34148/teknika.v15i3.1582).

## Repository layout

| Path | Content |
| --- | --- |
| `data/README.md` | Structure of the survey file expected by the code (the survey responses themselves are not published; see *Data availability*). |
| `notebooks/ANRO_experiments.ipynb` | Google Colab notebook that runs every survey experiment and writes all tables. |
| `code/encode.py` | Questionnaire encoding and label construction (Digital Readiness Score → Low / Medium / High). |
| `code/mcnro.py` | Re-implementation of MC-NRO and the proposed A-NRO (and the ablation variants). |
| `code/anro_exp.py` | Experiment core: preprocessing, 15 remedies, repeated stratified 5-fold × 10 cross-validation. |
| `code/anro_analysis.py` | Tables: pooled performance, corrected resampled t-test, bootstrap, respondent-level results, indicator stability. |
| `code/extras.py` | Macro ROC-AUC, macro PR-AUC, average precision of High, and Spearman ρ of indicator rankings (Tables 3 and 6). |
| `code/degen.py`, `code/degen2.py`, `code/fidelity.py` | Empty-region diagnostics: seed vs candidate potentials, data-scaled ε, and the sampling rule of the public MC-NRO code. |
| `code/keel_bench.py`, `code/keel_summarize.py` | External benchmark on 20 KEEL datasets and its summary (Table 7). |
| `code/samplers.py` | Plain implementations of ROS and SMOTE used in the KEEL benchmark. |
| `code/test_mcnro.py` | Property tests for `mcnro.py` (`python test_mcnro.py`, 15/15 PASS). |
| `results/` | Results reported in the manuscript: `ANRO_results_final.xlsx`, per-fold raw results (`raw/*.pkl`), sensitivity grid, empty-region diagnostics, KEEL results. |

## Reproducing the results

**Survey experiments (Tables 1–6, Figures 2–5).** Open `notebooks/ANRO_experiments.ipynb` in Google Colab and run the cells in order. The first cell pins scikit-learn 1.6.1 and imbalanced-learn 0.13.0 (the environment of the published baseline) and restarts the runtime once. Upload the survey file when the notebook asks for it. The notebook then checks that the baseline reproduces the published figures (accuracy 0.7384, balanced accuracy 0.4555, macro-F1 0.4470), runs all 15 remedies and the sensitivity grid, and exports `ANRO_results_final.xlsx` together with the raw per-fold results. A full run takes about 60–90 minutes on a standard Colab CPU.

Locally:

```bash
pip install -r requirements.txt
cd code
python test_mcnro.py          # property tests
python extras.py              # ROC-AUC, PR-AUC, Spearman rho from results/raw/*.pkl
python degen.py && python degen2.py && python fidelity.py   # empty-region diagnostics
```

`anro_exp.py` reads the data path from the environment variable `ANRO_DATA` and defaults to `data/msme_digital_readiness_anonymized.csv`. The scripts `extras.py`, `keel_summarize.py` and the tables in `results/` need no survey data: they work from the stored results.

**KEEL benchmark (Table 7).** The benchmark uses the 20 multiclass KEEL datasets and the five-fold partitions (repeated twice) distributed with the MC-NRO reference code. Download that repository, then point `MCNRO_FOLDS` to its `data/folds` directory:

```bash
export MCNRO_FOLDS=/path/to/MC-NRO-master/data/folds
python keel_bench.py          # appends to results/keel_results.jsonl
python keel_summarize.py      # writes results/keel_summary.csv, prints ranks and Holm-adjusted Wilcoxon tests
```

`results/keel_results.jsonl` already holds the run reported in the manuscript, so `keel_summarize.py` can be run directly.

## Protocol in brief

19 non-label ordinal items and 7 demographic variables (50 columns after one-hot encoding); the five items that define the label (Q8, Q9, Q10, Q12, Q14) are excluded from the predictors. Imputation, encoding and every resampling step are fitted inside each training fold. Random forest: 500 trees, Gini criterion, square-root feature sampling, bootstrap, `random_state=42`. Outer validation: repeated stratified 5-fold cross-validation with 10 repetitions (`random_state=42`).

## Data availability

The survey responses are not published in this repository, to protect the respondents: even without names and contact details, the combination of demographic variables could identify individual traders in the three markets. The anonymized data set (names, e-mail addresses, telephone numbers and timestamps removed) is available from the corresponding author on reasonable request. All results reported in the manuscript are included in `results/`, and the per-fold raw results in `results/raw/` hold only row indices, predictions, class probabilities and importances; `results/raw/labels.csv` gives the readiness class of each row, which `extras.py` needs.

## License

Code: MIT License (see `LICENSE`). Please cite the manuscript and the TEKNIKA article above when using the code.

## Contact

Amal Khairan, Department of Informatics, Faculty of Engineering, Universitas Khairun, Ternate, Indonesia — amalkhairan@unkhair.ac.id
