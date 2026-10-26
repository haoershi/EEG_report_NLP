# Seizure classification of EEG reports with an LLM: analysis code

This repository reproduces the tables and figures of the manuscript from de-identified data. It contains no report text, no dates, no demographics, and no original identifiers.

## Contents


| Path                              | What it is                                                                                                                                                                         |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Manuscript.ipynb`                | The full analysis. It builds the model labels from the five runs, then produces every table and figure.                                                                            |
| `release_data/`                   | The de-identified inputs, described below.                                                                                                                                         |
| `derived/`                        | Cached label arrays and bootstrap scores, described below. The notebook rebuilds them if they are missing.                                                                        |
| `ds_pred_allnote.py`, `run_ds.py` | The prompts and inference loop used to run DeepSeek-R1-Distill-Llama-8B on the reports. They need the report text, which is not released, so they are provided for reference only. |




## Data (`release_data/`)

`PATIENT_ID` and `ORDER_ID` are random identifiers. 


| File             | Columns                                                                                                | Rows                                |
| ---------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------- |
| `orders.csv`     | `PATIENT_ID`, `ORDER_ID`, `TYPE` (LTM / ROUTINE / AMB), and the 0/1 flags `nlp`, `development`, `test` | 16,976 EEG orders (the full cohort) |
| `annot_dev.csv`  | `ORDER_ID`, plus `Label1`–`Label5`: five raters' classes, comma-separated when a report has several    | 166 (development set)               |
| `annot_test.csv` | `ORDER_ID`, `rater_1`, `rater_2`, and `label` (the consensus label)                                    | 100 (test set)                      |


`Manuscript.ipynb` reads the prediction tables below. `<set>` is `dev`, `test`, or `nlp`. The full cohort uses the NLP consensus, and every order outside that set is class 0.


| File                       | Columns                                                                                                                | What the cells are                                                                                                                                                              |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pred_by_run_<set>.csv`    | `ORDER_ID`, `class`, `runs_0`–`runs_4`, `consensus`                                                                    | The five cleaned answers for that class (class 0 is already flipped) and the majority vote before the exclusion rules. A blank cell means the model was not run on that report. |
| `pred_consensus_<set>.csv` | `ORDER_ID`, then one column per class (`Class0`, `Class1`, `Class2`, `Class3`, `Class4`, `Class5`, `Class6`, `Class9`) | The consensus after the exclusion rules. Reports with no model output are class 0, which is how the full cohort is labeled outside the NLP set.                                 |


The classes are:

- 0: no seizure or event
- 1: electroclinical seizure
- 2: subclinical seizure
- 3: electrographic seizure, clinical status unknown
- 4: clinical event without an EEG correlate
- 5: clinical event, EEG status unknown
- 6: myoclonus
- 9: unclear or not mentioned

Class 0 was asked as "is there a seizure or event", so a raw answer of 1 means yes. In `pred_by_run_*.csv` that answer is already flipped, and `consensus` is the majority vote before the exclusion rules.

## Derived files (`derived/`)

These are produced by `Manuscript.ipynb` and included so the tables can be rebuilt without repeating the bootstrap. Delete a file and rerun the notebook to recreate it.


| File                    | What it holds                                                                                                                                                                                                                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `dev_data.pkl`          | Development set (n=166). A list `[y_phys, y_phys_bin, y_true, y_true_bin, y_ds, y_ds_bin, full_pred]`: the five raters, those labels binarized, the consensus and its binarized form, the five model runs, and the consensus prediction after the exclusion rules.                                    |
| `test_data.pkl`         | Test set (n=100), in the same list. `y_phys` is the two raters (`rater_1`, `rater_2`), and `y_true` is the consensus `label`.                                                                                                                                                                          |
| `scores_bootstrap.pkl`  | Pair-level scores and 1,000 note-level bootstrap replicates (seed 42). Development-set human and model scores are `scores_phys` and `scores_ds`, with `_bs` arrays for the replicates. The test set is under `val2` (labels, full-data scores, and replicates). Set `SCORE_RECOMPUTE = True` to rebuild it. |


## Running

Python 3.11 with:

```
pip install pandas numpy scikit-learn scipy matplotlib seaborn tableone statsmodels tqdm ppi-python jupyter
```

The analysis was run with pandas 3.0, numpy 2.4, scikit-learn 1.9, scipy 1.17, matplotlib 3.11, seaborn 0.13, tableone 0.9.6, statsmodels 0.15, and ppi-python 0.2.3.

The notebook writes the tables (`.csv`) and figures (`.svg`) to `manuscript/`. The bootstrap cached in `derived/scores_bootstrap.pkl` is the slow step.

Patient demographics (age and sex) are not released, so Table 2 reports only patient counts, report counts, and recording type. The report timeline figure needs dates and is not included.