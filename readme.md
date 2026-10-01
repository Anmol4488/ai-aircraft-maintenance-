# Aircraft Maintenance Decision Support

Machine-learning and language-model tools built on 700k+ FAA Service Difficulty Reports (SDRs, 2015 to Sept 2026), served through a FastAPI service.

> **Decision support only.** Nothing here is a maintenance instruction or an airworthiness decision. All outputs are statistical suggestions that a qualified, certified person must verify against approved maintenance data.

## What it does (and what it does not)

| Endpoint | Input | Output |
|---|---|---|
| `POST /risk` | an aircraft's make, model, system (ATA chapter), date and earlier reports | probability of **another defect in the same system within 90 days** |
| `POST /classify` | defect report text | the 3 most likely systems (ATA chapters) |
| `POST /suggest-fix` | problem description | a drafted corrective action |
| `GET /health` | none | which models are loaded |

**It does not predict maintenance due dates.** The data are defect reports, not maintenance schedules or flight-hour telemetry, so the honest question the data can answer is: *"given this aircraft's history, how likely is a repeat defect in this system soon?"*

## Results

All models are evaluated on **later dates than they were trained on** (train: 2015-2023, validation: 2024, test: 2025 to mid-2026), never on a random split, and always against a simple baseline.

### 1. Repeat-defect risk (LightGBM)
Label: another report in the same system within 90 days. 9,904 aircraft with at least 10 reports; 55,781 test examples.

| Model | Accuracy | AUC |
|---|---|---|
| Always predict "no repeat" | 57.1% | n/a |
| Rule: 2+ same-system reports in last 90 days | 68.1% | 0.654 |
| **LightGBM** | **71.6%** | **0.782** |

The model catches about 53% of real repeats (recall), so it misses a lot. Examples from the last 90 days of the data are dropped because their outcome is not yet known.

### 2. Report text to system (SmolLM2-135M vs TF-IDF)
32 systems; 60,000 training reports; 6,000 test reports.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Always predict the most common system | 37.7% | n/a |
| TF-IDF + logistic regression (about 1 min on CPU) | 90.4% | 0.699 |
| SmolLM2-135M fine-tuned (2 h 9 min on a 4 GB laptop GPU) | 90.2% | 0.666 |

**A tie.** For pure classification the cheap baseline is the better engineering choice.

### 3. Problem to drafted fix (SmolLM2-135M)
Trained on 40,000 problem/fix pairs, 1 epoch (about 65 min); scored on 500 test problems.

| Method | Word-overlap F1 | Starts with same action word |
|---|---|---|
| Always answer "REPAIRED FRAME." | 0.030 | 10.5% |
| Copy the fix of the most similar past problem | 0.271 | 37.7% |
| **SmolLM2-135M fine-tuned** | **0.391** | **46.0%** |

Word overlap is a crude metric: a correct fix worded differently scores low. Human review of samples is still needed.

## Known limitations

- **Learns common practice, not diagnosis.** The fix generator mostly suggests "remove and replace". In one test it proposed replacing an engine for a fuel-imbalance report where the real fix was a wiring repair. In another it suggested replacing a smoking oven where the real fix was cleaning and resetting it.
- **Invented verification claims.** The model learned to end fixes with phrases like "OPS CHECKED GOOD". The API removes any sentence claiming a check was performed.
- **Manual references removed from training targets.** A model cannot know the right manual page, so it would invent one.
- **Classifier mistakes happen.** A galley-oven smoke report was classified as air conditioning (21) with 47% confidence; cabin equipment (25) was not in the top 3.
- **Risk scores are uncalibrated and need history.** The model was trained on aircraft with 10+ reports; the API warns when given fewer. Model names must match the FAA spelling or are treated as unknown.
- **Repeat defects partly reflect inspection intensity.** One heavy check can generate many reports, which inflates the repeat rate for that aircraft.
- Only about 43% of reports could be split automatically into problem and fix; the rest are not used for the fix generator.

## How it was built

1. `prepare_data.py` cleans the CSVs (dates, aircraft identity, duplicates, impossible hours/cycles, usage values that contradict an aircraft's own timeline) and builds the per-aircraft history table.
2. `build_risk_dataset.py` creates labelled examples with strictly past-only features and a date-based split.
3. `train_lgbm.py` trains and saves the risk model.
4. `build_text_dataset.py`, `baseline_text.py`, `train_smollm_classifier.py` handle report text to system.
5. `build_gen_dataset.py`, `baseline_gen.py`, `train_smollm_generator.py` handle problem to fix.
6. `app/` is the FastAPI service; `tests/` has 16 automated tests (feature logic, input validation, safety filter, error handling).

The API computes risk features itself from the history sent, with the same logic as training. This was verified on 300 real examples with 0 mismatches.

## Run it

Data and trained models are **not** included in this repository (size). To reproduce:

```bash
# 1. Install PyTorch with the CUDA build for your GPU from pytorch.org (without torchvision/torchaudio)
python -m pip install -r requirements.txt

# 2. Put the FAA SDR yearly CSVs in data/faa/ (SDR-2015.csv ... SDR-2026.csv)
python prepare_data.py --data-dir data/faa --out-dir processed
python build_risk_dataset.py --in-dir processed --horizon 90
python train_lgbm.py --in-dir processed
python build_text_dataset.py --in-dir processed
python train_smollm_classifier.py --epochs 2 --batch-size 8 --grad-ckpt
python build_gen_dataset.py --in-dir processed
python train_smollm_generator.py --max-train 40000 --epochs 1 --batch-size 16 --grad-ckpt

# 3. Tests and server
python -m pytest -q
python -m uvicorn app.main:app --port 8000      # then open http://127.0.0.1:8000/docs
```

Set `LOAD_LLMS=0` to run the API without the two language models (the tests do this).

## Data

FAA Service Difficulty Reports, 2015 to 25 Sept 2026: 728,940 raw reports, 715,240 after cleaning, from 30,925 aircraft. Public data; not redistributed here.