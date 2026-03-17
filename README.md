# MaBoost — ICU Mortality & LOS Prediction

> **Mamba SSM + XGBoost hybrid for irregular clinical time-series**  
> Predicts **in-hospital mortality** (AUROC) and **length-of-stay** (MAE) simultaneously  
> LIDP Lab · Ha Noi University of Science and Technology

---

## Quick Start

```bash
# 1. Place MIMIC-III CSV.gz files in the data folder
#    /home/milis/maboost/data/raw/mimic3/
#    Required: CHARTEVENTS.csv.gz, ICUSTAYS.csv.gz,
#              ADMISSIONS.csv.gz, PATIENTS.csv.gz
#    Optional: LABEVENTS.csv.gz, DIAGNOSES_ICD.csv.gz

# 2. Run the full pipeline
cd /home/milis/maboost
chmod +x run.sh
./run.sh

# 3. Skip ETL on subsequent runs (much faster)
./run.sh --skip-etl

# 4. ETL only (no training)
./run.sh --etl-only
```

Results are saved to `/home/milis/maboost/results/`  
Checkpoints are saved to `/home/milis/maboost/checkpoints/`

---

## Project Structure

```
maboost/
├── run.sh                   ← bash pipeline runner
├── run_experiment.py        ← Python orchestrator (called by run.sh)
├── config.yaml              ← all hyperparameters in one place
├── README.md
│
├── data/
│   ├── raw/mimic3/          ← put MIMIC-III CSV.gz files here
│   ├── processed/           ← etl_output.pkl cached here after first run
│   └── splits/              ← train/val/test_ids.pkl saved here
│
├── src/
│   ├── data/
│   │   ├── loader.py        stream CSV.gz files (Polars, no OOM)
│   │   ├── preprocess.py    8-stage ETL pipeline
│   │   └── dataset.py       PyTorch Dataset, dual labels, normalisation
│   │
│   ├── models/
│   │   ├── mamba_encoder.py MambaEncoder + DualHeadMamba
│   │   ├── xgboost_head.py  XGBMortality + XGBLos
│   │   └── baselines.py     GRU-D, Transformer, LSTM
│   │
│   ├── training/
│   │   ├── stage1_train.py  joint BCE + MSE training, FP16, gradient clip
│   │   └── stage2_train.py  freeze encoder, extract features, dual XGBoost
│   │
│   ├── inference/
│   │   ├── online_pipeline.py   hospital real-time predictor + drift update
│   │   └── offline_pipeline.py  5-model benchmark, dual metrics
│   │
│   └── visualization/
│       ├── training_plots.py    8 evaluation plots (mortality + LOS)
│       └── shap_explain.py      SHAP beeswarm, waterfall, UMAP
│
└── checkpoints/
    ├── encoder_best.pth     saved after Stage 1
    ├── xgb_mortality.ubj    saved after Stage 2
    └── xgb_los.ubj          saved after Stage 2
```

---

## Architecture

```
MIMIC-III CSV.gz
      │
      ▼
┌─────────────────────────────────────────────────┐
│  ETL  (src/data/)                               │
│  loader → preprocess (8 stages) → dataset      │
│  Output: seq (48×47) · tau · mask · static (42) │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  Stage 1 — Mamba Encoder  (train_stage1.py)     │
│                                                 │
│  x (B,48,47) ──► TimeEmbedding                 │
│               ──► Mamba blocks × 3              │
│               ──► ChannelAttention              │
│               ──► WeightedPooling               │
│               ──► z_T (B, 128)                  │
│                                                 │
│  Loss = BCE(mortality) + 0.5 × MSE(log LOS)    │
│  FP16 · AdamW · CosineWarmRestarts · clip=1.0  │
│  Saves: encoder_best.pth                        │
└──────────────────────┬──────────────────────────┘
                       │  freeze encoder (permanent)
                       ▼
┌─────────────────────────────────────────────────┐
│  Stage 2 — Dual XGBoost  (train_stage2.py)      │
│                                                 │
│  F_final = [z_T (128) ; x_static (42)] ∈ ℝ¹⁷⁰  │
│                                                 │
│  XGBMortality → binary:logistic                 │
│  XGBLos       → reg:squarederror (log scale)    │
│                                                 │
│  Saves: xgb_mortality.ubj · xgb_los.ubj         │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   Offline benchmark          Online predictor
   (5 models, dual metrics)   predict(stay_id, x, τ, static)
                               → {mortality_risk, los_days,
                                  risk_level: LOW/MOD/HIGH}
                               update() on discharge batch
                               → Leaf Refresh / Continued Boost
```

---

## Features

### Time-Series (47 channels)
| Group | Features |
|-------|---------|
| Vitals | heart_rate, sbp, dbp, map, spo2, resp_rate, temp_c |
| Neuro | gcs_total, gcs_verbal, gcs_motor, gcs_eye |
| Metabolic | glucose, lactate, bicarbonate, ph, pao2, paco2, fio2 |
| Renal | creatinine, bun, urine_output |
| Haematology | hemoglobin, hematocrit, wbc, platelet |
| Liver | bilirubin_total, alt, ast |
| Coagulation | inr, pt, ptt |
| Electrolytes | sodium, potassium, chloride, calcium, magnesium, phosphate |
| Inflammatory | crp, procalcitonin |

### Static (42 features)
age, gender, admission type, ICU unit (5 types), insurance (3 types),  
30 Elixhauser comorbidity binary flags (CHF, COPD, renal failure, …)

---

## Output Files

```
results/
├── mortality/
│   ├── roc_curve.png          ROC with 95% bootstrap CI
│   ├── pr_curve.png           Precision-Recall
│   ├── confusion_matrix.png   at best-F1 threshold
│   ├── calibration.png        reliability diagram
│   └── threshold_sweep.png    sensitivity/specificity/F1 vs threshold
├── los/
│   ├── scatter.png            predicted vs actual (log scale)
│   ├── residuals.png          error distribution
│   └── error_by_quartile.png  MAE per LOS quartile
├── shap/
│   ├── mortality_beeswarm.png
│   ├── mortality_bar.png
│   ├── mortality_waterfall_top1-5.png  per high-risk patient
│   ├── los_beeswarm.png
│   ├── los_bar.png
│   └── latent_umap.png        Mamba z_T space coloured by risk
├── benchmark_comparison.png   AUROC bar chart all models
└── benchmark.csv              full metrics table
```

---

## Hospital Online Usage

```python
from src.inference.online_pipeline import MaBoostOnlinePipeline

# Load once at startup
model = MaBoostOnlinePipeline.load(
    ckpt_dir = "/home/milis/maboost/checkpoints/",
    d_input  = 47,
    d_model  = 128,
    device   = "cuda",
)

# Call every time a new observation arrives
result = model.predict(
    stay_id  = "ICU-00123",
    x_new    = vitals_array,      # shape (47,)
    tau_new  = 3600.0,            # seconds since last observation
    x_static = static_features,  # shape (42,)
)
# result = {'mortality_risk': 0.34, 'los_days': 4.2, 'risk_level': 'MODERATE'}

# Call once per discharge batch (CPU only, < 100 ms)
model.update(F_batch, y_true_batch)

# Clean up on patient discharge
model.discharge("ICU-00123")
```

---

## Configuration

All hyperparameters live in `config.yaml`. Edit there — never touch Python files.

Key settings:
```yaml
data:
  mimic_dir: /home/milis/maboost/data/raw/mimic3
  min_age:   18      # set 65 for geriatric cohort
  seq_len:   48      # timesteps per sequence

stage1:
  epochs:    100
  lr:        0.001
  device:    cuda

stage2:
  n_estimators: 500
  device:       cuda
```

---

## Dependencies

```
torch >= 2.7  (CUDA 12.8)
mamba-ssm >= 2.3.1
polars >= 1.39
xgboost >= 3.2
scikit-learn >= 1.8
river >= 0.23
shap >= 0.51
matplotlib >= 3.10
pyyaml
```

---

## References

1. Gu & Dao — *Mamba: Linear-time sequence modeling with selective state spaces* (2023)
2. Mottalib et al. — *HyMaTE: Hybrid Mamba and Transformer for EHR* (2025)
3. Chen & Guestrin — *XGBoost: A scalable tree boosting system* (KDD 2016)
4. Che et al. — *GRU-D: Recurrent Neural Networks for Multivariate Time Series with Missing Values* (2018)
5. Johnson et al. — *MIMIC-III, a freely accessible critical care database* (2016)# maboost-icu
