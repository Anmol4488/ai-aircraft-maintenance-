import pandas as pd

r = pd.read_parquet("processed/risk_dataset.parquet")
test = r[r["split"] == "test"]

# 1. "Always say no" baseline
acc_always_no = (test["label"] == 0).mean()

# 2. Simple rule: predict 1 if the same system had a report in the last 90 days
pred = (test["n_ata_last90"] >=0).astype(int)
acc_rule = (pred == test["label"]).mean()

print(acc_always_no, acc_rule)