import glob
import pandas as pd

print("Raw files:", glob.glob("data/raw/*"))
print("Processed files:", glob.glob("data/processed/*"))

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

print("\nTrain rows:", len(train), "(expect 736)")
print("Test rows:", len(test), "(expect 184)")
print("Total:", len(train) + len(test), "(expect 920)")
print("\nColumns:", train.columns.tolist())
print("\nMissing values per column (train):")
print(train.isna().sum())