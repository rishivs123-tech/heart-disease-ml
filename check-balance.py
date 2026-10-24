import pandas as pd

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
full = pd.concat([train, test])

print("Overall target counts (expect 411 absent / 509 present):")
print(full["target"].value_counts())
print("\nTrain target:", train["target"].value_counts().to_dict())
print("Test target:", test["target"].value_counts().to_dict())
print("\nMissing chol overall:", full["chol"].isna().sum())
print("\nSource sites:")
print(full["dataset"].value_counts())