import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
df = pd.read_csv("results/metrics/statistical_comparisons.csv")
print(df.round(4).to_string(index=False))