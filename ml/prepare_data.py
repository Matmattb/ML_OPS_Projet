import pandas as pd

INPUT_PATH = "data/raw/credit_card.xls"
OUTPUT_PATH = "data/raw/credit_default.csv"

df = pd.read_excel(INPUT_PATH, header=1)


df = df.drop(columns=["ID"])
df = df.rename(columns={"default payment next month": "default_next_month"})

df.to_csv(OUTPUT_PATH, index=False)
print(f"OK : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(df["default_next_month"].value_counts(normalize=True))