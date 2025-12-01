from datasets import load_dataset
ds = load_dataset("AiresPucrs/adult-census-income")["train"]
df = ds.to_pandas()
print(df.columns)