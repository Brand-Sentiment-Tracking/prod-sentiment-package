import pandas as pd
from langdetect import detect

df = pd.read_parquet('finetune.parquet')[:2000]

# for i, t in enumerate(df['text'].values):
#     if detect(t) == 'en':
#         pass
#     else:
#         df = df.drop(labels=i, axis=0)

# print(df.count())
df.to_csv('finetune.csv')
