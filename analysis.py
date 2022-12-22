import pandas as  pd 
import numpy as np 
import re
import string


df_fake = pd.read_csv("../data/Fake.csv")
df_true = pd.read_csv("../data/True.csv")

df_fake["class"] = 0
df_true["class"] = 1

df_combined = pd.concat([df_fake,df_true], axis = 0)
df = df_combined.drop(["title", "subject","date"], axis = 1)

df = df.sample(frac = 1)
df= df.reset_index(drop=True)

df.to_csv("../data/dataset.csv")

print(df.head())