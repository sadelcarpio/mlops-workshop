import pandas as pd

df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")
preprocessed_df = (df - df.min()) / (df.max() - df.min())

X = preprocessed_df.drop(["quality"], axis=1)
y = preprocessed_df[["quality"]]
X.to_csv("../data/X.csv", index=None)
y.to_csv("../data/y.csv", index=None)
