import pandas as pd

df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")

X = df.drop(["quality"], axis=1)
y = df[["quality"]]
X.to_csv("../data/X.csv", index=None)
y.to_csv("../data/y.csv", index=None)
