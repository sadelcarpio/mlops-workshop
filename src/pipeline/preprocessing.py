import pandas as pd

df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")

X = df.drop(["quality"], axis=1)
X = X.rename(columns={
    "fixed acidity": "fixed_acidity",
    "volatile acidity": "volatile_acidity",
    "citric acid": "citric_acid",
    "residual sugar": "residual_sugar",
    "free sulfur dioxide": "free_sulfur_dioxide",
    "total sulfur dioxide": "total_sulfur_dioxide",
})
y = df[["quality"]]
X.to_csv("../data/X.csv", index=None)
y.to_csv("../data/y.csv", index=None)
