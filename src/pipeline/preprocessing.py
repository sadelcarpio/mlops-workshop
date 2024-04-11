import pandas as pd

df = pd.read_csv("https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv")
preprocessed_df = (df - df.min()) / (df.max() - df.min())

X = preprocessed_df.drop(["quality"], axis=1)
y = preprocessed_df[["quality"]]
X.to_csv("../data/X.csv", index=None)
y.to_csv("../data/y.csv", index=None)
