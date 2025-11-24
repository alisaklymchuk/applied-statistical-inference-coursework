import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
from scipy.stats import bootstrap

ist_full_df = pd.read_csv("IST-data.CSV")

ist_df = ist_full_df[["SEX", "AGE", "RCONSC", "RSBP", "FDEAD", "RXASP", "RDATE", "COUNTRY"]]
ist_df = ist_df.rename(columns={
    "RSBP": "Systolic BP",
    "RCONSC": "Conscious Level",
    "RXASP": "Aspirin Allocated",
    "FDEAD": "Dead",
    "RDATE": "Randomisation month"
}).dropna()

sex_counts = ist_df[["SEX", "Aspirin Allocated"]].value_counts().reset_index()

fig_sex = px.bar(
    sex_counts,
    x="SEX",
    y="count",
    color="Aspirin Allocated",
    text="count",
    barmode='group',  # This puts bars side-by-side
    title="Number of men and women by aspirin allocation in IST dataset",
    color_discrete_map={
        'No Aspirin': 'red',
        'Aspirin': 'green'
    }
)
sex_data = ist_df["SEX"].value_counts().reset_index()

ist_df["SEX"] = ist_df["SEX"].map({"M": 0, "F": 1})
ist_df["Conscious Level"] = ist_df["Conscious Level"].map({"D": 0.5, "F": 1, "U": 0})
ist_df = ist_df[ist_df["Dead"].isin(["Y", "N"])]
ist_df["Dead"] = ist_df["Dead"].map({"N": 0, "Y": 1}).astype(int)
ist_df["Aspirin Allocated"] = ist_df["Aspirin Allocated"].map({"N": 0, "Y": 1})

triples_df = ist_df[["COUNTRY", "AGE", "Randomisation month", "Dead"]]
death_proportions = ist_df[[
    "COUNTRY",
    "AGE",
    "Randomisation month",
    "Dead"]
    ].groupby([
        'COUNTRY',
        'AGE',
        'Randomisation month'
        ]).agg(
    total_count=('Dead', 'count'),
    death_count=('Dead', 'sum'),
    death_proportion=('Dead', 'mean')
).reset_index()
death_proportions.sort_values("death_proportion", ascending=False).head()

mean_var = death_proportions.groupby(["COUNTRY", "AGE",]).agg(
    mean=("death_proportion", "mean"),
    variance=("death_proportion", "var")
).fillna(0).sort_values("mean", ascending=False)

# Fitting GLM

y = ist_df["Dead"]
covariates = ["AGE", "SEX", "Systolic BP", "Conscious Level", "Aspirin Allocated"]
X = ist_df[covariates]
X_const = sm.add_constant(X)
binomial_model = sm.GLM(y, X_const, family=sm.families.Binomial())

results = binomial_model.fit()

coef_table = results.params.to_frame(name="coef")
coef_table.index.name = "Variable"

latex_table = coef_table.to_latex(float_format="%.4f")