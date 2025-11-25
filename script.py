import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib as plt
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

# Split data
bp_N = ist_df[ist_df["Aspirin Allocated"] == "N"]["Systolic BP"]
bp_Y = ist_df[ist_df["Aspirin Allocated"] == "Y"]["Systolic BP"]

# Create boxplot
plt.figure(figsize=(6, 5))
plt.boxplot([bp_N, bp_Y], labels=["No Aspirin (N)", "Aspirin (Y)"])

plt.title("Systolic BP by Aspirin Allocation")
plt.ylabel("Systolic Blood Pressure (mmHg)")
plt.grid(axis='y')

plt.show()

# Count records by AGE + Aspirin allocation
age_counts = ist_df.groupby(["AGE", "Aspirin Allocated"]).size().unstack(fill_value=0)

# Prepare data
ages = age_counts.index
aspirin = age_counts["Y"]
no_aspirin = age_counts["N"]

# Plot
plt.figure(figsize=(10, 5))

plt.bar(ages, no_aspirin, color="red", label="No Aspirin")
plt.bar(ages, aspirin, bottom=no_aspirin, color="green", label="Aspirin")

plt.title("Distribution of age by aspirin allocation")
plt.xlabel("AGE")
plt.ylabel("Count")
plt.legend()

plt.show()


plt.figure(figsize=(8,5))
plt.scatter(mean_var["mean"], mean_var["variance"], alpha=0.7, label="Observed")

# Add theoretical binomial curve y = p(1-p)
p = np.linspace(0, 1, 300)
plt.plot(p, p*(1-p), color="red", linewidth=2, label="Binomial variance: p(1-p)")


plt.xlabel("Mean")
plt.ylabel("Variance")
plt.title("Mean vs Variance")
plt.grid(True)
plt.show()