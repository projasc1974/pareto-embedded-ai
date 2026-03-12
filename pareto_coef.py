"""
Pareto Coefficient Estimation for Embedded AI Efficiency Model

This script estimates the coefficients of a log-linear regression model
that relates classification accuracy (A) with latency (L) and memory (M)
for Pareto-efficient configurations identified in the experimental results.

The regression follows the form:

    ln(A) = β0 + β1 ln(L) + β2 ln(M)

From this model, the efficiency equation can be expressed as:

    A = c * L^(β1) * M^(β2)

or equivalently:

    A = c * L^(-β) * M^(-γ)

where:
    c = exp(β0)
    β = -β1
    γ = -β2

The script also computes predictions, residuals, and statistical indicators
(R² and adjusted R²) to evaluate the goodness of fit of the model.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

# =========================
# Data: Table 6 (Pareto-efficient configurations)
# =========================
data = [
    {"Dataset": "Figures",         "Configuration": "MLR - ESP32-S3",   "A": 0.9467, "L": 0.1464, "M": 9.22},
    {"Dataset": "Figures",         "Configuration": "CNN - RPi Zero W", "A": 0.9400, "L": 1.4720, "M": 197.34},
    {"Dataset": "MNIST",           "Configuration": "MLR - ESP32-S3",   "A": 0.9267, "L": 5.0380, "M": 30.66},
    {"Dataset": "MNIST",           "Configuration": "CNN - RPi Zero W", "A": 0.9833, "L": 5.0740, "M": 197.34},
    {"Dataset": "Fashion-MNIST",   "Configuration": "MLR - ESP32-S3",   "A": 0.8633, "L": 5.0370, "M": 30.66},
    {"Dataset": "Fashion-MNIST",   "Configuration": "CNN - RPi Zero W", "A": 0.9067, "L": 1.5480, "M": 197.34},
]

df = pd.DataFrame(data)

# =========================
# Logarithmic transformation
# =========================
df["lnA"] = np.log(df["A"])
df["lnL"] = np.log(df["L"])
df["lnM"] = np.log(df["M"])

# =========================
# Regression model:
# ln(A) = beta0 + beta1*ln(L) + beta2*ln(M)
# =========================
X = df[["lnL", "lnM"]]
X = sm.add_constant(X)   # adds beta0 (intercept)
y = df["lnA"]

model = sm.OLS(y, X).fit()

# =========================
# Model coefficients
# =========================
beta0 = model.params["const"]
beta1 = model.params["lnL"]
beta2 = model.params["lnM"]
r2 = model.rsquared
adj_r2 = model.rsquared_adj

# Parameters for the form A = c * L^(beta1) * M^(beta2)
# If using the form A = c * L^(-β) * M^(-γ),
# then β = -beta1 and γ = -beta2
c = np.exp(beta0)
beta = -beta1
gamma = -beta2

print("\n=== LOG-LINEAR MODEL RESULTS ===")
print(f"beta0 = {beta0:.6f}")
print(f"beta1 = {beta1:.6f}")
print(f"beta2 = {beta2:.6f}")
print(f"c     = {c:.6f}")
print(f"beta  = {-beta1:.6f}")   # for A = c * L^(-beta) * M^(-gamma)
print(f"gamma = {-beta2:.6f}")
print(f"R^2   = {r2:.6f}")
print(f"Adj R^2 = {adj_r2:.6f}")

print("\n=== FITTED EQUATION ===")
print(f"ln(A) = {beta0:.6f} + ({beta1:.6f}) ln(L) + ({beta2:.6f}) ln(M)")
print(f"A = {c:.6f} * L^({beta1:.6f}) * M^({beta2:.6f})")
print(f"A = {c:.6f} * L^(-{beta:.6f}) * M^(-{gamma:.6f})")

# =========================
# Predictions and residuals
# =========================
df["lnA_pred"] = model.predict(X)
df["A_pred"] = np.exp(df["lnA_pred"])
df["Residual_ln"] = df["lnA"] - df["lnA_pred"]
df["Residual_A"] = df["A"] - df["A_pred"]

print("\n=== FIT TABLE ===")
print(df[[
    "Dataset", "Configuration", "A", "L", "M",
    "lnA", "lnA_pred", "A_pred", "Residual_ln", "Residual_A"
]].to_string(index=False))

print("\n=== STATISTICAL SUMMARY ===")
print(model.summary())

# =========================
# Correlation between ln(L) and ln(M)
# =========================
corr = df["lnL"].corr(df["lnM"])
print(f"\nCorrelation between ln(L) and ln(M): r = {corr:.6f}")
