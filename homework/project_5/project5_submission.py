#!/usr/bin/env python
# coding: utf-8

# # Project 5
# 
# In this exercise, you will use **bootstrap** and **cross-validation** to estimate statistical quantities for two datasets.
# 
# This version uses:
# - **public local checks** for immediate feedback,
# - a **Gradescope `.py` submission** for official scoring, and
# - synthetic public-check data that is unrelated to the real assignment datasets.
# 
# The public checker does **not** contain the reference implementation or the real Project 5 answers. Gradescope uses separate hidden data.
# 
# ## Submission
# 
# Complete the notebook normally. At the end, use the provided conversion function to create a `.py` version of your notebook and upload that `.py` file to Gradescope.
# 

# ## Getting Setup
# 

# In[1]:


import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold

RANDOM_STATE = 42

# Notebook-friendly equivalent of %matplotlib inline.
try:
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass

print("Python:", sys.version.split()[0])
print("Pandas:", pd.__version__)


# ## Public checker
# 
# The checker below only verifies your functions on unrelated synthetic examples. A passing public check is useful feedback, but it is **not** your official Project 5 score.
# 
# The expected outputs are stored as opaque fingerprints rather than visible values.
# 

# In[2]:


# -----------------------------------------------------------------------------
# Public checks
# -----------------------------------------------------------------------------
# These checks use synthetic data unrelated to the assignment datasets.
# Expected outputs are stored only as opaque fingerprints. No reference
# implementation or real Project 5 answer is embedded in this notebook.

import hashlib
import json

_PUBLIC_RESULTS = {}

def _normalize_public_value(value):
    if isinstance(value, (np.floating, float)):
        return {"scalar": round(float(value), 7)}
    if isinstance(value, (np.integer, int)):
        return {"scalar": int(value)}
    if isinstance(value, tuple):
        return {"tuple": [_normalize_public_value(v) for v in value]}
    if isinstance(value, list):
        return {"list": [_normalize_public_value(v) for v in value]}
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.number):
            return {
                "array": np.round(arr.astype(float), 6).tolist(),
                "shape": list(arr.shape),
            }
        return {"array": arr.astype(str).tolist(), "shape": list(arr.shape)}
    if isinstance(value, pd.Series):
        vals = value.to_numpy()
        if np.issubdtype(vals.dtype, np.number):
            vals = np.round(vals.astype(float), 6)
        else:
            vals = vals.astype(str)
        return {
            "series_index": [str(x) for x in value.index],
            "series_values": vals.tolist(),
        }
    if hasattr(value, "params"):
        return _normalize_public_value(value.params)
    if isinstance(value, dict):
        return {
            str(k): _normalize_public_value(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    return {"repr": repr(value)}

def _public_digest(value):
    payload = json.dumps(
        _normalize_public_value(value),
        sort_keys=True,
        separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

_PUBLIC_HASHES = {
    "estimate_mean": "c18429bda09a34b80c09fbf32134fbbff033f3b5f923423d759f00fa90cafd23",
    "estimate_mean_se": "c2f8656704351929533967494fe2c70ba6be5e2000897221ad7154868abad413",
    "bootstrap_mean_se": "c39b9eaa47c97b7065efda50fd82c42ab1c0f26b2dd3497db88a98b4c82c7610",
    "mean_confidence_interval": "28d5a075c9ba401f1fec07c216a151a6bc9761969f7665692d5386f474651cc0",
    "estimate_median": "7c3300ee5dffe992d6964f1615869824f03a523cdb928aa46686d7dbc92da934",
    "bootstrap_median_se": "0ace7163cdbc0cc38f3ab8839fd78f474fee7f522d51090d70583b557abf26f1",
    "fit_default_logit": "5a0da0f808f9755baf5aeb09b3e59089f4a3b284cfc6be3935b84fb0d541402c",
    "validation_misclassification": "a24d91cdec593c63cc945a225d00c0b497c1f5c301a45460983a6bf3ab51c539",
    "repeated_validation": "1be79663e28c04e9d5dca7f7c38724cba4d0dbf369048a899d28eaf543a69c91",
    "kfold_misclassification": "74418ffcf4a19092fe796c35ad4e2a8e89e6a7366d134dd95aaaa6c922c0a1ff",
    "boot_sample": "d9613375d9226d273b98f0e5673a4faec0218d52c7a4b7e4d41d7fbf88ac79eb",
    "bootstrap_coefficient_estimates": "4462eb80feae4b416a5121edb4eb7e62a9cf1ce1f9e09979bf60b7ffd204e1ae"
}

def _public_result(label, passed, hint=""):
    _PUBLIC_RESULTS[label] = bool(passed)
    if passed:
        print(f"[PASS] {label}")
    else:
        print(f"[FAIL] {label}")
        if hint:
            print(hint)
    return bool(passed)

def _fingerprint_check(label, value, hint):
    try:
        passed = _public_digest(value) == _PUBLIC_HASHES[label]
    except Exception:
        passed = False
    return _public_result(label, passed, hint)

# Public synthetic data. These values are unrelated to the real homework data.
_PUBLIC_HOUSE = pd.DataFrame({
    "MedHouseVal": [1.2, 1.8, 2.4, 2.4, 3.0, 3.6, 4.2, 4.8]
})

_public_rng = np.random.RandomState(123)
_public_n = 60
_public_income = _public_rng.normal(45000, 12000, _public_n)
_public_balance = _public_rng.normal(900, 350, _public_n)
_public_student = np.where(_public_rng.rand(_public_n) < 0.40, "Yes", "No")
_public_linear = -4.0 + 0.000025 * _public_income + 0.0025 * _public_balance + 0.15 * (_public_student == "Yes")
_public_probability = 1.0 / (1.0 + np.exp(-_public_linear))
_public_default = np.where(_public_rng.rand(_public_n) < _public_probability, "Yes", "No")

_PUBLIC_DEFAULT = pd.DataFrame({
    "default": _public_default,
    "student": _public_student,
    "income": _public_income,
    "balance": _public_balance,
})

def check_estimate_mean():
    try:
        value = estimate_mean(_PUBLIC_HOUSE.copy())
    except Exception as exc:
        return _public_result("estimate_mean", False, f"Your function raised {type(exc).__name__}.")
    return _fingerprint_check(
        "estimate_mean", value,
        "The public example does not match. Check the returned scalar and calculation."
    )

def check_estimate_mean_se():
    try:
        value = estimate_mean_se(_PUBLIC_HOUSE.copy())
    except Exception as exc:
        return _public_result("estimate_mean_se", False, f"Your function raised {type(exc).__name__}.")
    return _fingerprint_check(
        "estimate_mean_se", value,
        "The public example does not match. Check the standard-error calculation and return type."
    )

def check_bootstrap_mean_se():
    try:
        value = bootstrap_mean_se(_PUBLIC_HOUSE.copy(), random_state=7)
    except Exception as exc:
        return _public_result("bootstrap_mean_se", False, f"Your function raised {type(exc).__name__}.")
    return _fingerprint_check(
        "bootstrap_mean_se", value,
        "The bootstrap public example does not match. Check sampling, replacement, repetitions, and returned scalar."
    )

def check_mean_confidence_interval():
    try:
        value = mean_confidence_interval(_PUBLIC_HOUSE.copy(), random_state=7)
    except Exception as exc:
        return _public_result("mean_confidence_interval", False, f"Your function raised {type(exc).__name__}.")
    return _fingerprint_check(
        "mean_confidence_interval", value,
        "The public interval does not match. Return a two-value interval in lower, upper order."
    )

def check_estimate_median():
    try:
        value = estimate_median(_PUBLIC_HOUSE.copy())
    except Exception as exc:
        return _public_result("estimate_median", False, f"Your function raised {type(exc).__name__}.")
    return _fingerprint_check(
        "estimate_median", value,
        "The public median example does not match."
    )

def check_bootstrap_median_se():
    try:
        value = bootstrap_median_se(_PUBLIC_HOUSE.copy(), random_state=7)
    except Exception as exc:
        return _public_result("bootstrap_median_se", False, f"Your function raised {type(exc).__name__}.")
    return _fingerprint_check(
        "bootstrap_median_se", value,
        "The bootstrap-median public example does not match."
    )

def check_fit_default_logit():
    try:
        model = fit_default_logit(_PUBLIC_DEFAULT.copy())
        value = model.params
    except Exception as exc:
        return _public_result("fit_default_logit", False, f"Your function raised {type(exc).__name__}.")
    return _fingerprint_check(
        "fit_default_logit", value,
        "The fitted model does not match the public behavior check."
    )

def check_validation_misclassification():
    try:
        value = validation_misclassification(
            _PUBLIC_DEFAULT.copy(),
            test_size=0.20,
            random_state=42
        )
    except Exception as exc:
        return _public_result("validation_misclassification", False, f"Your function raised {type(exc).__name__}.")
    return _fingerprint_check(
        "validation_misclassification", value,
        "The validation-set behavior does not match the public check."
    )

def check_repeated_validation():
    try:
        rates, params = repeated_validation(
            _PUBLIC_DEFAULT.copy(),
            test_sizes=(0.10, 0.30, 0.50),
            random_state=42
        )
        value = {"rates": rates, "params": params}
    except Exception as exc:
        return _public_result("repeated_validation", False, f"Your function raised {type(exc).__name__}.")
    return _fingerprint_check(
        "repeated_validation", value,
        "The repeated-validation behavior does not match the public check."
    )

def check_kfold_misclassification():
    try:
        value = kfold_misclassification(_PUBLIC_DEFAULT.copy(), n_splits=5, trials=5)
    except Exception as exc:
        return _public_result("kfold_misclassification", False, f"Your function raised {type(exc).__name__}.")
    return _fingerprint_check(
        "kfold_misclassification", value,
        "The K-fold public behavior does not match."
    )

def check_boot_sample():
    try:
        index = np.arange(len(_PUBLIC_DEFAULT))[::-1]
        value = boot_sample(_PUBLIC_DEFAULT.copy(), index)
    except Exception as exc:
        return _public_result("boot_sample", False, f"Your function raised {type(exc).__name__}.")
    return _fingerprint_check(
        "boot_sample", value,
        "The bootstrap-coefficient function does not match the public behavior check."
    )

def check_bootstrap_coefficient_estimates():
    try:
        income, balance = bootstrap_coefficient_estimates(
            _PUBLIC_DEFAULT.copy(),
            n_boot=20,
            random_state=7
        )
        value = {"income": income, "balance": balance}
    except Exception as exc:
        return _public_result("bootstrap_coefficient_estimates", False, f"Your function raised {type(exc).__name__}.")
    return _fingerprint_check(
        "bootstrap_coefficient_estimates", value,
        "The bootstrap coefficient estimates do not match the public behavior check."
    )

def show_public_check_summary():
    if not _PUBLIC_RESULTS:
        print("No public checks have been run yet.")
        return
    passed = sum(_PUBLIC_RESULTS.values())
    total = len(_PUBLIC_RESULTS)
    print(f"Public checks passed: {passed}/{total}")
    print("These are practice checks only. Gradescope uses different hidden data.")


# ## Graded function definitions
# 

# In[3]:


def estimate_mean(df):
    """Return the estimated mean of MedHouseVal as a scalar."""
    return df["MedHouseVal"].mean()

def estimate_mean_se(df):
    """Return the standard error of the estimated mean as a scalar."""
    return df["MedHouseVal"].std(ddof=1) / np.sqrt(len(df))
    raise NotImplementedError

def bootstrap_mean_se(df, random_state=RANDOM_STATE):
    """Estimate the standard error of the mean using bootstrap."""
    random_state = np.random.RandomState(random_state)
    values = df["MedHouseVal"].to_numpy()
    boot_means = []
    for _ in range(len(values)):
        sample = random_state.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))
    return np.std(boot_means, ddof=1)

def mean_confidence_interval(df, random_state=RANDOM_STATE):
    """Return (lower, upper) for the requested 95% bootstrap-based interval."""
    se = bootstrap_mean_se(df, random_state=random_state)
    mu_hat = estimate_mean(df)
    lower = mu_hat - 2 * se
    upper = mu_hat + 2 * se
    return lower, upper

def estimate_median(df):
    """Return the estimated median of MedHouseVal as a scalar."""
    return df["MedHouseVal"].median()

def bootstrap_median_se(df, random_state=RANDOM_STATE):
    """Estimate the standard error of the median using bootstrap."""
    random_state = np.random.RandomState(random_state)
    values = df["MedHouseVal"].to_numpy()
    boot_medians = []
    for _ in range(len(values)):
        sample = random_state.choice(values, size=len(values), replace=True)
        boot_medians.append(np.median(sample))
    return np.std(boot_medians, ddof=1)

def _encode_default_columns(df):
    """Return a copy with binary Default-data columns encoded numerically."""
    df = df.copy()
    df["default"] = df["default"].map({"No": 0, "Yes": 1})
    df["student"] = df["student"].map({"No": 0, "Yes": 1})
    return df

def fit_default_logit(df):
    """Fit the requested logistic model using income and balance."""
    return smf.logit("default ~ income + balance", data=_encode_default_columns(df)).fit(disp=False)

def split_default_data(df, test_size=0.20, random_state=RANDOM_STATE):
    """Return (train, test) using the requested split settings."""
    return train_test_split(df, test_size=test_size, random_state=random_state)

def predict_default(model, test_df):
    """Return binary predictions for the supplied test observations."""
    prob = model.predict(_encode_default_columns(test_df))
    return (prob >= 0.5).astype(int).to_numpy()


def misclassification_rate(predicted, actual):
    """Return the fraction of observations that are misclassified."""
    return np.mean(predicted != actual)

def validation_misclassification(df, test_size=0.20, random_state=RANDOM_STATE):
    """Return the validation-set misclassification rate."""
    train, test = split_default_data(df, test_size=test_size, random_state=random_state)
    model = fit_default_logit(train)
    predicted = predict_default(model, test)
    actual = _encode_default_columns(test)["default"].to_numpy()
    return misclassification_rate(predicted, actual)

def repeated_validation(df, test_sizes=(0.10, 0.30, 0.50), random_state=RANDOM_STATE):
    """Return (model_rates, model_params) for the three requested split sizes."""
    model_rates = []
    model_params = []
    for test_size in test_sizes:
        rate = validation_misclassification(df, test_size=test_size, random_state=random_state)
        train, test = split_default_data(df, test_size=test_size, random_state=random_state)
        model = fit_default_logit(train)
        model_rates.append(rate)
        model_params.append(model.params)
    return model_rates, model_params

def kfold_misclassification(df, n_splits=5, trials=5):
    """Return the average misclassification rate across the requested K-fold trials."""
    rates = []
    for trial in range(trials):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=trial)
        trial_rates = []
        for train_index, test_index in kf.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            model = fit_default_logit(train)
            predicted = predict_default(model, test)
            actual = _encode_default_columns(test)["default"].to_numpy()
            trial_rates.append(misclassification_rate(predicted, actual))
        rates.append(np.mean(trial_rates))
    return np.mean(rates)

def boot_sample(data, index):
    """Return the income and balance coefficient estimates for one bootstrap sample."""
    model = fit_default_logit(data.iloc[index])
    return model.params[["income", "balance"]].to_numpy()

def bootstrap_coefficient_estimates(data, n_boot=100, random_state=None):
    """Return (boot_income, boot_balance), each shaped (n_boot, 1)."""

    random_state = np.random.RandomState(random_state)
    n = len(data)
    boot_income = []
    boot_balance = []
    for _ in range(n_boot):
        index = random_state.choice(n, size=n, replace=True)
        income, balance = boot_sample(data, index)
        boot_income.append(income)
        boot_balance.append(balance)
    return np.array(boot_income).reshape(-1, 1), np.array(boot_balance).reshape(-1, 1)


# ## Part A — Bootstrap with the California housing dataset
# 
# We will use a 500-observation subset of the California housing dataset.
# 

# In[4]:


# DO NOT CHANGE
california_dataset = fetch_california_housing()
data = pd.DataFrame(
    data=california_dataset.data,
    columns=california_dataset.feature_names
)
data["MedHouseVal"] = pd.Series(california_dataset.target)
data = data.sample(n=500, random_state=RANDOM_STATE)
data.head()


# ### A1. Estimate the population mean
# 
# Provide an estimate for \(\hat\mu\), the population mean of `MedHouseVal`. Store the real-data answer as `mu_hat`.
# 

# In[5]:


mu_hat = estimate_mean(data)
print(mu_hat)
check_estimate_mean()


# ### A2. Standard error of the estimated mean
# 
# Estimate the standard error of \(\hat\mu\). Remember that the sample standard deviation uses Bessel's correction. Store the real-data answer as `se_mu_hat`.
# 
# Interpret the result in a Markdown cell.
# 

# In[6]:


se_mu_hat = estimate_mean_se(data)
print(se_mu_hat)
check_estimate_mean_se()


# ### A3. Bootstrap standard error of the mean
# 
# Estimate the standard error of the mean using your own bootstrap implementation.
# 
# Use \(M=n\) artificial datasets, each containing \(N=n\) bootstrap observations. Use `RANDOM_STATE` so your work is reproducible. Store the answer as `se_bootstrap`.
# 
# Compare this estimate with A2 in a Markdown response.
# 

# In[7]:


se_bootstrap = bootstrap_mean_se(data, random_state=RANDOM_STATE)
print(se_bootstrap)
check_bootstrap_mean_se()


# ### A4. 95% confidence interval
# 
# Using your bootstrap estimate from A3, construct the requested approximate 95% confidence interval for the mean:
# 
# \[
# [\hat\mu - 2SE(\hat\mu),\ \hat\mu + 2SE(\hat\mu)]
# \]
# 
# Store the endpoints as `CI_lower` and `CI_upper`.
# 

# In[8]:


CI_lower, CI_upper = mean_confidence_interval(
    data,
    random_state=RANDOM_STATE
)
print(CI_lower, CI_upper)
check_mean_confidence_interval()


# ### A5. Estimate the population median
# 
# Estimate the median of `MedHouseVal`. Store it as `median_hat`.
# 

# In[9]:


median_hat = estimate_median(data)
print(median_hat)
check_estimate_median()


# ### A6. Bootstrap standard error of the median
# 
# There is no simple standard-error formula for the median here. Estimate it using bootstrap and store the result as `se_bootstrap_median`.
# 

# In[10]:


se_bootstrap_median = bootstrap_median_se(
    data,
    random_state=RANDOM_STATE
)
print(se_bootstrap_median)
check_bootstrap_median_se()


# In[11]:


print(se_bootstrap_median > se_bootstrap)
print(se_bootstrap_median < se_bootstrap)
print(se_bootstrap_median - se_bootstrap)


# ### A7. Interpretation
# 
# Is the estimated standard error of the median greater than or less than the estimated standard error of the mean? Does that make intuitive sense? Comment on your other findings below.
# 

# **Your response:**
# 
# median is greater than mean by approximately 0.007942380591433255. It does make sense because the mean uses all values, while the median depends primarily on the middle values.
# 

# ## Part B — Validation, K-fold cross-validation, and bootstrap with `Default.csv`
# 
# Load `Default.csv`. The dataset should contain `default`, `student`, `balance`, and `income`.
# 
# Before fitting logistic models, encode `No = 0` and `Yes = 1` for the binary columns as specified in the assignment.
# 

# In[12]:


default = pd.read_csv("Default.csv")
default.head()


# ### B1. Fit the full logistic regression model
# 
# Using **all** observations, fit a logistic regression model using `income` and `balance` to predict `default`. Store the real-data model as `fit1`.
# 

# In[13]:


fit1 = fit_default_logit(default)
print(fit1.summary())
check_fit_default_logit()


# ### B2. Validation-set approach
# 
# Split the observations into training and validation sets using:
# - test size = 20%
# - `random_state = RANDOM_STATE`
# 
# Fit the logistic model on the **training observations only**, classify validation observations using a probability threshold of 0.5, and calculate the validation misclassification rate.
# 

# In[14]:


train, test = split_default_data(
    default,
    test_size=0.20,
    random_state=RANDOM_STATE
)

fit2 = fit_default_logit(train)
predicted = predict_default(fit2, test)
mis_rate = misclassification_rate(
    predicted,
    _encode_default_columns(test)["default"].to_numpy()
)

print("train shape:", train.shape)
print("test shape:", test.shape)
print("misclassification rate:", mis_rate)

# The combined function is what Gradescope tests directly.
print(
    "combined validation rate:",
    validation_misclassification(
        default,
        test_size=0.20,
        random_state=RANDOM_STATE
    )
)
check_validation_misclassification()


# ### B3. Repeat the validation-set approach
# 
# Repeat B2 using test sizes of 10%, 30%, and 50%, keeping `random_state = RANDOM_STATE`.
# 
# Store:
# - the three misclassification rates in `model_rates`
# - the fitted model parameters in `model_params`
# 
# Comment on how changing the validation-set size affects your results.
# 

# In[15]:


model_rates, model_params = repeated_validation(
    default,
    test_sizes=(0.10, 0.30, 0.50),
    random_state=RANDOM_STATE
)

mis_rate_10, mis_rate_30, mis_rate_50 = model_rates

print("model_rates:", model_rates)
check_repeated_validation()


# ### B4. Five-fold cross-validation across five trials
# 
# Using five-fold cross-validation across five trials, calculate the **average misclassification rate**.
# 
# Use trial numbers `0, 1, 2, 3, 4` as the K-fold random states. Store the result as `mis_rate_kfold`.
# 
# Briefly explain the K-fold process in a Markdown response.
# 

# In[16]:


mis_rate_kfold = kfold_misclassification(
    default,
    n_splits=5,
    trials=5
)
print(mis_rate_kfold)
check_kfold_misclassification()


# ### B5. Bootstrap logistic-regression coefficients
# 
# `boot_sample(data, index)` should return the coefficient estimates for:
# 1. `income`
# 2. `balance`
# 
# for the observations specified by `index`.
# 
# Use the function to generate 100 bootstrap estimates for each coefficient. Store them in:
# - `boot_income`
# - `boot_balance`
# 
# Each should have shape `(100, 1)`.
# 
# For the real assignment run, do **not** set a random seed for the 100-sample bootstrap.
# 

# In[17]:


# Public behavior check for one explicitly specified sample.
check_boot_sample()

boot_income, boot_balance = bootstrap_coefficient_estimates(
    default,
    n_boot=100,
    random_state=None
)

print("boot_income shape:", boot_income.shape)
print("boot_balance shape:", boot_balance.shape)

# The public checker uses a separate deterministic call so it can provide feedback.
check_bootstrap_coefficient_estimates()

income_bootstrap_se = float(np.std(boot_income, ddof=1))
balance_bootstrap_se = float(np.std(boot_balance, ddof=1))

print("Bootstrap SE for income coefficient:", income_bootstrap_se)
print("Bootstrap SE for balance coefficient:", balance_bootstrap_se)


# ### B6. Interpretation
# 
# Comment on the estimated standard errors and construct confidence intervals for the logistic-regression coefficients as requested by your course instructions.
# 
# **Your response:**
# 
# - Both confidence intervals exclude zero, suggesting that income and balance are statistically significant predictors of default.
# - Both coefficients are positive, indicating that higher income and higher balance are associated with higher predicted probabilities of default
# - The bootstrap results are consistent with the logistic regression model
# 

# ## Public-check summary
# 

# In[18]:


show_public_check_summary()


# ## Convert your completed notebook to `.py` for Gradescope
# 
# Run the final cell **after saving your notebook**.
# 
# The function:
# 1. converts an `.ipynb` file to Python using `nbconvert`,
# 2. checks that the generated `.py` is syntactically valid, and
# 3. automatically downloads the `.py` file when running in Google Colab.
# 
# If the notebook file is not available in the Colab runtime, the function opens a file picker so you can select the saved `.ipynb`.
# 

# In[19]:


from pathlib import Path
import subprocess
import sys

def download_notebook_as_py(
    ipynb_path="project5_student.ipynb",
    output_name="project5_submission.py"
):
    """
    Convert a Jupyter notebook to a .py file, verify the result,
    and download it automatically in Google Colab.

    If the notebook is not present in the current runtime, Colab
    will ask you to select the saved .ipynb file.
    """
    ipynb_path = Path(ipynb_path)

    # In Colab, the currently open notebook is not always stored
    # as a normal file in /content. If it is missing, allow the
    # student to select the saved notebook.
    if not ipynb_path.exists():
        try:
            from google.colab import files
            print(
                f"Could not find {ipynb_path.name} in the runtime.\n"
                "Select your saved .ipynb notebook in the file picker."
            )
            uploaded = files.upload()
            notebook_names = [
                name for name in uploaded
                if name.lower().endswith(".ipynb")
            ]
            if len(notebook_names) != 1:
                raise ValueError(
                    "Please upload exactly one .ipynb notebook."
                )
            ipynb_path = Path(notebook_names[0])
        except ImportError:
            raise FileNotFoundError(
                f"Could not find notebook: {ipynb_path}"
            )

    if ipynb_path.suffix.lower() != ".ipynb":
        raise ValueError("The input file must be a .ipynb file.")

    if not output_name.lower().endswith(".py"):
        output_name += ".py"

    output_path = Path(output_name)

    # Convert the notebook.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "python",
            str(ipynb_path),
            "--output",
            output_path.stem,
            "--output-dir",
            str(output_path.parent),
        ],
        check=True,
    )

    # Verify that the generated file is syntactically valid Python.
    source = output_path.read_text(encoding="utf-8")
    compile(source, str(output_path), "exec")

    print(f"Created and verified: {output_path.resolve()}")
    print("Upload this .py file to Gradescope.")

    # Download automatically in Colab.
    try:
        from google.colab import files
        files.download(str(output_path))
    except ImportError:
        print("Not running in Google Colab; the file remains on disk.")

    return str(output_path)

# After saving this notebook, run:
download_notebook_as_py()

