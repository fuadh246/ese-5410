# %% [markdown]
# # Project 3
# 
# In the coding project below, answer relevant questions on Canvas via the assignment named **Programming Project 3 Quiz Questions**.
# 
# In this exercise, you will perform linear regression to fit several datasets and make predictions. The training and test data must remain separate when fitting models.
# 
# ## Local checks in this notebook
# This version uses **public local checks** instead of public checker. The public checks are designed to tell you whether your code behaves correctly on small synthetic examples **without revealing the solution method or the answers for the real Project 3 datasets**.
# 
# - Public checks contain only small fake inputs and precomputed expected outputs.
# - They do **not** contain the correct values from `Ch3PartA.csv` or `Ch3PartB.csv`.
# - They do **not** contain reference solution code.
# - Written-response checks only verify completeness/format; they do not reveal the correct interpretation.
# - Passing the public checks is useful evidence that your implementation is on the right track, but your actual assignment answers still come from running your code on the real Project 3 datasets.
# 
# > Keep each function syntactically valid while you work. Replace `raise NotImplementedError` when you implement it.
# 
# %% [markdown]
# ## Getting Setup
# Run the cells below to import the packages used in Project 3 and initialize the public local checker.
# 
# %%
# If needed, install dependencies in your environment.
# !pip install pandas numpy scikit-learn statsmodels matplotlib

# %% [markdown]
# The local checks below are intentionally **not** an answer key. They exercise your functions on unrelated synthetic data.
# 
# %% [markdown]
# You do not need to enter a PennID for the local checker. Your name/ID can still be recorded below if your course workflow requires it.
# 
# %%
STUDENT_ID = 88888888                   # YOUR 8-DIGIT PENNID GOES HERE
STUDENT_NAME = "Fuad Hassan"     # YOUR FULL NAME GOES HERE

# %%
# Data Wrangling
import pandas as pd
import numpy as np

# ML
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Statistics
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Evaluation Metrics
from sklearn.metrics import mean_squared_error

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass

# %%
# -----------------------------------------------------------------------------
# PUBLIC LOCAL CHECKER
# -----------------------------------------------------------------------------
# These checks use synthetic inputs and precomputed outputs only.
# There is no reference-solution implementation and no answer from the real
# Project 3 datasets in this cell.

import math

_PUBLIC_RESULTS = {}

def _public_result(name, passed, message=""):
    _PUBLIC_RESULTS[name] = bool(passed)
    tag = "PASS" if passed else "FAIL"
    suffix = f" — {message}" if message else ""
    print(f"[{tag}] {name}{suffix}")
    return bool(passed)

def _close(a, b, atol=1e-7, rtol=1e-6):
    try:
        return bool(np.isclose(float(a), float(b), atol=atol, rtol=rtol))
    except Exception:
        return False

def check_max_y_function(func):
    fake = pd.DataFrame({
        "x_tr": [-2.0, 0.0, 2.0, 4.0],
        "y_tr": [-8.0, 3.5, 11.0, 7.0],
        "x_te": [-1.0, 1.0, 3.0, 5.0],
        "y_te": [4.0, 9.25, 2.0, 8.0],
    })
    try:
        actual = func(fake)
        ok = isinstance(actual, (tuple, list)) and len(actual) == 2 and _close(actual[0], 11.0) and _close(actual[1], 9.25)
        return _public_result("A2 maximum-y logic", ok, "Expected a two-value result on the public example.")
    except Exception as exc:
        return _public_result("A2 maximum-y logic", False, f"{type(exc).__name__}: {exc}")

def check_polynomial_features_function(func):
    fake_x = pd.Series([2.0, -1.0], name="x")
    expected = np.array([[1.0, 2.0, 4.0, 8.0], [1.0, -1.0, 1.0, -1.0]])
    try:
        actual = func(fake_x, degree=3)
        arr = np.asarray(actual, dtype=float)
        ok = arr.shape == expected.shape and np.allclose(arr, expected)
        return _public_result("A3 polynomial-feature logic", ok, "Public check uses degree 3 on two synthetic x values.")
    except Exception as exc:
        return _public_result("A3 polynomial-feature logic", False, f"{type(exc).__name__}: {exc}")

def check_polynomial_mse_function(func):
    fake = pd.DataFrame({
        "x_tr": [-2., -1., 0., 1., 2.],
        "y_tr": [5., 2., 1., 2., 5.],
        "x_te": [-1.5, -0.5, 0.5, 1.5, 2.5],
        "y_te": [3.25, 1.25, 1.25, 3.25, 7.25],
    })
    expected_train = np.array([2.8, 0.0, 0.0])
    expected_test = np.array([4.8625, 0.0, 0.0])
    try:
        actual = func(fake, max_degree=3)
        ok_shape = isinstance(actual, (tuple, list)) and len(actual) == 2
        if not ok_shape:
            return _public_result("A4 polynomial-MSE logic", False, "Return (mse_train, mse_test).")
        tr = np.asarray(actual[0], dtype=float)
        te = np.asarray(actual[1], dtype=float)
        ok = tr.shape == (3,) and te.shape == (3,) and np.allclose(tr, expected_train, atol=1e-8) and np.allclose(te, expected_test, atol=1e-8)
        return _public_result("A4 polynomial-MSE logic", ok, "Public check uses only degrees 1–3 on synthetic data.")
    except Exception as exc:
        return _public_result("A4 polynomial-MSE logic", False, f"{type(exc).__name__}: {exc}")

def check_min_mse_function(func):
    try:
        actual = func([7.0, 3.0, 4.0], [8.0, 5.5, 6.0])
        ok = isinstance(actual, (tuple, list)) and len(actual) == 2 and _close(actual[0], 3.0) and _close(actual[1], 5.5)
        return _public_result("A5 minimum-MSE logic", ok)
    except Exception as exc:
        return _public_result("A5 minimum-MSE logic", False, f"{type(exc).__name__}: {exc}")

def check_a6_format(degree, rse_train_sq, rse_test_sq):
    ok = isinstance(degree, (int, np.integer)) and not isinstance(degree, bool)
    ok = ok and 1 <= int(degree) <= 20
    ok = ok and all(isinstance(v, (int, float, np.number)) and np.isfinite(v) and v >= 0 for v in (rse_train_sq, rse_test_sq))
    return _public_result("A6 answer format", ok, "This check validates only type/range, not the correct Project 3 answer.")

def check_written_response(text, label, min_chars=40):
    ok = isinstance(text, str) and len(text.strip()) >= min_chars
    return _public_result(label, ok, f"Write at least {min_chars} non-whitespace characters. Correctness is not checked locally.")

def check_correlation_function(func):
    fake = pd.DataFrame({"x1": [0., 1., 2., 3., 4.], "x2": [0., 2., 1., 4., 3.], "y": [1., 2., 3., 4., 5.]})
    try:
        actual = func(fake)
        return _public_result("B2 correlation logic", _close(actual, 0.8), "Public synthetic correlation should be returned as one scalar.")
    except Exception as exc:
        return _public_result("B2 correlation logic", False, f"{type(exc).__name__}: {exc}")

def _check_model_params(model, expected, label):
    try:
        params = model.params
        vals = np.asarray(params, dtype=float)
        ok = vals.shape == np.asarray(expected).shape and np.allclose(vals, np.asarray(expected), atol=1e-5, rtol=1e-5)
        return _public_result(label, ok, "Model coefficients do not match the public synthetic example." if not ok else "")
    except Exception as exc:
        return _public_result(label, False, f"{type(exc).__name__}: {exc}")

def _public_b_df():
    return pd.DataFrame({
        "x1": [0.,1.,2.,3.,4.,5.,6.,7.],
        "x2": [1.,0.,2.,1.,3.,5.,4.,6.],
        "y":  [0.4,3.1,4.2,6.7,7.6,8.3,11.2,12.9],
    })

def check_multiple_regression_function(func):
    try:
        model = func(_public_b_df())
        return _check_model_params(model, [0.9166666667, 2.1141025641, -0.5512820513], "B3 multiple-regression logic")
    except Exception as exc:
        return _public_result("B3 multiple-regression logic", False, f"{type(exc).__name__}: {exc}")

def check_x1_regression_function(func):
    try:
        model = func(_public_b_df())
        return _check_model_params(model, [0.9166666667, 1.6809523810], "B4 x1-only regression logic")
    except Exception as exc:
        return _public_result("B4 x1-only regression logic", False, f"{type(exc).__name__}: {exc}")

def check_x2_regression_function(func):
    try:
        model = func(_public_b_df())
        return _check_model_params(model, [2.2253968254, 1.6634920635], "B5 x2-only regression logic")
    except Exception as exc:
        return _public_result("B5 x2-only regression logic", False, f"{type(exc).__name__}: {exc}")

def check_hypothesis_format(test_statistic, reject_null, label, significant=None):
    numeric = isinstance(test_statistic, (int, float, np.number)) and np.isfinite(test_statistic)
    boolean = isinstance(reject_null, (bool, np.bool_))
    if significant is not None:
        boolean = boolean and isinstance(significant, (bool, np.bool_))
    return _public_result(label, numeric and boolean, "This check validates only types; it does not reveal the correct hypothesis decision.")

def show_public_check_summary():
    if not _PUBLIC_RESULTS:
        print("No public checks have been run yet.")
        return
    passed = sum(_PUBLIC_RESULTS.values())
    print(f"Public checks passed: {passed}/{len(_PUBLIC_RESULTS)}")
    print("These are practice checks, not the official Project 3 score.")

# %% [markdown]
# ## Data Leakage
# 
# %% [markdown]
# A very important (read: **the most important** topic) in practical data science scenarios is that of data leakage. Data leakage is a situation that occurs when the creator of a machine learning model allows the model to read both training data and test data to train the model. In Programming Project 3, the training data and the test data are separated for you already. Thus, the linear regression model should only be trained with the training data. Predictions can be made on either the training data or the test data. In the upcoming weeks, we will explore why you shouldn't train the model with the test data as well and what methods we can employ to choose the training set and the test set.
# 
# %% [markdown]
# ## Part A
# 
# First, we will use the `Ch3PartA` dataset to generate polynomial regressions using `scikit-learn`.
# This dataset contains 100 observations of points $x$ and their corresponding response, $y$. The data
# is divided into a training set $(x_{tr}, y_{tr})$ and a test set $(x_{te}, y_{te})$, and all the values are doubles.
# 
# ### A1.
# 
# To start, load `Ch3PartA.csv` into your notebook.
# 
# %%
def load_part_a(path="Ch3PartA.csv"):
    """Load the Part A dataset and return a pandas DataFrame."""
    return pd.read_csv(path)

part_a = load_part_a("Ch3PartA.csv")
part_a.head()

# %% [markdown]
# ### A2.
# 
# Create a scatter plot of: 
# 
# (a) `y_tr` against `x_tr` and another of 
# 
# (b) `y_te` against `x_te`. 
# 
# 
# Then, observe and comment on the similarities and differences between the plots.
# 
# %%
# Scatter plot: y_tr against x_tr
sns.scatterplot( data=part_a, x="x_tr", y="y_tr")
plt.xlabel("x_tr")
plt.ylabel("y_tr")
plt.title("Training Data")
plt.show()

# %%
# Scatter plot: y_te against x_te
sns.scatterplot(data=part_a, x="x_te", y="y_te")
plt.xlabel("x_te")
plt.ylabel("y_te")
plt.title("Test Data")
plt.show()
# %%
sns.scatterplot( data=part_a, x="x_tr", y="y_tr")
sns.scatterplot(data=part_a, x="x_te", y="y_te")

# %% [markdown]
# What is the maximum value of `y` in the training set and in the test set? Please store these variables as `max_y_train` and `max_y_test` below and run the first public-check cell! 
# 
# Run the public check after implementing the function. The check uses unrelated synthetic data and does not contain the real Project 3 answer.
# 
# %%
def get_max_y_values(df):
    """Return (maximum training y, maximum test y)."""
    return max(df.y_tr), max(df.y_te)

max_y_train, max_y_test = get_max_y_values(part_a)

# %%
# View the results here before you submit
print(max_y_train, max_y_test)
# %%
check_max_y_function(get_max_y_values)

# %% [markdown]
# Now, comment on the plot differences below. Please record your response into the multiline string named `plot_diffs_string` and then submit it to us via the public-check cell!
# 
# %%
plot_diffs_string = '''
The difference between two graph is not the significant both of them are a polynomial graph. Wild test data has lower Y value, then train data.
'''
# Write your comparison above.

# %%
check_written_response(plot_diffs_string, "A2 plot comparison")

# %% [markdown]
# ### A3. 
# Generate the necessary features to fit polynomial regressions up to the 20th degree (up to and including the $x_{20}$ term) on the training data. Hint: You will be fitting multi-variate linear regression models with polynomial features of $x$. Familiarize yourself with `sklearn.preprocessing.PolynomialFeatures`. 
# 
# Here, we're just asking you to practice generating the features. You'll pass one of them into the public checker for a quick check (although the public checker will not be very strict, so if you end up failing the next test case definitely make sure your work here is correct!)
# 
# %%
def make_polynomial_features(x, degree=20):
    if isinstance(x, pd.Series):
        x = x.to_frame()
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    return poly.fit_transform(x)

# You may call this function for any degrees you want to inspect.
make_polynomial_features(part_a, degree=20)

# %% [markdown]
# Now, let's check to make sure your highest-degree polynomial features are correct; namely the set of features that includes $x_{20}$, or `PolynomialFeatures(degree = 20)`. Please set `polynomial_features_df` as this **dataframe**.
# 
# If you do not receive full points, that means that either you have the wrong number of columns or some column values aren't correct!
# 
# %%
polynomial_features_df = make_polynomial_features(part_a[["x_tr"]], degree=20)
polynomial_features_df

# %%
check_polynomial_features_function(make_polynomial_features)
# %% [markdown]
# ### A4. 
# Calculate the training MSE and the test MSE for 20 polynomial models up to degree 20. Store these as lists named `mse_train` and `mse_test` respectively. Hint: Familiarize yourself with the `sklearn.metrics.mean_squared_error` package and try to automate the process, e.g., using a for loop with degrees going from 1 to 20.
# 
# %%
def calculate_polynomial_mses(df, max_degree=20):
    """Return (mse_train, mse_test) for degrees 1 through max_degree."""
    # Rename to a common column so sklearn sees identical feature names
    x_tr = df[["x_tr"]].rename(columns={"x_tr": "x"})
    x_te = df[["x_te"]].rename(columns={"x_te": "x"})
    y_tr = df["y_tr"]
    y_te = df["y_te"]

    mse_train = []
    mse_test = []

    for d in range(1, max_degree + 1):
        poly = PolynomialFeatures(degree=d, include_bias=False)
        Xtr_poly = poly.fit_transform(x_tr)
        Xte_poly = poly.transform(x_te)

        model = LinearRegression()
        model.fit(Xtr_poly, y_tr)

        mse_train.append(mean_squared_error(y_tr, model.predict(Xtr_poly)))
        mse_test.append(mean_squared_error(y_te, model.predict(Xte_poly)))

    return mse_train, mse_test

mse_train, mse_test = calculate_polynomial_mses(part_a, max_degree=20)

# %% [markdown]
# Run the public-check cells for both `mse_train` and `mse_test` in order; please make sure you don't put the wrong cell in!
# 
# %%
check_polynomial_mse_function(calculate_polynomial_mses)

# %%
# Inspect your real-data MSE lists without revealing any expected answers.
print("Training MSE count:", len(mse_train))
print("Test MSE count:", len(mse_test))

# %% [markdown]
# ### A5.
# 
# Generate a plot of both the training MSE and test MSE against flexibility (polynomial degree) for degrees 1 to 20. 
# 
# Find the minimum training and testing MSEs and set them to `min_train_mse` and `min_test_mse` respectively.
# 
# %%
def get_min_mses(mse_train, mse_test):
    """Return (minimum training MSE, minimum test MSE)."""
    return min(mse_train), min(mse_test)

min_train_mse, min_test_mse = get_min_mses(mse_train, mse_test)
# %%
print(min_train_mse, min_test_mse)
# %% [markdown]
# *Hint*: You should see the `min_train_mse < min_test_mse` since we have a bit of overfitting. Run the public-check cell below! Each of the variables is worth 1 point; we assign points based on how close you are to the true answer
# 
# %%
check_min_mse_function(get_min_mses)
# %% [markdown]
# ### A6. 
# From your plot, make an educated guess about the polynomial degree of the function that
# was used to generate the data. Then, give an estimate of the irreducible error $Var(\epsilon)$ for the optimal model on both the training set and test set. 
# 
# *Hint*: The optimal model is obtained when we use the maximal degree polynomial that does not overfit. Revisit the section on hypothesis testing and think about the relationship between MSE, RSS, and RSE to calculate the irreducible error.
# 
# %%
degree = int(np.argmin(mse_test) + 1)
RSE_train_sq = float(mse_train[degree - 1])
RSE_test_sq = float(mse_test[degree - 1])
# Compute these from your analysis of the real Part A data.

# %% [markdown]
# Please set your respective irreducible errors as `RSE_train_sq` and `RSE_test_sq` respectively, and set the number of polynomial features as `degree`. Then, run the public-check cell below. It grades similar to above, but we add 1 point for the `degree` variable!
# 
# %%
print("Desired degree for best model: ", degree)
print("Irreducible error (training): ", RSE_train_sq)
print("Irreducible error (test): ", RSE_test_sq)
# %%
check_a6_format(degree, RSE_train_sq, RSE_test_sq)

# %% [markdown]
# ## Part B
# 
# Next, we will use the `Ch3PartB` dataset to observe the effects of collinearity using `statsmodels`.
# This dataset contains 100 observations of points $(x1, x2)$, and $y$, the response variable.
# 
# ### B1. 
# Load the data from `Ch3PartB.csv` into a pandas DataFrame.
# 
# %%
def load_part_b(path="Ch3PartB.csv"):
    """Load the Part B dataset and return a pandas DataFrame."""
    return pd.read_csv(path)

part_b = load_part_b("Ch3PartB.csv")
part_b.head()

# %% [markdown]
# ### B2. 
# Show a scatterplot displaying the relationship between $x1$ and $x2$ 
# 
# What is the correlation coefficient between $x1$ and $x2$? Compute the answer and store it as `correlation_variable` -- it should be a single floating-point number
# 
# %%
# Scatterplot of x1 against x2
plt.scatter(part_b['x1'], part_b['x2'])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Relationship between x1 and x2')
plt.show()

# %%
def get_x1_x2_correlation(df):
    """Return the correlation between x1 and x2 as one scalar."""
    return df['x1'].corr(df['x2'])

correlation_variable = get_x1_x2_correlation(part_b)
print(correlation_variable)

# %%
check_correlation_function(get_x1_x2_correlation)

# %% [markdown]
# ### B3. 
# 
# Using the data, fit a least squares regression to predict $y$ using $x1$ and $x2$. Describe your results in a Markdown cell. 
# 
# *Hint*: Familiarize yourself with `statsmodels.formula.api.ols`. 
# 
# We have several questions here as well:
# 
# (a) What are the estimates $\hat{\beta_0}, \hat{\beta_1}, \hat{\beta_2}$?
# 
# (b) At a 95% confidence level, can you reject the null hypothesis $H_0: \beta_1 = 0$? 
# 
# (c) What about $H_0: \beta_2 = 0$?
# 
# %%
def fit_multiple_regression(df):
    """Fit the Part B model predicting y from x1 and x2; return the fitted model."""
    import statsmodels.formula.api as sm
    model = sm.ols(formula='y ~ x1 + x2', data=df).fit()
    return model

multiple_model = fit_multiple_regression(part_b)
print(multiple_model.summary())

# %% [markdown]
# For your answers to $\hat{\beta_0}, \hat{\beta_1}, \hat{\beta_2}$, please input them either using code or typing the numbers from `statsmodels`' output into the variables below. You should include at least 4 digits after the decimal point.
# 
# %%
# Extract your coefficient estimates from multiple_model.
beta_0 = multiple_model.params['Intercept']
beta_1 = multiple_model.params['x1']
beta_2 = multiple_model.params['x2']

# %% [markdown]
# Please run but *do not change* the cell below to set up your public checker. Afterr, run the first public-check cell with these variables! You will receive 0.5 points for each variable.
# 
# %%
# Keep this dictionary if you want a convenient record of your three coefficients.
answer_dict = {0: beta_0, 1: beta_1, 2: beta_2}

# %%
check_multiple_regression_function(fit_multiple_regression)

# %% [markdown]
# Now, we want to evaluate the hypothesis test of $H_0: \beta_1 = 0$ when both $x1$ and $x2$ are present.
# 
# Please compute either the $t$-value or $p$-value and set it as `test_statistic_b1`, then determine whether or not you reject the null hypothesiss at the $95\%$ confidence level. Set that variable as a boolean (`True/False`) as `reject_null_b1`. 
# 
# If your values are either incorrect or do not agree (i.e. you said the null would be rejected when it should not be), then you will not receive full points!
# 
# You will receive 0.5 points for getting the correct statistics as well as 1 points for your in-context evaluation of the null hypothesis.
# 
# %%
test_statistic_b1 = multiple_model.pvalues['x1'] # use t or p, to at least 3 significant digits
reject_null_b1 = test_statistic_b1 < 0.05      # bool
# Determine both values from multiple_model.
print(f"Test statistic for beta1: {test_statistic_b1}")
print(f"Reject null hypothesis for beta1: {reject_null_b1}")

print(f"Reject null hypothesis for beta1: {reject_null_b1}")
# %%
check_hypothesis_format(test_statistic_b1, reject_null_b1, "B3 beta1 hypothesis format")

# %% [markdown]
# Let's do the same for the other test, to evaluate the hypothesis test of $H_0: \beta_2 = 0$  when both $x1$ and $x2$ are present.
# 
# Please compute either the $t$-value or $p$-value and set it as `test_statistic_b2`, then determine whether or not you reject the null hypothesiss at the $95\%$ confidence level. Set that variable as a boolean (`True/False`) as `reject_null_b2`
# 
# You will receive 0.5 points for getting the correct statistics as well as 1 point for your in-context evaluation of the null hypothesis.
# 
# %%
test_statistic_b2 = multiple_model.pvalues['x2'] # use t or p, to at least 3 significant digits
reject_null_b2 = test_statistic_b2 < 0.05      # bool
# Determine both values from multiple_model.
print(f"Test statistic for beta2: {test_statistic_b2}")
print(f"Reject null hypothesis for beta2: {reject_null_b2}")

# %%
check_hypothesis_format(test_statistic_b2, reject_null_b2, "B3 beta2 hypothesis format")

# %% [markdown]
# ### B4. 
# 
# Now fit a least squares regression to predict $y$ using only $x1$. Comment on your results.
# 
# Can you reject the null hypothesis $H_0: \beta_1 = 0$?
# 
# %%
def fit_x1_regression(df):
    """Fit the Part B model predicting y from x1 only; return the fitted model."""
    return smf.ols(formula='y ~ x1', data=df).fit()

x1_model = fit_x1_regression(part_b)
print(x1_model.summary())

# %% [markdown]
# When evaluating $H_0: \beta_1 = 0$ when fitting with only $x1$ please do the following:
# 
# - Please compute either the $t$-value or $p$-value and set it as `test_statistic_b4`
# - Determine whether or not you reject the null hypothesiss at the $95\%$ confidence level; set that variable as a boolean (`True/False`) as `reject_null_b4`
# - Determine if $x1$ is significant; set that result as a boolean (`True/False`) as `is_x1_significant`
# 
# Similar to previously, you'll receive points both for your test statistics and the evaluation.
# 
# %%
test_statistic_b4 = x1_model.pvalues['x1']
reject_b4_null_hypothesis = test_statistic_b4 < 0.05
is_x1_significant = reject_b4_null_hypothesis
# Determine these from x1_model.
print(f"Test statistic for beta1 (x1 only): {test_statistic_b4}")
print(f"Reject null hypothesis for beta1 (x1 only): {reject_b4_null_hypothesis}")
print(f"Is x1 significant (x1 only): {is_x1_significant}")

# %%
check_x1_regression_function(fit_x1_regression)
check_hypothesis_format(test_statistic_b4, reject_b4_null_hypothesis, "B4 hypothesis format", is_x1_significant)

# %% [markdown]
# ### B5. 
# 
# Now fit a least squares regression to predict $y$ using only $x2$. Comment on your results.
# 
# Can you reject the null hypothesis $H_0: \beta_2 = 0$?
# 
# %%
def fit_x2_regression(df):
    """Fit the Part B model predicting y from x2 only; return the fitted model."""
    return smf.ols(formula='y ~ x2', data=df).fit()

x2_model = fit_x2_regression(part_b)
print(x2_model.summary())

# %% [markdown]
# When evaluating $H_0: \beta_2 = 0$ when fitting with only $x1$ please do the following:
# 
# - Please compute either the $t$-value or $p$-value and set it as `test_statistic_b5`
# - Determine whether or not you reject the null hypothesiss at the $95\%$ confidence level; set that variable as a boolean (`True/False`) as `reject_null_b5`
# - Determine if $x2$ is significant; set that result as a boolean (`True/False`) as `is_x2_significant`
# 
# Scoring is identical to B4 above!
# 
# %%
test_statistic_b5 = x2_model.pvalues['x2']
reject_b5_null_hypothesis = test_statistic_b5 < 0.05
is_x2_significant = reject_b5_null_hypothesis
# Determine these from x2_model.
print(f"Test statistic for beta2 (x2 only): {test_statistic_b5}")
print(f"Reject null hypothesis for beta2 (x2 only): {reject_b5_null_hypothesis}")
print(f"Is x2 significant (x2 only): {is_x2_significant}")

# %%
check_x2_regression_function(fit_x2_regression)
check_hypothesis_format(test_statistic_b5, reject_b5_null_hypothesis, "B5 hypothesis format", is_x2_significant)

# %% [markdown]
# ### B6. 
# 
# Do Part B Questions 3-5 contradict each other? Explain why or why not.
# 
# %%
# Use this space for any additional analysis you want to support your B6 explanation.

# %% [markdown]
# Now, comment on the apparent contradiction below. Enter a boolean (`True/False`) for whether or not the answers contradict as `is_contradiction`, and then record your explanation into the multiline string named `contradiction_string` and then submit it to us via the public-check cell!
# 
# Please note that you'll need to have the right answer as well as have an explanation that has a reasonable set of keywords in order to get full credit!
# 
# *Note*: if you have an explanation that you think is reasonable but you aren't passing the public checker, let us know on Piazza!
# 
# %%
is_contradiction = False
contradiction_string = '''B3 uses both predictors together. B4 and B5 each use only one predictor, so their slopes can absorb effects from the omitted correlated predictor.
'''
# Complete both fields from your interpretation of B3–B5.

# %%
_contradiction_type_ok = isinstance(is_contradiction, (bool, np.bool_))
_public_result("B6 contradiction boolean format", _contradiction_type_ok, "This check does not reveal whether the correct answer is True or False.")
check_written_response(contradiction_string, "B6 explanation completeness", min_chars=60)

# %%
show_public_check_summary()

# %% [markdown]
# ## Submit
# 
# You are done when you have completed the Project 3 analysis, answered the associated Canvas questions, and run the public checks you want to use for feedback.
# 
# You can review the public-check status with:
# 
# ```python
# show_public_check_summary()
# ```
# 
# Remember: the public checks use synthetic examples and are **not an answer key or official score**.
# 