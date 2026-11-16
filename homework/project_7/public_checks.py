"""
Public checks for Project 7.

This is a module to run each check right after the graded cell it belongs to. 
A check looks at your own results on the real Heart.csv and tells you which of Gradescope's requirements they don't meet yet: 
variable names and types, model settings such as random_state and cv, and whether your metrics match your models.
The multiple-choice checks only look at the format of your answer, never at whether it's right.

Usage from the notebook:

    import public_checks

    public_checks.check_read_data(heart)
    public_checks.check_dt_score(clf, X_test, y_test, dt_precision, dt_recall, dt_accuracy)
    ...
    public_checks.show_public_check_summary()
    public_checks.check_submission_file("project7_submission.py")

"""

import ast
import builtins
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

__all__ = [
    "check_read_data",
    "check_hot_encode",
    "check_age",
    "check_corr",
    "check_split",
    "check_dt",
    "check_image",
    "check_dt_score",
    "check_dt_tune",
    "check_dt_tuned",
    "check_bag",
    "check_bag_score",
    "check_bag_tuned",
    "check_bag_best",
    "check_rf",
    "check_rf_score",
    "check_rf_tuned",
    "check_rf_tuned_params",
    "check_base_choice",
    "check_tuned_choice",
    "show_public_check_summary",
    "check_submission_file",
]

_PUBLIC_RESULTS = {}

RAW_COLUMNS = [
    "Age", "Sex", "ChestPain", "RestBP", "Chol", "Fbs", "RestECG",
    "MaxHR", "ExAng", "Oldpeak", "Slope", "Ca", "Thal", "HD",
]

ENCODED_COLUMNS = [
    "Age", "Sex", "RestBP", "Chol", "Fbs", "RestECG", "MaxHR", "ExAng",
    "Oldpeak", "Slope", "Ca", "HD",
    "ChestPain_asymptomatic", "ChestPain_nonanginal",
    "ChestPain_nontypical", "ChestPain_typical",
    "Thal_fixed", "Thal_normal", "Thal_reversable",
]

DT_DEPTHS = list(range(3, 21))
BAG_N_ESTIMATORS = list(range(10, 50))
RF_DEPTHS = list(range(3, 21))
RF_N_ESTIMATORS = list(range(10, 30))


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------

def _run(label, find_problems):
    """Run a generator of problem messages and print PASS or FAIL."""
    try:
        problems = list(find_problems())
    except Exception as exc:
        problems = [f"The check could not run ({type(exc).__name__}: {exc})."]

    passed = not problems
    _PUBLIC_RESULTS[label] = passed

    if passed:
        print(f"[PASS] {label}")
    else:
        print(f"[FAIL] {label}")
        for problem in problems:
            print(f"  - {problem}")

    return passed


def _is_fitted(model, attribute):
    return hasattr(model, attribute)


def _metric_problems(model, X_test, y_test, values, names, model_name):
    """Compare precision, recall and accuracy with the model's test predictions."""
    pred = model.predict(X_test)
    scorers = (precision_score, recall_score, accuracy_score)

    for name, value, scorer in zip(names, values, scorers):
        try:
            value = float(value)
        except (TypeError, ValueError):
            yield f"`{name}` must be a number; it is {type(value).__name__}."
            continue

        if np.isclose(value, scorer(y_test, pred)):
            continue

        if np.isclose(value, scorer(pred, y_test)):
            yield (
                f"`{name}` was computed with the arguments swapped. Pass the "
                f"true labels first: {scorer.__name__}(y_test, predictions)."
            )
        else:
            yield (
                f"`{name}` doesn't match the {scorer.__name__} of {model_name}'s "
                "predictions on X_test."
            )


def _grid_problems(grid, grid_name, estimator_class, expected_grid):
    """Check a GridSearchCV against the required estimator, grid and cv."""
    if not isinstance(grid, GridSearchCV):
        yield f"`{grid_name}` must be a GridSearchCV; it is {type(grid).__name__}."
        return

    if not isinstance(grid.estimator, estimator_class):
        yield (
            f"The estimator inside `{grid_name}` must be a "
            f"{estimator_class.__name__}; it is {type(grid.estimator).__name__}."
        )
    elif grid.estimator.random_state != 42:
        yield (
            f"Create the estimator with random_state=42: "
            f"{estimator_class.__name__}(random_state=42)."
        )

    if not (isinstance(grid.cv, int) and grid.cv == 10):
        yield f"Use cv=10 (the integer 10); `{grid_name}` has cv={grid.cv!r}."

    if not isinstance(grid.param_grid, dict):
        yield "The parameter grid must be a single dictionary."
        return

    for key, expected in expected_grid.items():
        actual = list(grid.param_grid.get(key, []))
        if actual != expected:
            yield (
                f"Search {key} over every integer from {expected[0]} to "
                f"{expected[-1]}: np.arange({expected[0]}, {expected[-1] + 1})."
            )

    if not _is_fitted(grid, "best_estimator_"):
        yield f"Fit `{grid_name}` on X_train and y_train."


def _choice_problems(value, name, letters, single):
    if not isinstance(value, list):
        yield f"`{name}` must be a list, for example [\"{letters[0]}\"]."
        return

    if not value:
        yield f"`{name}` is empty. Put your answer letter(s) in the list."
        return

    if single and len(value) != 1:
        yield f"`{name}` must contain exactly one letter."

    for item in value:
        if item not in list(letters):
            yield (
                f"Each answer must be one uppercase letter as a string, one of "
                f"{', '.join(letters)}; got {item!r}."
            )

    if len(set(map(str, value))) != len(value):
        yield f"`{name}` lists the same letter more than once."


# ------------------------------------------------------------------
# Part 1: data
# ------------------------------------------------------------------

def check_read_data(heart):
    """Task 1.1. Pass your `heart` DataFrame."""
    def problems():
        if not isinstance(heart, pd.DataFrame):
            yield f"`heart` must be a pandas DataFrame; it is {type(heart).__name__}."
            return

        if list(heart.columns) != RAW_COLUMNS:
            yield (
                "`heart` should have the 14 columns of Heart.csv, starting with "
                "Age. Read it with pd.read_csv(\"Heart.csv\") and don't set index_col."
            )

        if heart.isna().any().any():
            yield "Some rows still contain missing values. Drop every row with a NaN."
        elif len(heart) != 297:
            yield f"`heart` should have 297 rows after dropping missing values; it has {len(heart)}."

    return _run("read_data", problems)


def check_hot_encode(heart):
    """Task 1.2. Pass your encoded `heart` DataFrame."""
    def problems():
        if not isinstance(heart, pd.DataFrame):
            yield f"`heart` must be a pandas DataFrame; it is {type(heart).__name__}."
            return

        columns = list(heart.columns)
        if columns != ENCODED_COLUMNS:
            missing = [c for c in ENCODED_COLUMNS if c not in columns]
            extra = [c for c in columns if c not in ENCODED_COLUMNS]
            if missing:
                yield f"Missing columns: {missing}."
            if extra:
                yield f"Unexpected columns: {extra}."
            if not missing and not extra:
                yield "The columns are right but in the wrong order. See the required order in Task 1.2."

        if "HD" in heart.columns and not heart["HD"].isin([0, 1]).all():
            yield (
                "`HD` must hold the integers 0 (No) and 1 (Yes). "
                "Use heart[\"HD\"].map({\"No\": 0, \"Yes\": 1})."
            )

        dummies = [c for c in columns if c.startswith(("ChestPain_", "Thal_"))]
        if dummies and not heart[dummies].isin([0, 1]).all().all():
            yield "The one-hot columns must hold only True/False or 1/0."

        if len(heart) != 297:
            yield f"Encoding shouldn't change the number of rows (297); `heart` has {len(heart)}."

    return _run("hot_encode", problems)


def check_age(heart, min_age, max_age):
    """Task 1.3. Pass `heart`, `min_age` and `max_age`."""
    def problems():
        for name, value, expected, word in (
            ("min_age", min_age, heart["Age"].min(), "minimum"),
            ("max_age", max_age, heart["Age"].max(), "maximum"),
        ):
            if isinstance(value, str) or not np.isscalar(value):
                yield f"`{name}` must be a single number; it is {type(value).__name__}."
            elif value != expected:
                yield f"`{name}` doesn't match the {word} of heart[\"Age\"]."

    return _run("age", problems)


def check_corr(heart, highest_corr):
    """Task 1.4. Pass `heart` and `highest_corr`."""
    def problems():
        if not isinstance(highest_corr, str):
            yield f"`highest_corr` must be a string (a column name); it is {type(highest_corr).__name__}."
            return

        if highest_corr == "HD":
            yield "Leave out HD itself: its correlation with itself is always 1."
            return

        if highest_corr not in heart.columns:
            yield f"{highest_corr!r} isn't a column of `heart`. Use the exact column name."
            return

        expected = heart.corr(numeric_only=True)["HD"].drop("HD").idxmax()
        if highest_corr != expected:
            yield f"{highest_corr!r} isn't the predictor with the highest positive correlation with HD."

    return _run("corr", problems)


def check_split(heart, X_train, X_test, y_train, y_test):
    """Task 1.5. Pass `heart` and the four split variables."""
    def problems():
        for name, value, kind in (
            ("X_train", X_train, pd.DataFrame),
            ("X_test", X_test, pd.DataFrame),
            ("y_train", y_train, pd.Series),
            ("y_test", y_test, pd.Series),
        ):
            if not isinstance(value, kind):
                yield f"`{name}` must be a pandas {kind.__name__}; it is {type(value).__name__}."
        if any(not isinstance(v, (pd.DataFrame, pd.Series)) for v in (X_train, X_test, y_train, y_test)):
            return

        if "HD" in X_train.columns:
            yield "X must not contain the response HD. Use heart.drop(\"HD\", axis=1)."
        elif list(X_train.columns) != [c for c in heart.columns if c != "HD"]:
            yield "X must contain every column of `heart` except HD, in the same order."

        expected = train_test_split(
            heart.drop("HD", axis=1),
            heart["HD"],
            test_size=0.2,
            random_state=42,
        )
        for name, actual, exp in zip(
            ("X_train", "X_test", "y_train", "y_test"),
            (X_train, X_test, y_train, y_test),
            expected,
        ):
            if list(actual.index) != list(exp.index):
                yield (
                    f"`{name}` doesn't contain the expected rows. Use "
                    "train_test_split(X, y, test_size=0.2, random_state=42) and "
                    "unpack it as X_train, X_test, y_train, y_test."
                )

    return _run("split", problems)


# ------------------------------------------------------------------
# Part 2: decision tree
# ------------------------------------------------------------------

def check_dt(clf, X_train):
    """Task 2.1. Pass `clf` and `X_train`."""
    def problems():
        if not isinstance(clf, DecisionTreeClassifier):
            yield f"`clf` must be a DecisionTreeClassifier; it is {type(clf).__name__}."
            return
        if clf.random_state != 42:
            yield "Create the tree with random_state=42."
        if not _is_fitted(clf, "tree_"):
            yield "Fit `clf` on X_train and y_train."
        elif clf.n_features_in_ != X_train.shape[1]:
            yield "Fit the tree on all predictors in X_train."

    return _run("dt", problems)


def check_image(img):
    """Task 2.2. Pass `img`."""
    def problems():
        if str(type(img)) != "<class 'IPython.core.display.Image'>":
            yield (
                f"`img` must be an IPython.display.Image; it is {type(img).__name__}. "
                "Build it with Image(graph.create_png())."
            )

    return _run("image", problems)


def check_dt_score(clf, X_test, y_test, dt_precision, dt_recall, dt_accuracy):
    """Task 2.3. Pass the base tree, the test set and its three metrics."""
    return _run("dt_score", lambda: _metric_problems(
        clf, X_test, y_test,
        (dt_precision, dt_recall, dt_accuracy),
        ("dt_precision", "dt_recall", "dt_accuracy"),
        "clf",
    ))


def check_dt_tune(answers):
    """Task 2.4. Format only: does not say whether the answer is right."""
    return _run("dt_tune", lambda: _choice_problems(answers, "answers", "ABCD", single=False))


def check_dt_tuned(grid_search_cv_dt, X_test, y_test,
                   dt_tuned_precision, dt_tuned_recall, dt_tuned_accuracy):
    """Task 2.5. Pass the fitted grid search, the test set and its three metrics."""
    def problems():
        yield from _grid_problems(
            grid_search_cv_dt, "grid_search_cv_dt",
            DecisionTreeClassifier, {"max_depth": DT_DEPTHS},
        )
        if _is_fitted(grid_search_cv_dt, "best_estimator_"):
            yield from _metric_problems(
                grid_search_cv_dt, X_test, y_test,
                (dt_tuned_precision, dt_tuned_recall, dt_tuned_accuracy),
                ("dt_tuned_precision", "dt_tuned_recall", "dt_tuned_accuracy"),
                "grid_search_cv_dt",
            )

    return _run("dt_tuned", problems)


# ------------------------------------------------------------------
# Part 3: bagging
# ------------------------------------------------------------------

def check_bag(bag, X_train, y_train):
    """Task 3.1. Pass `bag` and the training set."""
    def problems():
        if not isinstance(bag, BaggingClassifier):
            yield f"`bag` must be a BaggingClassifier; it is {type(bag).__name__}."
            return
        if bag.random_state != 42:
            yield "Create the model with random_state=42."
        if not _is_fitted(bag, "estimators_samples_"):
            yield "Fit `bag` on X_train and y_train."
            return

        expected = BaggingClassifier(random_state=42).fit(X_train, y_train)
        same_samples = (
            len(bag.estimators_samples_) == len(expected.estimators_samples_)
            and all(
                np.array_equal(a, b)
                for a, b in zip(bag.estimators_samples_, expected.estimators_samples_)
            )
        )
        if not same_samples:
            yield (
                "Keep every other setting at its default: BaggingClassifier(random_state=42), "
                "with no estimator and the default 10 trees, fitted on X_train and y_train."
            )

    return _run("bag", problems)


def check_bag_score(bag, X_test, y_test, bag_precision, bag_recall, bag_accuracy):
    """Task 3.2. Pass the base bagging model, the test set and its three metrics."""
    return _run("bag_score", lambda: _metric_problems(
        bag, X_test, y_test,
        (bag_precision, bag_recall, bag_accuracy),
        ("bag_precision", "bag_recall", "bag_accuracy"),
        "bag",
    ))


def check_bag_tuned(grid_search_cv_bag, X_test, y_test,
                    bag_tuned_precision, bag_tuned_recall, bag_tuned_accuracy):
    """Task 3.3. Pass the fitted grid search, the test set and its three metrics."""
    def problems():
        yield from _grid_problems(
            grid_search_cv_bag, "grid_search_cv_bag",
            BaggingClassifier, {"n_estimators": BAG_N_ESTIMATORS},
        )
        if _is_fitted(grid_search_cv_bag, "best_estimator_"):
            yield from _metric_problems(
                grid_search_cv_bag, X_test, y_test,
                (bag_tuned_precision, bag_tuned_recall, bag_tuned_accuracy),
                ("bag_tuned_precision", "bag_tuned_recall", "bag_tuned_accuracy"),
                "grid_search_cv_bag",
            )

    return _run("bag_tuned", problems)


def check_bag_best(bag_best_param, grid_search_cv_bag):
    """Task 3.3. Pass `bag_best_param` and the fitted grid search."""
    def problems():
        if not isinstance(bag_best_param, dict):
            yield f"`bag_best_param` must be a dict; it is {type(bag_best_param).__name__}."
        elif bag_best_param != grid_search_cv_bag.best_params_:
            yield "`bag_best_param` must equal grid_search_cv_bag.best_params_."

    return _run("bag_best", problems)


# ------------------------------------------------------------------
# Part 4: random forest
# ------------------------------------------------------------------

def check_rf(rf):
    """Task 4.1. Pass `rf`."""
    def problems():
        if not isinstance(rf, RandomForestClassifier):
            yield f"`rf` must be a RandomForestClassifier; it is {type(rf).__name__}."
            return
        if rf.random_state != 42:
            yield "Create the forest with random_state=42."
        if rf.n_estimators != RandomForestClassifier().n_estimators:
            yield "Keep the default number of trees (don't set n_estimators)."
        if not _is_fitted(rf, "estimators_"):
            yield "Fit `rf` on X_train and y_train."

    return _run("rf", problems)


def check_rf_score(rf, X_test, y_test, rf_precision, rf_recall, rf_accuracy):
    """Task 4.2. Pass the base forest, the test set and its three metrics."""
    return _run("rf_score", lambda: _metric_problems(
        rf, X_test, y_test,
        (rf_precision, rf_recall, rf_accuracy),
        ("rf_precision", "rf_recall", "rf_accuracy"),
        "rf",
    ))


def check_rf_tuned(grid_search_cv_rf, X_test, y_test,
                   rf_tuned_precision, rf_tuned_recall, rf_tuned_accuracy):
    """Task 4.3. Pass the fitted grid search, the test set and its three metrics."""
    def problems():
        yield from _grid_problems(
            grid_search_cv_rf, "grid_search_cv_rf",
            RandomForestClassifier,
            {"max_depth": RF_DEPTHS, "n_estimators": RF_N_ESTIMATORS},
        )
        if _is_fitted(grid_search_cv_rf, "best_estimator_"):
            yield from _metric_problems(
                grid_search_cv_rf, X_test, y_test,
                (rf_tuned_precision, rf_tuned_recall, rf_tuned_accuracy),
                ("rf_tuned_precision", "rf_tuned_recall", "rf_tuned_accuracy"),
                "grid_search_cv_rf",
            )

    return _run("rf_tuned", problems)


def check_rf_tuned_params(rf_tuned, grid_search_cv_rf):
    """Task 4.4. Pass `rf_tuned` and the fitted grid search."""
    def problems():
        if not isinstance(rf_tuned, dict):
            yield f"`rf_tuned` must be a dict; it is {type(rf_tuned).__name__}."
        elif rf_tuned != grid_search_cv_rf.best_params_:
            yield "`rf_tuned` must equal grid_search_cv_rf.best_params_."

    return _run("rf_tuned_params", problems)


# ------------------------------------------------------------------
# Part 5: model comparison (format only)
# ------------------------------------------------------------------

def check_base_choice(choice):
    """Task 5.1. Format only: does not say whether the answer is right."""
    return _run("base_choice", lambda: _choice_problems(choice, "choice", "ABC", single=True))


def check_tuned_choice(choice):
    """Task 5.2. Format only: does not say whether the answer is right."""
    return _run("tuned_choice", lambda: _choice_problems(choice, "choice", "ABC", single=True))


def show_public_check_summary():
    if not _PUBLIC_RESULTS:
        print("No public checks have been run yet.")
        return

    passed = sum(_PUBLIC_RESULTS.values())
    total = len(_PUBLIC_RESULTS)
    failed = [label for label, ok in _PUBLIC_RESULTS.items() if not ok]

    print(f"Public checks passed: {passed}/{total}")
    if failed:
        print("Still failing: " + ", ".join(failed))
    print(
        "Gradescope reruns each graded cell on hidden data and also grades "
        "the three multiple-choice answers, which these checks don't."
    )


# ------------------------------------------------------------------
# Submission file
# ------------------------------------------------------------------
# These tables mirror how the Gradescope autograder (grade_submission.py,
# October 20, 2026 version) reads a submission. It finds each task's tag
# line, takes the code from there up to the next notebook cell, and runs
# that code by itself in a fresh namespace holding only the names below.
# Keep them in sync with the autograder.

_TAG = "# PROJECT7_TASK:"

_TASKS = [
    "read_data", "hot_encode", "age", "corr", "split",
    "dt", "image", "dt_score", "dt_tune", "dt_tuned",
    "bag", "bag_score", "bag_tuned",
    "rf", "rf_score", "rf_tuned", "rf_tuned_params",
    "base_choice", "tuned_choice",
]

_SETUP_NAMES = {
    "pd", "np", "tree", "DecisionTreeClassifier", "BaggingClassifier",
    "RandomForestClassifier", "precision_score", "recall_score",
    "accuracy_score", "train_test_split", "GridSearchCV", "Image", "pydotplus",
}

_SPLIT_NAMES = {"X_train", "X_test", "y_train", "y_test"}

_TASK_INPUTS = {
    "read_data": set(),
    "hot_encode": {"heart"},
    "age": {"heart"},
    "corr": {"heart"},
    "split": {"heart"},
    "dt": _SPLIT_NAMES,
    "image": _SPLIT_NAMES | {"clf", "y_pred"},
    "dt_score": _SPLIT_NAMES | {"clf", "y_pred"},
    "dt_tune": set(),
    "dt_tuned": _SPLIT_NAMES,
    "bag": _SPLIT_NAMES,
    "bag_score": _SPLIT_NAMES | {"bag"},
    "bag_tuned": _SPLIT_NAMES,
    "rf": _SPLIT_NAMES,
    "rf_score": _SPLIT_NAMES | {"rf", "y_pred"},
    "rf_tuned": _SPLIT_NAMES,
    "rf_tuned_params": _SPLIT_NAMES,  # plus whatever the rf_tuned cell defines
    "base_choice": set(),
    "tuned_choice": set(),
}

# Variables Gradescope reads after running each cell. A task's inputs (such as `heart` in hot_encode) already exist, so they aren't listed again.
_TASK_OUTPUTS = {
    "read_data": {"heart"},
    "hot_encode": set(),
    "age": {"min_age", "max_age"},
    "corr": {"highest_corr"},
    "split": {"X_train", "X_test", "y_train", "y_test"},
    "dt": {"clf"},
    "image": {"img"},
    "dt_score": {"dt_precision", "dt_recall", "dt_accuracy"},
    "dt_tune": {"answers"},
    "dt_tuned": {"grid_search_cv_dt", "dt_tuned_precision", "dt_tuned_recall", "dt_tuned_accuracy"},
    "bag": {"bag"},
    "bag_score": {"bag_precision", "bag_recall", "bag_accuracy"},
    "bag_tuned": {"grid_search_cv_bag", "bag_tuned_precision", "bag_tuned_recall",
                  "bag_tuned_accuracy", "bag_best_param"},
    "rf": {"rf"},
    "rf_score": {"rf_precision", "rf_recall", "rf_accuracy"},
    "rf_tuned": {"grid_search_cv_rf", "rf_tuned_precision", "rf_tuned_recall", "rf_tuned_accuracy"},
    "rf_tuned_params": {"rf_tuned"},
    "base_choice": {"choice"},
    "tuned_choice": {"choice"},
}

_CHOICE_VARIABLE = {"dt_tune": "answers", "base_choice": "choice", "tuned_choice": "choice"}

_TUNING_TASKS = {"dt_tuned", "bag_tuned", "rf_tuned"}
_MODEL_CLASSES = {"DecisionTreeClassifier", "BaggingClassifier", "RandomForestClassifier"}
_TUNED_SETTINGS = {"max_depth", "n_estimators"}

# Packages that aren't installed on Gradescope. pydotplus is replaced by a
# stand-in there, so importing the real one inside a graded cell fails.
_NOT_ON_GRADESCOPE = {"seaborn", "matplotlib", "graphviz", "pydot", "pydotplus"}


def _extract_tasks(source):
    """Cut out each task's code exactly the way the autograder does."""
    positions = sorted(
        (source.find(_TAG + task), task)
        for task in _TASKS
        if source.find(_TAG + task) >= 0
    )

    tasks = {}
    for i, (pos, task) in enumerate(positions):
        start = pos + len(_TAG + task)
        next_task = positions[i + 1][0] if i + 1 < len(positions) else len(source)
        next_cell = source.find("# In[", start)
        end = next_cell if 0 <= next_cell < next_task else next_task
        tasks[task] = source[start:end].strip()

    return tasks


def _bound_names(tree_):
    """Names a piece of code defines itself (assignments, imports, defs, ...)."""
    names = set()
    for node in ast.walk(tree_):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def _raises_not_implemented(tree_):
    for node in ast.walk(tree_):
        if isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                return True
    return False


def _only_empty_list(tree_, name):
    """True if `name` is only ever assigned [] and never added to."""
    assigned = []
    for node in ast.walk(tree_):
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                assigned.append(node.value)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                return False
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            target = node.func.value
            if isinstance(target, ast.Name) and target.id == name:
                return False
    return bool(assigned) and all(
        isinstance(value, ast.List) and not value.elts for value in assigned
    )


def _typed_in_settings(tree_):
    """Model constructors called with a literal max_depth or n_estimators."""
    for node in ast.walk(tree_):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name not in _MODEL_CLASSES:
            continue
        for keyword in node.keywords:
            if (
                keyword.arg in _TUNED_SETTINGS
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, (int, float))
            ):
                yield f"{name}({keyword.arg}={keyword.value.value!r})"


def _cell_problems(task, code, extra_inputs):
    try:
        tree_ = ast.parse(code)
    except SyntaxError as exc:
        yield f"has a syntax error on its line {exc.lineno}: {exc.msg}."
        return

    if _raises_not_implemented(tree_):
        yield "still raises NotImplementedError. Delete that line once your answer is in."
        return

    bound = _bound_names(tree_)
    for name in sorted(_TASK_OUTPUTS[task] - bound):
        yield (
            f"doesn't define `{name}`. Gradescope looks for it in this cell only, so "
            "define it here even if another cell already does."
        )

    if task in _CHOICE_VARIABLE and _only_empty_list(tree_, _CHOICE_VARIABLE[task]):
        yield f"`{_CHOICE_VARIABLE[task]}` is still an empty list. Put your answer letter(s) in it."

    if task in _TUNING_TASKS:
        for call in _typed_in_settings(tree_):
            yield (
                f"builds {call} with a value typed in. The best value on Gradescope's "
                "data is different: compute the metrics from the grid search's own "
                "predictions instead."
            )

    for node in ast.walk(tree_):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
            if any(alias.name == "GridSearchCV" for alias in node.names):
                yield (
                    "imports GridSearchCV. Gradescope swaps in a faster GridSearchCV, "
                    "and importing it inside a graded cell undoes that, so the test "
                    "fails. Use the GridSearchCV from the setup cell."
                )
        else:
            continue
        for module in modules:
            root = module.split(".")[0]
            if root in _NOT_ON_GRADESCOPE:
                yield f"imports {root}, which isn't available on Gradescope. Move that code to an ungraded cell."

    available = (
        _SETUP_NAMES | _TASK_INPUTS[task] | extra_inputs
        | bound | set(dir(builtins))
    )
    used = {
        node.id
        for node in ast.walk(tree_)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }

    for name in sorted(used - available):
        if name == "get_ipython":
            yield "contains a notebook command (a line starting with % or !). Move it to an ungraded cell."
        elif name == "display":
            yield "calls display(), which only exists in notebooks. Use print() instead."
        else:
            inputs = sorted(_TASK_INPUTS[task] | extra_inputs)
            yield (
                f"uses `{name}`, which this cell doesn't define and Gradescope doesn't "
                "provide to it. Gradescope runs this cell on its own, so it can only use "
                + (f"{', '.join(inputs)}, " if inputs else "")
                + "the setup imports and what the cell itself defines."
            )


def check_submission_file(path="project7_submission.py"):
    """
    Check the converted .py file the way Gradescope reads it.

    Looks for missing or duplicated task tags, syntax errors, cells that still
    raise NotImplementedError, imports that fail on Gradescope, and names a
    graded cell uses but Gradescope doesn't give it.
    """
    def problems():
        file = Path(path)
        if not file.exists():
            yield f"{file} doesn't exist. Run the conversion cell first."
            return

        source = file.read_text(encoding="utf-8")

        try:
            compile(source, str(file), "exec")
        except SyntaxError as exc:
            yield (
                f"The file has a syntax error on line {exc.lineno}: {exc.msg}. "
                "Gradescope can't grade a file with a syntax error, so every test would score 0."
            )
            return

        line_starts = [0]
        for line in source.splitlines(keepends=True):
            line_starts.append(line_starts[-1] + len(line))

        for task in _TASKS:
            tag = _TAG + task
            exact = [
                start
                for start, end in zip(line_starts, line_starts[1:])
                if source[start:end].strip() == tag
            ]
            if not exact:
                yield f"{task}: its tag line is missing. Don't edit or delete the tag line at the top of a graded cell."
                continue
            first = source.find(tag)
            if first != exact[0]:
                line = source.count("\n", 0, first) + 1
                yield (
                    f"{task}: the tag text also appears earlier, on line {line}, so Gradescope "
                    "reads your answer from the wrong place. Remove that copy."
                )
            if len(exact) > 1:
                yield f"{task}: the tag line appears {len(exact)} times. Gradescope only grades the first; delete the other copies."

        tasks = _extract_tasks(source)
        rf_tuned_defines = set()
        if "rf_tuned" in tasks:
            try:
                rf_tuned_defines = _bound_names(ast.parse(tasks["rf_tuned"]))
            except SyntaxError:
                pass

        for task in _TASKS:
            if task in tasks:
                extra = rf_tuned_defines if task == "rf_tuned_params" else set()
                for problem in _cell_problems(task, tasks[task], extra):
                    yield f"{task}: {problem}"

    return _run("submission_file", problems)
