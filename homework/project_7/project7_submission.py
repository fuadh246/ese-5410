#!/usr/bin/env python
# coding: utf-8

# # Project 7
# 
# In addition to answering the bolded questions on Coursera, also attach your notebook, both as
# `.ipynb` and `.html`.
# 
# A hospital in Philadelphia is trying to better streamline its operations and plan for the future by reducing the number hospitalizations that could have been prevented. They have approached you, a data scientist, to help predict if patients will develop heart diseases so that healthcare providers have a chance to intervene much earlier. For this task, you have been provided with the Heart dataset, containing a binary outcome 'HD' that indicates the presence of heart disease in 303 patients. There are 13 predictors including 'Age', 'Sex', 'Chol' (a cholesterol measurement), and other heart and lung function measurements.
# 
# 
# In this version of the assignment, the original PennGrader checks have been replaced with local notebook checks for immediate feedback. Official scoring is performed in Gradescope using hidden tests. The assignment questions, variable names, model specifications, and notebook flow are otherwise preserved.
# 
# <b>NOTE：Please remember to remove the </b>
# 
# ```python
# raise notImplementedError
# ```
# <b>after your implementation, otherwise the cell will not compile.</b>
# 

# ## How this notebook is graded
# 
# Read this section before you start. Most points lost on this project come from
# these rules, not from the machine learning.
# 
# **What you submit.** At the end you convert this notebook to
# `project7_submission.py` and upload that file to Gradescope. Gradescope runs 20
# tests worth 1 point each; the table below lists them. The questions keep their
# original numbers (there is no question 5), so they line up with the quiz.
# 
# **Notes for Gradescope.** Under each graded question there's a numbered list
# headed *Notes for Gradescope*. The question tells you what to do; the notes tell
# you how Gradescope checks it:
# 
# - the variable your answer must be stored in;
# - settings that must be exact, such as `random_state=42`, a search grid or
#   `cv=10`;
# - mistakes that cost the point even though your notebook looks right;
# - what to do instead, where an older instruction in the question (such as the
#   Codio steps) no longer applies.
# 
# Read the notes before you write the cell. When a public check fails, the fix is
# usually in that question's notes.
# 
# **How Gradescope runs your code.** Gradescope doesn't run your notebook from top
# to bottom. It runs each graded cell by itself, in a fresh Python session, on its
# own hidden copy of the data. Before running a cell, it loads the setup imports
# and the results of the earlier questions that the cell needs, for example
# `heart` for question 3, or the four split variables for the model questions.
# So:
# 
# - Use only those, plus what the cell defines itself. A helper function or a
#   variable from an ungraded cell doesn't exist on Gradescope, so the cell fails
#   there with a `NameError`, even though it runs in your notebook. Where a cell
#   gets less than you might expect, its notes say so.
# - Compute every answer with code. Don't type in numbers or column names you
#   read off your output; on the hidden data they're different. The
#   multiple-choice answers are the only exception.
# - The setup imports are `pd`, `np`, `tree`, `DecisionTreeClassifier`,
#   `BaggingClassifier`, `RandomForestClassifier`, `train_test_split`,
#   `GridSearchCV`, `precision_score`, `recall_score`, `accuracy_score`, `Image`
#   and `pydotplus`. Nothing else is loaded, so don't use `f1_score`,
#   `confusion_matrix`, `plt` or `sns` in a graded cell.
# - Don't write `import` lines in graded cells. Gradescope swaps in a faster
#   `GridSearchCV`, and importing `GridSearchCV` yourself undoes that, so the test
#   fails. `seaborn`, `matplotlib`, `graphviz` and `pydot` aren't installed on
#   Gradescope at all.
# - Don't use `display()` in graded cells. Use `print()`.
# - Keep plots and exploration in ungraded cells.
# - Don't add notebook commands (lines starting with `%` or `!`, such as
#   `!pip install`) **anywhere** in the notebook. Depending on how the notebook
#   is converted, such a line can make the whole `.py` file invalid Python, and
#   Gradescope then gives 0 for every test.
# 
# **Graded cells.** Each graded cell starts with a tag line beginning with
# `# PROJECT7_TASK`, followed by a `# Define:` line that names the variables
# Gradescope reads.
# 
# - Write your answer under these lines and delete the `raise NotImplementedError`
#   line.
# - Don't change, move or copy the tag line, and don't split the cell or merge it
#   with another cell.
# - Right after each graded cell comes its public-check cell, whose first line
#   starts with `# In[ ]:`. Keep that line, and don't insert cells between a
#   graded cell and its check cell: Gradescope uses that line to find where your
#   answer ends.
# 
# **Public checks.** After each graded cell there's a cell that runs a check from
# `public_checks.py`. It looks at your results on the real `Heart.csv` and tests
# what Gradescope tests: variable names and types, settings such as
# `random_state=42` and `cv=10`, the exact search grids, and whether your metrics
# match your model. If a check fails, it lists what to fix; fix it before moving
# on. For the three multiple-choice questions, the checks look only at the format
# of your answer, not whether it's right.
# 
# **Before you submit.** The last section converts the notebook and runs
# `public_checks.check_submission_file()`, which reads the converted file the way
# Gradescope does and reports problems such as names from other cells or imports
# in graded cells.

# ### The 20 Gradescope tests
# 
# | Question | What it grades | Variables | Points |
# |---|---|---|---|
# | 1 | Load and clean the data | `heart` | 1 |
# | 2 | Encode `HD`, `ChestPain` and `Thal` | `heart` | 1 |
# | 3 | Minimum and maximum age | `min_age`, `max_age` | 1 |
# | 4 | Highest positive correlation with `HD` | `highest_corr` | 1 |
# | 6 | Train/test split | `X_train`, `X_test`, `y_train`, `y_test` | 1 |
# | Part A, 1 | Base decision tree | `clf` | 1 |
# | Part A, 2 | Tree plot | `img` | 1 |
# | Part A, 3 | Base tree metrics | `dt_precision`, `dt_recall`, `dt_accuracy` | 1 |
# | Part A, 4a | Multiple choice: what to tune | `answers` | 1 |
# | Part A, 4b | Tuned decision tree and its metrics | `grid_search_cv_dt`, `dt_tuned_precision`, `dt_tuned_recall`, `dt_tuned_accuracy` | 1 |
# | Part B, 1 | Base bagging model | `bag` | 1 |
# | Part B, 2 | Base bagging metrics | `bag_precision`, `bag_recall`, `bag_accuracy` | 1 |
# | Part B, 3 | Tuned bagging model and its metrics | `grid_search_cv_bag`, `bag_tuned_precision`, `bag_tuned_recall`, `bag_tuned_accuracy` | 1 |
# | Part B, 3 | Best bagging hyperparameters | `bag_best_param` | 1 |
# | Part C, 1 | Base random forest | `rf` | 1 |
# | Part C, 2 | Base forest metrics | `rf_precision`, `rf_recall`, `rf_accuracy` | 1 |
# | Part C, 3 | Tuned random forest and its metrics | `grid_search_cv_rf`, `rf_tuned_precision`, `rf_tuned_recall`, `rf_tuned_accuracy` | 1 |
# | Part C, 3 | Best forest hyperparameters | `rf_tuned` | 1 |
# | 10 | Multiple choice: best base model | `choice` | 1 |
# | 11 | Multiple choice: best tuned model | `choice` | 1 |
# |  | **Total** |  | **20** |

# ## Getting Set Up
# Enter your Student ID and name below for your notebook record. Gradescope identifies your submitted file separately.
# 

# You also need two files in the same folder as this notebook:
# 
# - `Heart.csv`, the data;
# - `public_checks.py`, the public checks.
# 
# On Google Colab, upload both with the **Files** panel on the left (folder icon,
# then the upload button). Colab deletes uploaded files when the runtime
# restarts, so upload them again after a restart.
# 
# Run the next three cells in order. The first one installs `pydotplus`, which
# Part A question 2 needs to draw the tree.

# In[1]:


# Installs pydotplus for the tree plot in Part A, question 2.
# Colab already has the Graphviz program. On your own computer you also need it:
#   conda install graphviz      (or, on macOS: brew install graphviz)
# Plain Python instead of "%pip install": a % or ! line is not valid Python, so
# some ways of converting the notebook to .py would make Gradescope reject the file.
import subprocess
import sys

try:
    import pydotplus
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "pydotplus"], check=True)


# In[2]:


# Data handling
import numpy as np
import pandas as pd

# Plotting (use these in ungraded cells only)
import matplotlib.pyplot as plt
import seaborn as sns
import pydotplus
from IPython.display import Image

# Models, tuning and metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)

# Public checks: public_checks.py must be in the same folder as this notebook
import public_checks


# In[3]:


#PLEASE ENSURE YOUR STUDENT ID AND NAME ARE ENTERED CORRECTLY

STUDENT_ID = 27435656           # YOUR STUDENT ID GOES HERE
STUDENT_NAME = "Fuad Hassan"      # YOUR FULL NAME GOES HERE


# Now we are ready for this project. Before building our models, let's take a look at how the data looks like.
# 1. Load the Heart data and drop any rows with `NaN/null` values.

# *Notes for Gradescope*:
# 
# 1. Store the data in `heart`. Read it with `pd.read_csv("Heart.csv")`: exactly this file name, with no folder in front of it and no `index_col`. Gradescope supplies its own `Heart.csv` under that name.
# 2. Drop every row that has at least one missing value.
# 3. On the real data you should get 297 rows and 14 columns (6 of the 303 rows have a missing value).

# In[4]:


# PROJECT7_TASK:read_data
# Define: heart
heart = pd.read_csv("Heart.csv")
heart = heart.dropna()


# In[5]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_read_data(heart)


# In[6]:


heart.head()


# 2. Binarize the 'HD' values such that `No=0` and `Yes=1`. One hot encode categorical features, and set prefixes to original column names. Produce some numerical and graphical summaries of it. Do there appear to be any patterns? 
# 
# Hint: Don't forget to drop original columns after one hot encode.
# 
# The order of your columns should be as follows:  
# 
# * Age	
# * Sex	
# * RestBP	
# * Chol	
# * Fbs
# * RestECG
# * MaxHR
# * ExAng
# * Oldpeak
# * Slope
# * Ca
# * HD
# * ChestPain_asymptomatic
# * ChestPain_nonanginal
# * ChestPain_nontypical
# * ChestPain_typical
# * Thal_fixed
# * Thal_normal
# * Thal_reversable

# *Notes for Gradescope*:
# 
# 1. Store the encoded data back in `heart`, with the same rows as before.
# 2. Turn `HD` into the integers 0 and 1 with `Series.map` and a dictionary. Don't assign numbers with `heart.loc[...] = 1`, as older versions of this project did: Gradescope's version of pandas raises a `TypeError` for it.
# 3. One-hot encode `ChestPain` and `Thal` and nothing else, with one column for every category (no `drop_first=True`). The new columns may hold `True`/`False`, which is what `pd.get_dummies` gives, or 1/0.
# 4. Gradescope checks the column order, so keep exactly the order listed above. It's the order you get when the new columns are added at the end, `ChestPain` first and `Thal` second.
# 5. Put the summaries and plots in the ungraded cells after the check, not in this cell.
# 6. On the real data you should get 297 rows and 19 columns.

# In[7]:


# PROJECT7_TASK:hot_encode
# Define: heart (encoded)
heart["HD"] = heart["HD"].map({"No": 0, "Yes": 1})
encoded_categories = pd.get_dummies(
    heart[["ChestPain", "Thal"]],
    prefix=["ChestPain", "Thal"],
)
heart = pd.concat(
    [heart.drop(columns=["ChestPain", "Thal"]), encoded_categories],
    axis=1,
)


# In[8]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_hot_encode(heart)


# **Numerical and graphical summaries for question 2 (ungraded).** Some ideas:
# `heart.describe()`, a correlation heatmap with `sns.heatmap(heart.corr())`, or
# `sns.pairplot` on a few columns (a pairplot of all 19 columns is slow).

# In[9]:


# Ungraded: your summaries and plots
heart.describe()


# In[10]:


plt.figure(figsize=(14, 10))
sns.heatmap(heart.corr(numeric_only=True), center=0)
plt.show()


# In[11]:


heart.corr().style.background_gradient(cmap="coolwarm")


# **Do there appear to be any patterns? (ungraded)**
# 
# Heart disease is most positively associated with the ChestPain_asymptomatic and Thal_reversable features, and most negatively associated with the ChestPain_nonanginal and Thal_normal features.

# 3. What is the minimum and maximum ages of the patients in the dataset? Assign this value to the variable `min_age` and `max_age`.

# *Notes for Gradescope*:
# 
# 1. Store the smallest and the largest value of `heart["Age"]` in `min_age` and `max_age`, as numbers, and print them.

# In[12]:


# PROJECT7_TASK:age
# Define: min_age, max_age

# YOUR CODE HERE
min_age = heart["Age"].min()
max_age = heart["Age"].max()
print(f"Minimum age: {min_age}, Maximum age: {max_age}")


# In[13]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_age(heart, min_age, max_age)


# 4. Calculate the pairwise correlations between all the variables. Which predictor has the highest positive correlation with the response variable? Assign this to `highest_corr`. 
# 
# Hint: Your answer should be in the format `ColumnName_type` where the type is the result of one-hot encoding. 

# *Notes for Gradescope*:
# 
# 1. Store the predictor's exact column name in `highest_corr`, as a string, for example `"Thal_normal"` (this example isn't the answer).
# 2. Leave out `HD` itself: its correlation with itself is always 1.
# 3. Compute the name from the correlation matrix, for example from `heart.corr()`. Don't type it in: Gradescope runs this cell on different data.

# In[14]:


# PROJECT7_TASK:corr
# Define: highest_corr

highest_corr = heart.corr(numeric_only=True)["HD"].drop("HD").idxmax()
highest_corr


# In[15]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_corr(heart, highest_corr)


# 6. Since we are interested in building a predictive model, it is good practice to split the data into training and testing sets. Using `sklearn.model_selection.train_test_split`, divide the data into these sets using a 80/20 split with a `random_state=42`. These training and testing sets will be used for all the following parts.

# *Notes for Gradescope*:
# 
# 1. Store every column of `heart` except `HD` in `X` (18 predictors, in the same order), and the `HD` column in `y`.
# 2. Split with `train_test_split(X, y, test_size=0.2, random_state=42)` and unpack the result as `X_train, X_test, y_train, y_test`, in that order.
# 3. On the real data `X_train` is 237 × 18 and `X_test` is 60 × 18.

# In[16]:


# PROJECT7_TASK:split
# Define: X, y, X_train, X_test, y_train, y_test

# YOUR CODE HERE
X = heart.drop(columns=["HD"])
y = heart["HD"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_split(heart, X_train, X_test, y_train, y_test)


# ## 7. Part A: Decision Tree

# 1. Using all predictor variables, train a <u>base</u> classification tree to predict the response variable. Use `sklearn.tree.DecisionTreeClassifier` and set `random_state=42`. 
# 
# *Hint*: here is the link for the documentation of [sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).

# *Notes for Gradescope*:
# 
# 1. Store the tree in `clf`: `DecisionTreeClassifier(random_state=42)` with every other setting at its default, fitted on `X_train` and `y_train`.
# 2. Store its predictions for `X_test` in `y_pred`; question 3 uses them.

# In[18]:


# PROJECT7_TASK:dt
# Define: clf, y_pred

# YOUR CODE HERE
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[19]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_dt(clf, X_train)


# 2. Plot the decision tree using the `Graphviz` package. Do you notice anything interesting?
# 
# *Hint*: To do this, click on the little computer button on the Codio sidebar that says "Open Terminal." Once you are in the terminal, copy and paste the following and hit enter:
# 
# ```
# pip install --upgrade pip --user
# pip install pydotplus --user
# pip install graphviz --user
# pip install ipython --user
# sudo apt-get install graphviz
# ```
# 
# After, save your work and then click kernel -> 'restart and run all'. You should be able to use `Graphviz` package now!
# 
# If any of the above commands fails, please replace 'pip install' with 'sudo apt-get install `<package-name>`'. If you are having trouble with the installation, please post on Piazza or visit office hours.
#     
# To pass the test, your img should be of type `<class 'IPython.core.display.Image'>`.

# *Notes for Gradescope*:
# 
# 1. Store the picture in `img`, built in three steps: `dot_data = tree.export_graphviz(clf, out_file=None, ...)`, then `graph = pydotplus.graph_from_dot_data(dot_data)`, then `img = Image(graph.create_png())`.
# 2. You may add options such as `feature_names=list(X_train.columns)` and `filled=True`, but not `special_characters=True`: `pydotplus` can't read the labels it produces.
# 3. Gradescope doesn't have Graphviz, so this is the only way of drawing the tree it accepts. If you want to try `plot_tree` or `graphviz.Source`, do it in an ungraded cell.
# 4. Skip the Codio steps above; the install cell under *Getting Set Up* replaces them. On your own computer you also need the Graphviz program (see that cell).

# In[20]:


# PROJECT7_TASK:image
# Define: img

# YOUR CODE HERE
dot_data = tree.export_graphviz(
    clf,
    out_file=None,
    feature_names=list(X_train.columns),
    class_names=["No disease", "Disease"],
    filled=True,
    rounded=True,
)
graph = pydotplus.graph_from_dot_data(dot_data)
img = Image(graph.create_png())


# In[21]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_image(img)


# In[22]:


img


# In[23]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_image(img)


# In[24]:


img


# In[25]:


with open("output_image.png", "wb") as f:
    f.write(img.data)


# **Do you notice anything interesting? (ungraded)**
# 
# The tree makes a sequence of threshold-based decisions, so it is possible to follow a path from the root to a predicted class.

# 3. Evaluate your base model on the test set by calculating the precision, recall and accuracy. Assign these values to `dt_precision`, `dt_recall` and `dt_accuracy` respectively. Name your model `dt`. **Be sure to print your precision, recall, and accuracy for use later** 
# 
# *Hint*: A useful resource if you need a refresher of these terms is [here](https://en.wikipedia.org/wiki/Precision_and_recall).

# *Notes for Gradescope*:
# 
# 1. Store the test-set metrics of `clf` in `dt_precision`, `dt_recall` and `dt_accuracy`, using `precision_score`, `recall_score` and `accuracy_score`, and print them.
# 2. Pass the true labels first, as in `precision_score(y_test, y_pred)`. Swapping the two arguments swaps precision and recall.
# 3. The question says to name your model `dt`; you don't need to. Gradescope uses `clf` from question 1.

# In[26]:


# PROJECT7_TASK:dt_score
# Define: dt_precision, dt_recall, dt_accuracy

# YOUR CODE HERE
dt_precision = precision_score(y_test, y_pred)
dt_recall = recall_score(y_test, y_pred)
dt_accuracy = accuracy_score(y_test, y_pred)
print(f"Precision: {dt_precision}")
print(f"Recall: {dt_recall}")
print(f"Accuracy: {dt_accuracy}")


# In[27]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_dt_score(clf, X_test, y_test, dt_precision, dt_recall, dt_accuracy)


# 4a. Which hyperparameter should we tune when building a decision tree? Select all that applies:
# 
# A. Tree depth
# 
# B. Number of estimators
# 
# C. Number of output classes
# 
# D. Percent of samples trained

# *Notes for Gradescope*:
# 
# 1. Store your answer in `answers` as a list of uppercase letters, for example `["B"]` or `["B", "D"]` (these examples aren't the answer).
# 2. The public check only looks at the format of your answer, not whether it's right.

# In[28]:


# PROJECT7_TASK:dt_tune
# Define: answers, e.g. ["B"] or ["B", "D"]

answers = ["A"]


# In[29]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_dt_tune(answers)


# 4b. Tune the hyperparameter you selected in the above question and evaluate this tuned model by calculating the precision, recall and accuracy on the test set. Assign these calculated values to `dt_tuned_precision`, `dt_tuned_recall` and `dt_tuned_accuracy` respectively. **Be sure to print your precision, recall, and accuracy for use later**
# 
# *Hint*: 
# 1. Refer to the documentation to determine what can be tuned. 
# 2. For each tuned parameter, search through range between 3 and 20 (inclusive: `np.arange(3, 21)`) to find the best performing parameters. 
# 3. To tune hyperparameters, use GridSearchCV with `cv=10`. This will apply KFold on your training set so you split your training set into a smaller training set and a "validation" set, which GridSearchCV will use to pick the best parameters. If you do not specify `cv=10`, GridSearchCV will by default use 5-fold cross validation. Read more about train/validation/test sets [here](https://blog.roboflow.com/train-test-split/). 

# *Notes for Gradescope*:
# 
# 1. Store the search in `grid_search_cv_dt`: a `GridSearchCV` with `DecisionTreeClassifier(random_state=42)`, the grid `{"max_depth": np.arange(3, 21)}` and `cv=10`, fitted on `X_train` and `y_train`.
# 2. Create the tree inside `GridSearchCV`, as in note 1. Gradescope runs this cell by itself, and `clf` from question 1 doesn't exist there.
# 3. Write `cv=10` as the number 10, not as a `KFold` object, and leave `scoring` at its default (accuracy).
# 4. Compute `dt_tuned_precision`, `dt_tuned_recall` and `dt_tuned_accuracy` from `grid_search_cv_dt.predict(X_test)`. Don't refit a tree with a depth you typed in; on Gradescope's data the best depth is different.
# 5. Print `grid_search_cv_dt.best_params_` and the three metrics.

# In[30]:


# PROJECT7_TASK:dt_tuned
# Define: grid_search_cv_dt, dt_tuned_precision, dt_tuned_recall, dt_tuned_accuracy

# YOUR CODE HERE

grid_search_cv_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={"max_depth": np.arange(3, 21)},
    cv=10,
)
grid_search_cv_dt.fit(X_train, y_train)

dt_tuned_precision = precision_score(y_test, grid_search_cv_dt.predict(X_test))
dt_tuned_recall = recall_score(y_test, grid_search_cv_dt.predict(X_test))
dt_tuned_accuracy = accuracy_score(y_test, grid_search_cv_dt.predict(X_test))

print(f"Best parameters: {grid_search_cv_dt.best_params_}")
print(f"Tuned Precision: {dt_tuned_precision}")
print(f"Tuned Recall: {dt_tuned_recall}")
print(f"Tuned Accuracy: {dt_tuned_accuracy}")


# In[31]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_dt_tuned(
    grid_search_cv_dt, X_test, y_test,
    dt_tuned_precision, dt_tuned_recall, dt_tuned_accuracy,
)


# ## 8. Part B: Bagging

# 1. Using all predictor variables, train a base model to predict the response variable. Use `sklearn.ensemble.BaggingClassifier` and set `random_state=42`. Name your model as `bag`.

# *Notes for Gradescope*:
# 
# 1. Store the model in `bag`: `BaggingClassifier(random_state=42)` with every other setting at its default, fitted on `X_train` and `y_train`.
# 2. Don't pass an `estimator`: the default base model is a decision tree, and the default number of trees is 10.

# In[32]:


# PROJECT7_TASK:bag
# Define: bag

# YOUR CODE HERE
bag = BaggingClassifier(random_state=42)
bag.fit(X_train, y_train)


# In[33]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_bag(bag, X_train, y_train)


# 2. Evaluate your base model on the test set by calculating the precision, recall and accuracy. Assign these values to `bag_precision`, `bag_recall`  and `bag_accuracy` respectively. **Be sure to print your precision, recall, and accuracy for use later**

# *Notes for Gradescope*:
# 
# 1. Store the test-set metrics of `bag` in `bag_precision`, `bag_recall` and `bag_accuracy`, and print them.
# 2. Compute `bag`'s predictions in this cell. In your notebook `y_pred` still holds the decision tree's predictions, and on Gradescope it doesn't exist in this cell.

# In[34]:


# PROJECT7_TASK:bag_score
# Define: bag_precision, bag_recall, bag_accuracy

# YOUR CODE HERE
bag_precision = precision_score(y_test, bag.predict(X_test))
bag_recall = recall_score(y_test, bag.predict(X_test))
bag_accuracy = accuracy_score(y_test, bag.predict(X_test))
print(f"Bagging Precision: {bag_precision}")
print(f"Bagging Recall: {bag_recall}")
print(f"Bagging Accuracy: {bag_accuracy}")


# In[35]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_bag_score(bag, X_test, y_test, bag_precision, bag_recall, bag_accuracy)


# In[36]:


# Ungraded: confusion matrix of the base bagging model
# (rows: true class 0, 1; columns: predicted class 0, 1)
confusion_matrix(y_test, bag.predict(X_test))


# 3. Do <b>not</b> modify `base_classifier`. Tune the `n_estimators` hyperparameter of your model and evaluate this tuned model by calculating the precision, recall, and accuracy on the test set. Assign these calculated values to `bag_tuned_precision`, `bag_tuned_recall` and `bag_tuned_accuracy`  respectively. **Be sure to print your precision, recall, and accuracy for use later.** Store your best performing estimators to `bag_best_param` as a dictionary ('tuned estimator' : number). *Hint*: Set the searching space to all integers between range 10 and 49 (inclusive: `np.arange(10, 50)`). To tune hyperparameters, use GridSearchCV with `cv=10`. This will apply KFold on your training set so you split your training set into a smaller training set and a "validation" set, which GridSearchCV will use to pick the best parameters. If you do not specify `cv=10`, GridSearchCV will by default use 5-fold cross validation. Read more about train/validation/test sets [here](https://blog.roboflow.com/train-test-split/). 

# *Notes for Gradescope*:
# 
# 1. Store the search in `grid_search_cv_bag`: a `GridSearchCV` with `BaggingClassifier(random_state=42)`, the grid `{"n_estimators": np.arange(10, 50)}` and `cv=10` (the number 10), fitted on `X_train` and `y_train`.
# 2. Create the model inside `GridSearchCV`, as in note 1. Gradescope runs this cell by itself, and `bag` from question 1 doesn't exist there.
# 3. "Do not modify `base_classifier`" means: don't pass an `estimator` (scikit-learn's current name for the base classifier).
# 4. Compute `bag_tuned_precision`, `bag_tuned_recall` and `bag_tuned_accuracy` from `grid_search_cv_bag.predict(X_test)`.
# 5. Store `grid_search_cv_bag.best_params_` in `bag_best_param`, in this same cell. It's a dictionary such as `{"n_estimators": 23}`, and Gradescope grades it as a separate point.
# 6. Print `bag_best_param` and the three metrics.

# In[37]:


# PROJECT7_TASK:bag_tuned
# Define: grid_search_cv_bag, bag_tuned_precision, bag_tuned_recall, bag_tuned_accuracy, bag_best_param

# YOUR CODE HERE
grid_search_cv_bag = GridSearchCV(
    BaggingClassifier(random_state=42),
    param_grid={"n_estimators": np.arange(10, 50)},
    cv=10,
)
grid_search_cv_bag.fit(X_train, y_train)
bag_tuned_precision = precision_score(y_test, grid_search_cv_bag.predict(X_test))
bag_tuned_recall = recall_score(y_test, grid_search_cv_bag.predict(X_test))
bag_tuned_accuracy = accuracy_score(y_test, grid_search_cv_bag.predict(X_test))
bag_best_param = grid_search_cv_bag.best_params_
print(f"Best parameters: {bag_best_param}")
print(f"Tuned Bagging Precision: {bag_tuned_precision}")
print(f"Tuned Bagging Recall: {bag_tuned_recall}")
print(f"Tuned Bagging Accuracy: {bag_tuned_accuracy}")


# In[38]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_bag_tuned(
    grid_search_cv_bag, X_test, y_test,
    bag_tuned_precision, bag_tuned_recall, bag_tuned_accuracy,
)


# In[39]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_bag_best(bag_best_param, grid_search_cv_bag)


# ## 9. Part C: Random Forest

# 1. Using all predictor variables, train a base random forest to predict the response variable. Use `sklearn.ensemble.RandomForestClassifier` and set `random_state=42`. Name your model `rf`.

# *Notes for Gradescope*:
# 
# 1. Store the forest in `rf`: `RandomForestClassifier(random_state=42)` with every other setting at its default (100 trees), fitted on `X_train` and `y_train`.

# In[40]:


# PROJECT7_TASK:rf
# Define: rf

# YOUR CODE HERE
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)


# In[41]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_rf(rf)


# 2. Evaluate your base model on the test set by calculating the precision, recall and accuracy. Assign these values to `rf_precision`, `rf_recall` and `rf_accuracy` respectively. **Be sure to print your precision, recall, and accuracy for use later**

# *Notes for Gradescope*:
# 
# 1. Store the test-set metrics of `rf` in `rf_precision`, `rf_recall` and `rf_accuracy`, and print them.
# 2. Compute `rf`'s predictions in this cell. In your notebook `y_pred` still holds the decision tree's predictions.

# In[42]:


# PROJECT7_TASK:rf_score
# Define: rf_precision, rf_recall, rf_accuracy

# YOUR CODE HERE
rf_precision = precision_score(y_test, rf.predict(X_test))
rf_recall = recall_score(y_test, rf.predict(X_test))
rf_accuracy = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest Precision: {rf_precision}")
print(f"Random Forest Recall: {rf_recall}")
print(f"Random Forest Accuracy: {rf_accuracy}")


# In[43]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_rf_score(rf, X_test, y_test, rf_precision, rf_recall, rf_accuracy)


# 3. Tune the hyperparameters of your model and evaluate this tuned model by calculating the precision, recall and accuracy on the test set. Assign these calculated values to `rf_tuned_precision`, `rf_tuned_recall` and `rf_tuned_accuracy` respectively. Store your best-performing parameters to `rf_tuned`. **Be sure to print your precision, recall, and accuracy for use later.** Also don't forget to set `random_state=42`. 
# 
# *Hint*: 
# 1. In this question, we would like to tune only the parameters determining the depth of a tree and the number of trees. But beware that you can also adjust the maximum number of features considered for each tree(max_features), the minimum number of data points placed in a node before the node is split(min_samples_split), the minimum number of data points allowed in a leaf node(min_samples_leaf), and bootstrap option. Refer to [sklearn.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) for a complete list.
# 2. In this question, your parameter search space is (3,20) for tree depth, (10,29) for number of trees, both inclusive (`np.arange(3, 21)` and `np.arange(10, 30)`).
# 3. To tune hyperparameters, use GridSearchCV with `cv=10`. This will apply KFold on your training set so you split your training set into a smaller training set and a "validation" set, which GridSearchCV will use to pick the best parameters. If you do not specify `cv=10`, GridSearchCV will by default use 5-fold cross validation. Read more about train/validation/test sets [here](https://blog.roboflow.com/train-test-split/). 

# *Notes for Gradescope*:
# 
# 1. Store the search in `grid_search_cv_rf`: a `GridSearchCV` with `RandomForestClassifier(random_state=42)`, the grid `{"max_depth": np.arange(3, 21), "n_estimators": np.arange(10, 30)}` and `cv=10` (the number 10), fitted on `X_train` and `y_train`. Leave every other setting at its default.
# 2. Create the forest inside `GridSearchCV`, as in note 1. Gradescope runs this cell by itself, and `rf` from question 1 doesn't exist there.
# 3. Compute `rf_tuned_precision`, `rf_tuned_recall` and `rf_tuned_accuracy` from `grid_search_cv_rf.predict(X_test)`. Don't refit a forest with values you typed in.
# 4. Print `grid_search_cv_rf.best_params_` and the three metrics.
# 5. The search fits 3,600 forests (360 combinations × 10 folds) and takes a few minutes. Adding `n_jobs=-1` to `GridSearchCV` uses all your CPU cores and gives the same result.
# 6. Gradescope grades this cell together with the next one, which stores `rf_tuned`. Finish both, or you lose both points.

# In[44]:


# PROJECT7_TASK:rf_tuned
# Define: grid_search_cv_rf, rf_tuned_precision, rf_tuned_recall, rf_tuned_accuracy

# YOUR CODE HERE
grid_search_cv_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={
        "max_depth": np.arange(3, 21),
        "n_estimators": np.arange(10, 30),
    },
    cv=10,
)
grid_search_cv_rf.fit(X_train, y_train)
rf_tuned_precision = precision_score(y_test, grid_search_cv_rf.predict(X_test))
rf_tuned_recall = recall_score(y_test, grid_search_cv_rf.predict(X_test))
rf_tuned_accuracy = accuracy_score(y_test, grid_search_cv_rf.predict(X_test))
print(f"Best parameters: {grid_search_cv_rf.best_params_}")
print(f"Tuned Random Forest Precision: {rf_tuned_precision}")
print(f"Tuned Random Forest Recall: {rf_tuned_recall}")
print(f"Tuned Random Forest Accuracy: {rf_tuned_accuracy}")


# In[45]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_rf_tuned(
    grid_search_cv_rf, X_test, y_test,
    rf_tuned_precision, rf_tuned_recall, rf_tuned_accuracy,
)


# *Notes for Gradescope*:
# 
# 1. Store `grid_search_cv_rf.best_params_` in `rf_tuned`: a dictionary with the keys `max_depth` and `n_estimators`.

# In[46]:


# PROJECT7_TASK:rf_tuned_params
# Define: rf_tuned

# YOUR CODE HERE
rf_tuned = grid_search_cv_rf.best_params_


# In[47]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_rf_tuned_params(rf_tuned, grid_search_cv_rf)


# **All test-set metrics side by side (ungraded).** The next cell collects your
# results in one table. Use it for questions 10 and 11.

# In[48]:


# Ungraded: all test-set metrics side by side
comparison = pd.DataFrame(
    {
        "base precision": [dt_precision, bag_precision, rf_precision],
        "base recall": [dt_recall, bag_recall, rf_recall],
        "base accuracy": [dt_accuracy, bag_accuracy, rf_accuracy],
        "tuned precision": [dt_tuned_precision, bag_tuned_precision, rf_tuned_precision],
        "tuned recall": [dt_tuned_recall, bag_tuned_recall, rf_tuned_recall],
        "tuned accuracy": [dt_tuned_accuracy, bag_tuned_accuracy, rf_tuned_accuracy],
    },
    index=["A. Decision Tree", "B. Bagging", "C. Random Forest"],
)
comparison


# 10. Of the 3 base models that you trained, which model achieved the highest test accuracy? 
# 
# A. Decision Tree
# 
# B. Bagging
# 
# C. Random Forest

# *Notes for Gradescope*:
# 
# 1. Store your answer in `choice` as a list with one uppercase letter, for example `["A"]`.
# 2. The public check only looks at the format of your answer, not whether it's right.

# In[49]:


# PROJECT7_TASK:base_choice
# Define: choice, e.g. ["A"]

choice = ["C"]


# In[ ]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_base_choice(choice)


# 11. Of the 3 models in which you tuned the parameters, which model achieved the highest test accuracy? 
# 
# A. Decision Tree
# 
# B. Bagging
# 
# C. Random Forest

# *Notes for Gradescope*:
# 
# 1. Store your answer in `choice` as a list with one uppercase letter, for example `["A"]`.
# 2. Question 10 also uses the name `choice`. That's fine: Gradescope runs each cell by itself.
# 3. The public check only looks at the format of your answer, not whether it's right.

# In[ ]:


# PROJECT7_TASK:tuned_choice
# Define: choice, e.g. ["A"]

choice = ["C"]


# In[ ]:


# In[ ]: end of the graded cell above, for Gradescope. Keep this line.
public_checks.check_tuned_choice(choice)


# 12. *Food for thought (ungraded but interesting to think about)*: In this project, we asked you to pick the model with the highest test accuracy. For this task, do you think that accuracy is the best metric to use, or would precision, recall or even the F1-score be better?

# in this case, recall may matter because a false negative could delay follow-up. Precision also matters because false positives can lead to unnecessary tests.
# 

# ## Submitting to Gradescope
# 
# **Step 1: public-check summary.** Every check above should say `[PASS]`.

# In[ ]:


public_checks.show_public_check_summary()


# **Step 2: convert the notebook to `project7_submission.py`.**
# 
# - First, save the notebook. On Colab, also download it: **File → Download →
#   Download .ipynb**.
# - Run the next cell. It defines `download_notebook_as_py()`.
# - In the cell after it, delete the `#` in front of `download_notebook_as_py()`
#   and run it. It writes `project7_submission.py`, checks that the file is valid
#   Python and, on Colab, downloads it. If Colab can't find your notebook, a file
#   picker opens: select the `.ipynb` you downloaded.
# 
# If you renamed this notebook, pass the new name, for example
# `download_notebook_as_py("my_notebook.ipynb")`.

# In[ ]:


from pathlib import Path
import subprocess
import sys


def download_notebook_as_py(
    ipynb_path="project7_student.ipynb",
    output_name="project7_submission.py",
):
    """
    Convert the completed notebook to a .py file, check that the result is
    valid Python, and download it when running in Google Colab.
    """
    ipynb_path = Path(ipynb_path)

    if not ipynb_path.exists():
        try:
            from google.colab import files

            print(
                f"Could not find {ipynb_path.name} in the runtime.\n"
                "Select your saved .ipynb notebook in the file picker."
            )
            uploaded = files.upload()
            notebook_names = [
                name for name in uploaded if name.lower().endswith(".ipynb")
            ]
            if len(notebook_names) != 1:
                raise ValueError("Please upload exactly one .ipynb notebook.")
            ipynb_path = Path(notebook_names[0])

        except ImportError:
            raise FileNotFoundError(f"Could not find notebook: {ipynb_path}")

    if ipynb_path.suffix.lower() != ".ipynb":
        raise ValueError("The input file must be a .ipynb file.")

    if not output_name.lower().endswith(".py"):
        output_name += ".py"
    output_path = Path(output_name)

    subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "python",
            str(ipynb_path),
            "--output", output_path.stem,
            "--output-dir", str(output_path.parent),
        ],
        check=True,
    )

    source = output_path.read_text(encoding="utf-8")
    compile(source, str(output_path), "exec")

    print(f"Created and verified: {output_path.resolve()}")
    print("Upload this .py file to Gradescope.")

    try:
        from google.colab import files
        files.download(str(output_path))
    except ImportError:
        print("Not running in Google Colab; the file remains on disk.")

    return str(output_path)


# In[ ]:


download_notebook_as_py()


# **Step 3: check the converted file.** This reads `project7_submission.py` the
# way Gradescope does. If it reports problems, fix them in the notebook, save,
# convert again and rerun this cell.

# In[ ]:


public_checks.check_submission_file("project7_submission.py")


# **Step 4: upload.** Upload `project7_submission.py` to Gradescope, keeping that
# file name. Upload only this one file: Gradescope never uses `public_checks.py`
# or your `Heart.csv`, and if it gets two `.py` files and neither is named
# `project7_submission.py`, it rejects the submission ("Multiple .py files were
# submitted"). Each of the 20 tests shows its result; a failed test shows a
# message saying what was wrong.

# In[ ]:




