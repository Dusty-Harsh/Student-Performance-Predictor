# My First Machine Learning Projects: Student Performance Prediction

This repository contains two complete machine learning projects that predict student performance. This file serves as a step-by-step journal of how each project was built, which libraries were used, and what key concepts were learned.

---

## Core Libraries & Key Functions Used

This "cheat sheet" explains the main tools used in these projects:

* **`pandas`** (Data Manipulation)
    * **`pd.read_csv()`**: Used to load the `.csv` files into a DataFrame (a table-like structure).
    * **`df.info()`**: The *most important* first step. Used to see all column names, their data types (like `int64` or `object`), and if there was any missing data.
    * **`df.head()`**: Used to look at the first 5 rows of the data.
    * **`df.drop([...], axis=1)`**: Used to create our `X` (features) DataFrame by *dropping* the columns we wanted to predict (the target) or that were "leaky".
    * **`df.select_dtypes(include='object')`**: Used in Project 2 to automatically select *only* the text-based columns for inspection.
    * **`pd.get_dummies(df)`**: The *magic* function for Project 2. This automatically found all text columns and converted them into new numerical (0 or 1) columns. This process is called **One-Hot Encoding**.

* **`scikit-learn` (sklearn)** (The Machine Learning Engine)
    * **`model_selection.train_test_split()`**: Used in both projects to split our `X` and `y` data into two parts: a *training set* (for the model to learn from) and a *testing set* (to check its performance). The `random_state=42` was used to ensure the "random" split is the same every time, making my results reproducible.
    * **`tree.DecisionTreeClassifier()`**: The model used in Project 1. A simple, tree-based model perfect for "classification" tasks (predicting a category like `GradeClass`).
    * **`ensemble.RandomForestRegressor()`**: The model used in Project 2. A more powerful, tree-based model for "regression" tasks (predicting a number like `G3`).
    * **`model.fit(X_train, y_train)`**: The "training" command. This tells the model to learn the patterns between the training features (`X_train`) and the training answers (`y_train`).
    * **`model.predict(X_test)`**: The "prediction" command. After training, this asks the model to generate its best guess for the answers based on the unseen test features (`X_test`).
    * **`model.feature_importances_`**: The "insight" property. After training, this was used to ask the model *what features it thought were most important* in making its predictions.
    * **`metrics.accuracy_score()`**: The "evaluator" for Project 1. It compares the model's predictions (`y_pred`) to the *actual* answers (`y_test`) and gives a score (e.g., 97%).
    * **`metrics.mean_absolute_error()`**: The "evaluator" for Project 2. Since "accuracy" doesn't work for regression, this was used to find the *average error* of the predictions (e.g., 2.04 grade points).

* **`matplotlib` & `seaborn`** (Plotting)
    * Used to create bar charts of the `feature_importances_` to get a clean visual of what the model "thought" was most important.

---

## Project 1: Grade Class Predictor (Classification)

* **File:** `Student_performance_data _.csv`
* **Goal:** To predict a student's `GradeClass` (a category from 0-4).
* **Model:** `DecisionTreeClassifier`

### Step-by-Step Workflow:

1.  **Load & Explore:** Loaded the data. `df.info()` showed all columns were already numbers (`int64` or `float64`), so no cleaning was needed.
2.  **Define Features & Target:**
    * `y = df['GradeClass']`
    * `X = df.drop(['GradeClass', 'StudentID'], axis=1)`
3.  **Train & Evaluate (Attempt 1):**
    * Split the data using `train_test_split`.
    * Trained a `DecisionTreeClassifier` using `.fit()`.
    * Made predictions using `.predict()`.
    * Checked accuracy with `accuracy_score()` and got **97%**.
4.  **Investigate (The "Aha!" Moment):**
    * An accuracy of 97% seemed *too good*.
    * I used `model.feature_importances_` to see *why* it was so accurate.
    * **Finding:** The model was "cheating." It found that `GPA` had an importance of ~86%. The `GradeClass` is just the `GPA` put into a category, so the model was just looking at the answer. This is called **Data Leakage**.
5.  **Re-train Realistic Model (Attempt 2):**
    * Created a new, honest `X` by dropping `GPA`: `X_real = df.drop(['GradeClass', 'StudentID', 'GPA'], axis=1)`
    * Re-trained and re-evaluated the model.
6.  **Final Insight:**
    * Checked `model_real.feature_importances_` again.
    * **Conclusion:** The new, realistic model showed that **`Absences`** was the single most important factor in predicting a student's grade class (when not cheating by looking at `GPA`).

---

## Project 2: Final Grade Predictor (Regression)

* **File:** `student-por.csv`
* **Goal:** To predict a student's *exact* final grade `G3` (a number from 0-20).
* **Model:** `RandomForestRegressor`

### Step-by-Step Workflow:

1.  **Load & Explore:** Loaded the data. `df.info()` showed a major challenge: **17 columns were `object` type (text)**. The model cannot read text, so this data was "messy" and needed cleaning.
2.  **Define Features & Target:**
    * `y = df['G3']`
    * `X = df.drop(['G3', 'G1', 'G2'], axis=1)`
    * **Note:** `G1` and `G2` (first and second period grades) were also dropped. This is another form of **data leakage**, as my goal was to predict the final grade *before* any exams were taken.
3.  **Data Preprocessing (The "Cleaning" Step):**
    * This was the main challenge.
    * I used **`X = pd.get_dummies(X)`** to solve it.
    * This single function automatically found all 17 text columns, threw them away, and replaced them with new, numerical (0 or 1) columns.
    * Example: The single `Mjob` column (text) was replaced with 5 new columns: `Mjob_teacher`, `Mjob_health`, etc.
    * This expanded my `X` from 30 features to **56 features**, all of which were now numbers.
4.  **Train & Evaluate:**
    * Split the new, clean `X` and `y` using `train_test_split`.
    * Trained a `RandomForestRegressor` (a regression model) using `.fit()`.
    * Made predictions using `.predict()`.
    * Evaluated the model using **`mean_absolute_error(y_test, y_pred)`**.
5.  **Final Insight:**
    * The model's **Mean Absolute Error (MAE) was 2.04**. This means, on average, the model's prediction is only 2.04 points away from the student's *actual* final grade.
    * Checked `model.feature_importances_`.
    * **Conclusion:** The two most important predictors for a student's final grade were **`failures`** (number of past class failures) and **`absences`**.

## How to Use My Models (Custom Predictions)

* **Project 1:** I can use `model_real.predict([[...]])` with a list of **12** numerical features.
* **Project 2:** I must provide a list of **56** numerical features to `model.predict([[...]])`, matching the exact order of the columns `pd.get_dummies()` created.
