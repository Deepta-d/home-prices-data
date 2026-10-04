# House Prices Machine Learning Project

This repository compares several scikit-learn models on a 10,000-row house price dataset. The main task is **regression**: predict continuous `price`. A **classifier** is also trained after grouping prices into Low / Medium / High bands, so Random Forest Classifier (RFC) can be included without treating dollar amounts as class labels.

Notebook: [`House Prices Data - Implementation of Machine Learning models.ipynb`](House%20Prices%20Data%20-%20Implementation%20of%20Machine%20Learning%20models.ipynb)

## Dataset

File: `house_prices_dataset.csv`

| Column | Role |
|---|---|
| `square_feet` | Feature |
| `num_rooms` | Feature |
| `age` | Feature |
| `distance_to_city(km)` | Feature |
| `price` | Target (continuous) |

There are no missing values. A small number of rows have negative prices; those rows are dropped before modeling.

## Workflow

1. Load and inspect the CSV.
2. Drop invalid (negative) prices.
3. Explore relationships with scatter and box plots.
4. Add `age_x_distance` as a model feature. `price_per_sqft` is used only for EDA because it is derived from the target.
5. Use one train/test split (80/20, `random_state=42`) for all regressors.
6. Fit Linear Regression, Decision Tree, and Random Forest.
7. Tune the tree models with GridSearchCV and RandomizedSearchCV.
8. Compare models with MSE, RMSE, and R².
9. Bin `price` into three quantile bands and train Decision Tree Classifier + Random Forest Classifier (accuracy).

## Models

**Regression (predict `price`)**

- Linear Regression
- Decision Tree Regressor (base and GridSearchCV)
- Random Forest Regressor (base, GridSearchCV, RandomizedSearchCV)

**Classification (predict price band)**

- Decision Tree Classifier
- Random Forest Classifier

Regression results should be compared with **RMSE** and **R²**. Classifier **accuracy** is a different task and should not be mixed into the same ranking as RMSE.

## How to run

1. Clone this repository.
2. Install Python 3 and:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

3. Open the notebook in Jupyter or VS Code/Cursor and run all cells from top to bottom.

Keep `house_prices_dataset.csv` in the same folder as the notebook. Random Forest grid search can take several minutes.

## Repository contents

```
.
├── House Prices Data - Implementation of Machine Learning models.ipynb
├── house_prices_dataset.csv
├── README.md
└── .gitignore
```
