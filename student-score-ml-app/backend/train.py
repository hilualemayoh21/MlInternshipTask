import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("data/student_performance.csv")

# Data Cleaning
df.drop_duplicates(inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Basic Visualization
plt.scatter(df["Hours_Studied"], df["Exam_Score"])
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")
plt.savefig("model/study_vs_score.png")
plt.close()

# Feature Selection
features = ["Hours_Studied"]
X = df[features]
y = df["Exam_Score"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_test)

print("Linear Regression R2:", r2_score(y_test, y_pred))
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

poly_model = LinearRegression()
poly_model.fit(X_train_p, y_train_p)
y_pred_poly = poly_model.predict(X_test_p)

print("Polynomial R2:", r2_score(y_test_p, y_pred_poly))

# Multiple Features
multi_features = ["Hours_Studied", "Sleep_Hours", "Attendance", "Previous_Scores"]

X_multi = df[multi_features]
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)

multi_model = LinearRegression()
multi_model.fit(X_train_m, y_train_m)
y_pred_multi = multi_model.predict(X_test_m)

print("Multiple Feature R2:", r2_score(y_test_m, y_pred_multi))

# Cross Validation + Regularization
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures()),
    ("regressor", Ridge())
])

param_grid = {
    "poly__degree": [1, 2],
    "regressor": [Ridge(), Lasso()],
    "regressor__alpha": [0.01, 0.1, 1, 10]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="r2"
)

grid.fit(X_multi, y)

print("Best Parameters:", grid.best_params_)

joblib.dump(grid, "model/model.pkl")
