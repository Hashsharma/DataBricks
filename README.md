# MLflow on Databricks 🚀

This repository documents my learning journey with **MLflow** on **Databricks**, focusing on building, training, and tracking machine learning models using **XGBoost**.

The goal of this project is to understand how MLflow helps manage the **end-to-end ML lifecycle**, including experiment tracking, model logging, and model versioning.

---

## 📌 Technologies Used

- **Databricks**
- **MLflow**
- **XGBoost**
- **Python**
- **Apache Spark (Databricks Runtime)**
- **scikit-learn**
- **Pandas / NumPy**

---

## 📖 What I’m Learning

- Setting up MLflow experiments on Databricks  
- Tracking parameters, metrics, and artifacts using MLflow  
- Training machine learning models using **XGBoost**
- Logging and registering models with MLflow
- Comparing multiple model runs
- Understanding model versioning and reproducibility

---

## 🧠 Project Structure


---

## ⚙️ MLflow Features Used

- **Experiment Tracking**
  - Log parameters (e.g., learning rate, max depth)
  - Log metrics (e.g., accuracy, RMSE)
  - Log artifacts (models, plots)

- **Model Logging**
  - Log XGBoost models using `mlflow.xgboost`
  - Save models for reuse and deployment

- **Model Registry (Databricks)**
  - Register trained models
  - Track model versions and stages (Staging / Production)

---

## 🧪 Example Workflow

1. Load and preprocess data
2. Train an XGBoost model
3. Track experiments using MLflow
4. Compare multiple runs
5. Register the best-performing model

---

## 📝 Sample MLflow Code

```python
import mlflow
import mlflow.xgboost
import xgboost as xgb

with mlflow.start_run():
    model = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=100
    )
    
    model.fit(X_train, y_train)

    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("n_estimators", 100)

    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.xgboost.log_model(model, "xgboost_model")
