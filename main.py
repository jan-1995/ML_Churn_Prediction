from churn_library import *
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import pandas as pd
import joblib
import shap
from logger import logging
from exception import CustomException

# Set the seaborn style
sns.set()

if __name__ == "__main__":

    """Perform all the function testings here"""

    response = "Churn"
    df = import_data("C:/Users/haide/Desktop/ML_Churn_Prediction/data/bank_data.csv")
    cat_cols, num_cols = get_categorical_and_numerical_columns(df)
    df = encoder_helper(df, cat_cols, response)
    perform_eda(
        df,
        output_dir="C:/Users/haide/Desktop/ML_Churn_Prediction/images/EDA",
    )
    X_train, X_test, y_train, y_test, X_data = perform_feature_engineering(df, response)
    (
        y_train_preds_rf,
        y_test_preds_rf,
        y_train_preds_lr,
        y_test_preds_lr,
        model,
    ) = train_models(X_train, X_test, y_train, y_test)
    feature_importance_plot(
        model,
        X_data,
        "C:/Users/haide/Desktop/ML_Churn_Prediction/images/results",
    )
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
        output_dir="C:/Users/haide/Desktop/ML_Churn_Prediction/images/",
    )
