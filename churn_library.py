"""
This python file is a library of all functions that will be used
to produce a machine learning model for Predicting Customer Churn Rate
"""
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

def import_data(pth):
    """
    Returns a dataframe for the CSV found at pth.

    Args:
        pth (str): A path to the CSV.

    Returns:
        df: pandas DataFrame.
    """
    try:
        logging.info("Reading Data From Provided Source")
        data_frame = pd.read_csv(pth)
        logging.info("Data Read from Source")
        return data_frame
    except Exception as e:
        raise CustomException(e, sys)

def perform_eda(df, output_dir):
    """
    Perform EDA on df and save figures to the images folder.

    Args:
        df: pandas DataFrame.
        output_dir (str): Path to the output directory.

    Returns:
        None
    """
    feat_list = ["Churn", "Customer_Age"]

    try:
        os.makedirs(output_dir,exist_ok="True")
        logging.info("Perform EDA has started")
        for feature in feat_list:
            plt.figure(figsize=(20, 10))
            df[feature].hist()
            plt.title(f"{feature} Distribution")
            plt.savefig(os.path.join(output_dir, f"{feature}_distribution.png"))

        # Plot Marital Status distribution and save the figure
        plt.figure(figsize=(20, 10))
        df["Marital_Status"].value_counts(normalize=True).plot(kind="bar")
        plt.title("Marital Status Distribution")
        plt.savefig(os.path.join(output_dir, "marital_status_distribution.png"))
        plt.close()

        # Plot correlation heatmap and save the figure
        plt.figure(figsize=(20, 10))
        sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()
        logging.info(f"EDA has been performed and all data has been saved in the folder  {output_dir} ")
    except Exception as e:
        raise CustomException(e, sys)

def get_categorical_and_numerical_columns(df):
    """
    Get separate lists of categorical and numerical columns in the DataFrame.

    Args:
        df: pandas DataFrame.

    Returns:
        Tuple of categorical and numerical column lists.
    """
    try:
        logging.info("Getting all the categorical data ")
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        quant_cols = df.select_dtypes(include=["number"]).columns.tolist()
        return cat_cols, quant_cols
        logging.info("Got all the categorical data")
    except Exception as e:
        logging.info("An ERROR has occured")
        raise CustomException(e, sys)

def encoder_helper(df, category_lst, response):
    """
    Helper function to encode categorical columns based on churn rate.

    Args:
        df: pandas DataFrame.
        category_lst: List of columns that contain categorical features.
        response: String of response name [optional argument used for naming y column].

    Returns:
        df: pandas DataFrame with new columns for encoding.
    """
    try:
        logging.info("Encoding all the Categorical Columns and changing their names to _Churn ")
        df[response] = df["Attrition_Flag"].apply(lambda val: 0 if val == "Existing Customer" else 1)
        for category in category_lst:
            encoded_column_name = f"{category}_{response}"
            category_group = df.groupby(category).mean()[response]
            df[encoded_column_name] = df[category].map(category_group)
        logging.info("encoder helper has completed task successfully")
        return df
    except Exception as e:
        logging.info("An ERROR has been raised, check it out in log runs")
        raise CustomException(e, sys)

def perform_feature_engineering(df, response):

    """
    Perform feature engineering and split data into training and testing sets.

    Args:
        df: pandas DataFrame.
        response: String of response name [used for naming variables or index y column].

    Returns:
        X_train: X training data.
        X_test: X testing data.
        y_train: y training data.
        y_test: y testing data.
        X_data: Original X data.
    """
    try:
        logging.info("Performing feature engineering and data splits to produce for model training")
        y = df[response]
        keep_clms = df[keep_columns()]
        X = keep_clms
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        logging.info("Got the data splits and feature engineering has been completed")
    except Exception as e:
        logging.info("An ERROR has been raised")
        raise CustomException(e,sys)
    return X_train, X_test, y_train, y_test, X


def keep_columns():
    try:
        keep_cols = [
            "Customer_Age", "Dependent_count", "Months_on_book",
            "Total_Relationship_Count", "Months_Inactive_12_mon",
            "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
            "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt",
            "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio",
            "Gender_Churn", "Education_Level_Churn", "Marital_Status_Churn",
            "Income_Category_Churn", "Card_Category_Churn",
        ]
        return keep_cols
    except Exception as e:
        raise CustomException(e, sys)

def classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf, output_dir):
    """
    Produce a classification report for training and testing results and store the report as an image.

    Args:
        y_train: Training response values.
        y_test: Test response values.
        y_train_preds_lr: Training predictions from logistic regression.
        y_train_preds_rf: Training predictions from random forest.
        y_test_preds_lr: Test predictions from logistic regression.
        y_test_preds_rf: Test predictions from random forest.
        output_dir: Path to store the report image.

    Returns:
        None
    """
    try:
        logging.info("Getting all the plots and classification reports related to model training")
        plt.figure(figsize=(10, 5))
        plt.ioff()  # Turn off interactive mode

        # Create the first subplot for Random Forest results
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
        plt.text(0.01, 1.25, str("Random Forest Train"), {"fontsize": 10}, fontproperties="monospace")
        plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {"fontsize": 10}, fontproperties="monospace")
        plt.text(0.01, 0.6, str("Random Forest Test"), {"fontsize": 10}, fontproperties="monospace")
        plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {"fontsize": 10}, fontproperties="monospace")
        plt.axis("off")

        # Create the second subplot for Logistic Regression results
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
        plt.text(0.01, 1.25, str("Logistic Regression Train"), {"fontsize": 10}, fontproperties="monospace")
        plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {"fontsize": 10}, fontproperties="monospace")
        plt.text(0.01, 0.6, str("Logistic Regression Test"), {"fontsize": 10}, fontproperties="monospace")
        plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {"fontsize": 10}, fontproperties="monospace")
        plt.axis("off")

        # Save the figure to a file
        plt.savefig(os.path.join(output_dir, "Classification_Report.png"), bbox_inches="tight")
        # Close the figure to release resources
        plt.close()
        logging.info(f"All reports related to model training are received and saved in the folder: {output_dir}")
    except Exception as e:
        logging.info("An ERROR has been raised check it out in the terminal")
        raise CustomException(e,sys)

def feature_importance_plot(model, X_data, output_pth):
    """
    Create and store the feature importances in output_pth.

    Args:
        model: Model object containing feature_importances_.
        X_data: Pandas DataFrame of X values.
        output_pth: Path to store the figure.

    Returns:
        None
    """
    try:
        logging.info(f"Saving all the feature Importances plots in the desired output path: {output_pth}")    
        # Calculate feature importances
        importances = model.best_estimator_.feature_importances_

        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [X_data.columns[i] for i in indices]

        # Create the plot
        plt.figure(figsize=(20, 5))

        # Create the plot title
        plt.title("Feature Importance")
        plt.ylabel("Importance")

        # Add bars
        plt.bar(range(X_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(X_data.shape[1]), names, rotation=90)
        plt.savefig(os.path.join(output_pth, "feat_imp.png"))

    except Exception as e:
        logging.info("An error has occured, please check it in the terminal right now")
        raise CustomException(e,sys)

def train_models(X_train, X_test, y_train, y_test):
    """
    Train, store model results: images + scores, and store models.

    Args:
        X_train: X training data.
        X_test: X testing data.
        y_train: y training data.
        y_test: y testing data.

    Returns:
        None
    """
    try:
        logging.info("Training, storing model results and models")
        output_dir = "C:/Users/haide/Desktop/ML_Churn_Prediction/images/results"
        # Grid search
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

        param_grid = {
            "n_estimators": [200, 500],
            "max_features": ["auto", "sqrt"],
            "max_depth": [4, 5, 100],
            "criterion": ["gini", "entropy"],
        }
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=2)
        cv_rfc.fit(X_train, y_train)
        model = cv_rfc

        lrc.fit(X_train, y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

        y_train_preds_lr = lrc.predict(X_train)
        y_test_preds_lr = lrc.predict(X_test)

        # Store the
        directory = "C:/Users/haide/Desktop/ML_Churn_Prediction/images/results/"
        file_name = "model_scores.txt"
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, file_name)

        with open(file_path, "w") as file:
            # Write classification reports
            file.write("\nRandom Forest Results:\n")

            file.write("Test Results:\n")
            file.write(classification_report(y_test, y_test_preds_rf))

            file.write("\nTrain Results:\n")
            file.write(classification_report(y_train, y_train_preds_rf))

            file.write("\nLogistic Regression Results:\n")

            file.write("Test Results:\n")
            file.write(classification_report(y_test, y_test_preds_lr))

            file.write("\nTrain Results:\n")
            file.write(classification_report(y_train, y_train_preds_lr))

        lrc_plot = plot_roc_curve(lrc, X_test, y_test)
        plt.savefig(os.path.join(output_dir, "lrc_roc_curve.png"))

        # Plots
        plt.figure(figsize=(15, 8))
        ax = plt.gca()

        rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)

        lrc_plot.plot(ax=ax, alpha=0.8)
        plt.savefig(os.path.join(output_dir, "rfc_roc_curve.png"))

        # Save RFC and LRC best models
        joblib.dump(cv_rfc.best_estimator_, "C:/Users/haide/Desktop/ML_Churn_Prediction/models/rfc_model.pkl")
        joblib.dump(lrc, "C:/Users/haide/Desktop/ML_Churn_Prediction/models/logistic_model.pkl")

        explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        plt.savefig(os.path.join(output_dir, "tree_explainer.png"), dpi=700)  # .png,.pdf will also support here
    except Exception as e:
        raise CustomException(e,sys)

    return y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr, model
