"""
Function Testing file, running this file with pytest will 
test all the necessary functions that are used in the churn_library.py

Author: Haider Jan
Creation Date: 30/10/2023
"""
import sys
import os
import pytest
from churn_library import *
import random
from logger import logging
from exception import CustomException
import pandas as pd

@pytest.fixture(scope="module")
def data_path():
    """
    Fixture which provides the data path to the test_import function
    """
    return "C:/Users/haide/Desktop/ML_Churn_Prediction/data/bank_data.csv"

def test_import(data_path):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = import_data(data_path)
        logging.info("Testing import_data: SUCCESS")
    except 	Exception as e: 
        raise CustomException(e,sys)

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except Exception as e:
         raise CustomException(e,sys)

@pytest.fixture(scope="module")
def data():
        
	df = pd.DataFrame(
		{
			"Churn": [0, 1, 0, 1, 0],
			"Customer_Age": [25, 30, 35, 40, 45],
			"Marital_Status": ["Single", "Married", "Single", "Married", "Single"],
			}
		)
	return df

def test_perform_eda(tmpdir,data):
     
	"""
     Function which tests for the Perform EDA function
     Args: Tmpdir is created instead of testing out the real folder
    """

	try:     
		logging.info("test_perform_eda: creating example df for test_eda and then asserting")



		output_directory = str(tmpdir.mkdir("test_output"))
		perform_eda(data, output_directory)

	# Check if the output files exist
		assert os.path.exists(os.path.join(output_directory, "Churn_distribution.png"))
		assert os.path.exists(
			os.path.join(output_directory, "Customer_Age_distribution.png")
		)
		assert os.path.exists(
			os.path.join(output_directory, "marital_status_distribution.png")
		)
		assert os.path.exists(os.path.join(output_directory, "correlation_heatmap.png"))
	
	except Exception as e:
		raise CustomException(e,sys)


@pytest.fixture(scope="module")
def response():
    """ Returns response variable """
    return "Churn"

@pytest.fixture(scope="module")
def encode_feats():
    """ fixture feeds the label encoder all the necessary features"""
    d2encode = pd.DataFrame(
        {
            "Customer_Age": [25, 30, 35, 40, 45],
            "Marital_Status": ["Single", "Married", "Single", "Married", "Single"],
            "Attrition_Flag": [
                "Existing Customer",
                "Non Existing Customer",
                "Existing Customer",
                "Non",
                "Non",
            ],
        }
    )
    return d2encode
     

def test_encoder_helper(response,encode_feats):
    """ Encoder helpwe test 
        Args: response fixture, encode_feats fixture
    """
        
    try:
        logging.info("Starting to Encode Features")
        cat_cols = ["Customer_Age", "Marital_Status", "Attrition_Flag"]
        result = encoder_helper(encode_feats, cat_cols, response)

        assert f"Customer_Age_{response}" in result.columns
        assert f"Marital_Status_{response}" in result.columns
        assert f"Attrition_Flag_{response}" in result.columns
        logging.info("Features encoded successfully using encoder helper")

    except Exception as e:
            raise CustomException(e,sys)

@pytest.fixture(scope="module")
def response():
    """
    Fixture that returns a response string.
    """
    return "Churn"

@pytest.fixture(scope="module")
def Cols_to_Keep():
    """
    Fixture that defines a list of columns to keep.
    """
    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
        "Churn",
    ]
    return keep_cols

def test_perform_feature_engineering(response):
    """
    Test the perform_feature_engineering function.
    """
    # Create an example DataFrame with random data
    data = {
        col: [
            random.randint(20, 70) if col == "Customer_Age" else random.randint(0, 5)
            for _ in range(10)
        ]
        for col in Cols_to_Keep
    }

    df = pd.DataFrame(data)
    
    result = perform_feature_engineering(df, response)

    try:
        logging.info("Test Started for test perform feature engineering")
        assert len(result) == 5
    except Exception as e:
        raise CustomException(e, sys)

@pytest.fixture(scope="module")
def sample_data():
    """
    Fixture that defines sample data for testing.
    """
    # Defining a random data sample to be used for testing
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    X_test = np.array([[7, 8], [9, 10]])
    y_train = np.array([0, 1, 0])
    y_test = np.array([1, 1])
    return X_train, X_test, y_train, y_test

def test_train_models(sample_data):
    """
    Test the train_models function with sample data.
    """
    X_train, X_test, y_train, y_test = sample_data

    # Test the train_models function
    (
        y_train_preds_rf,
        y_test_preds_rf,
        y_train_preds_lr,
        y_test_preds_lr,
        model,
    ) = train_models(X_train, X_test, y_train, y_test)

    # Assertions
    assert len(y_train_preds_rf) == len(y_train)
    assert len(y_test_preds_rf) == len(y_test)
    assert len(y_train_preds_lr) == len(y_train)
    assert len(y_test_preds_lr) == len(y_test)
    
if __name__=="__main__":
     pytest.main()