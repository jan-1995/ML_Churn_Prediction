# Customer Churn Prediction with Machine Learning

This Python project focuses on predicting customer churn rate using machine learning. The code in this repository includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Data](#data)
- [Logging](#logging)
  - [logger.py](#loggerpy)
  - [exception.py](#exceptionpy)
- [Running The Files](#running-the-files)

- [Contact Information](#contact-information)


## Introduction

Customer churn, also known as customer attrition, occurs when customers stop doing business with a company. Predicting customer churn is crucial for businesses to retain valuable customers and take necessary actions to prevent them from leaving.

In this project, we build a machine learning model to predict customer churn based on customer data. We use a combination of logistic regression and random forest classifiers to make predictions, and we evaluate the models using classification reports and ROC curves.

## Project Structure

The project is organized as follows:

- `data/`: This directory contains the dataset used for training and testing.
- `images/`: Contains images generated during EDA and model evaluation.
- `models/`: Stores trained machine learning models.
- `src/`: This directory contains the source code files, including the main Python script.
- `README.md`: The main documentation file you are currently reading.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine:

   ```shell
   git clone https://github.com/jan-1995/customer-churn-prediction.git

2. Navigate to the project directory
   
    ```shell
   cd customer-churn-prediction

## Dependencies

To install the dependencies, run the following command in your terminal, you should have all the dependencies installed after the command has completed running.

   pip install -r requirements.txt

## Usage

To run the code and predict customer churn, you can execute the main script as follows:

    python main.py

## Data

A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction. This dataset is from a website with the URL as https://leaps.analyttica.com/home. 

1. Now, this dataset consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc. There are nearly 18 features.

2. We have only 16.07% of customers who have churned. Thus, it's a bit difficult to train our model to predict churning customers.

3. The dataset should include information about customers and whether they churned or not.

## Logging

This document explains the usage and purpose of two Python files, "logger.py" and "exception.py," that play a crucial role in logging and exception handling within a Python project.

### logger.py

"logger.py" is a Python module responsible for setting up logging functionality within the project. It performs the following tasks:

1. Import necessary libraries:
   - `import logging`: Import the built-in `logging` module for logging capabilities.
   - `import os`: Import the `os` module to work with file paths.
   - `from datetime import datetime`: Import `datetime` for timestamping log files.

2. Define the log file path:
   - The `LOG_FILE` variable generates a unique log file name using the current timestamp.
   - The `logs_path` variable creates a directory for log files using the `os.path.join` and `os.makedirs` functions.

3. Set up the logger configuration:
   - `logging.basicConfig()`: This function configures the logger with the specified log file path, log message format, and log level (INFO in this case).
   - The log messages will include the timestamp, line number, logger name, log level, and the message itself.

4. Example usage:
   - The module demonstrates its functionality by logging an "Logging has started" message when executed.

### exception.py

"exception.py" is a Python module for custom exception handling. It provides tools for capturing and handling exceptions, including creating informative error messages. Key features are as follows:

1. Import necessary libraries:
   - `import sys`: Import the `sys` module for exception details.
   - `from logger import logging`: Import the logging module from "logger.py" to log error messages.

2. Create an error message with details:
   - The `error_message_detail()` function captures exception details such as the file name, line number, and error message.
   - It creates an error message that includes this information.

3. Define a custom exception class:
   - The `CustomException` class inherits from the base `Exception` class.
   - It initializes with an error message and exception details.
   - The `__str__` method returns the error message.

4. Example usage:
   - The module demonstrates its functionality by raising a custom exception when division by zero occurs. The exception message is then logged.
  
## Running The Files

This Python script is the main entry point for running a Churn Prediction project. It leverages various functions and modules from a custom library, "churn_library," to perform tasks related to data processing, model training, and evaluation.

   python main.py

Here's what this script does:

### Import Necessary Libraries:
The script begins by importing essential libraries required for data processing, machine learning, and visualization. These include popular libraries like Pandas for data handling, Seaborn for data visualization, and scikit-learn for machine learning tasks.

### Initialize Logging and Exception Handling:
It utilizes a custom logging and exception handling system that logs important messages and handles exceptions gracefully. This is done through the "logger.py" and "exception.py" modules.

### Data Import:
The script reads a dataset from a specified file path using the "import_data" function. The user should replace the file path with their own data source.

### Data Preprocessing:
The dataset is preprocessed by identifying categorical and numerical columns, encoding categorical features, and conducting exploratory data analysis (EDA). EDA visualizations are saved in a user-specified directory. Users can customize the output directory to their preferred location.

### Feature Engineering:
Feature engineering is performed to prepare the data for model training. It selects specific columns and splits the dataset into training and testing sets. Users can modify the selected columns based on their project requirements.

### Model Training:
The script trains machine learning models, specifically a Random Forest Classifier and a Logistic Regression model. It utilizes GridSearchCV for hyperparameter tuning to find the best model configuration. Model training results and metrics are saved in a user-specified directory.

### Feature Importance Plot:
The importance of features used in the machine learning models is visualized and saved as an image. Users can specify the output directory for this image.

### Classification Report:
A classification report is generated for model evaluation. This report includes key metrics such as precision, recall, and F1-score for both the training and testing data. The report is saved as an image in a user-defined directory.

- To use this script for your own project, you should make the following customizations:

- Set the <mark>response</mark> variable to the target variable you want to predict (e.g., "Churn").
Update the <mark>data file path</mark> in the "import_data" function to point to your dataset.
Adjust the output directories to your preferred location, ensuring that they exist in your file system.
After these customizations, you can run the script to perform churn prediction tasks on your dataset. It's a versatile tool that can be adapted to various churn prediction projects by changing the dataset, target variable, and output paths.


## Contact Information 📬

If you're interested in collaboration, have specific queries, or would like to get in touch, feel free to reach out via email:

**Email:** janhaider040@gmail.com 📧

We look forward to hearing from you and working together to further improve this project or address any questions you may have! 🤝








   

