# Banking Dataset Analysis and Classification

## Overview

This repository contains code and analysis for predicting whether a bank client will subscribe to a term deposit. The primary goal is to develop a classification model that can handle imbalanced data, perform feature engineering, and produce meaningful predictions with an emphasis on the F1-score.

### Key Objectives

- **Data Cleaning & Preprocessing:**  
  - Handle missing values
  - Drop irrelevant features
  - Encode categorical variables using `OneHotEncoder`

- **Dealing with Imbalance:**  
  - Apply **Random Undersampling** to address the skewed distribution of the target variable

- **Model Training & Evaluation:**  
  - Train multiple models including:
    - Logistic Regression
    - Support Vector Machine (SVM)
    - Histogram-Based Gradient Boosting Classifier (HGBC)
    - Random Forest
    - Bagging Classifier
    - XGBoost
  - Evaluate models using precision, recall, accuracy, and **F1-score**, with a focus on improving F1-score due to class imbalance

- **Submission Preparation:**  
  - Predict on the test set and generate a submission file in the required format

## Dataset Description

### Files
- `Train-Set.csv` - Training data with features and the `Target` variable
- `Test-Set.csv` - Test data with features only, used for final predictions
- `sample_submission.csv` - A sample submission file showing the required format

### Key Features
- **Age:** Numeric age of the customer
- **Job:** Occupation type (e.g., management, technician, student)
- **Marital:** Marital status (e.g., married, single, divorced)
- **Education:** Educational background (e.g., primary, secondary, tertiary)
- **Default:** Indicates if the customer has previously defaulted (yes/no)
- **Housing:** Indicates if the customer has a housing loan (yes/no)
- **Loan:** Indicates if the customer has any other loan (yes/no)
- **Contact:** Preferred contact method (e.g., cellular, telephone)
- **Month & Day:** Indicates when the customer was last contacted
- **Duration, Campaign, Pdays, Previous:** Details about the current and past campaigns
- **Poutcome:** Outcome of the previous marketing campaign (success/failure)
- **Target:** The main variable indicating if the customer subscribed to the term deposit (yes/no)

## Steps to Reproduce

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/banking-dataset-analysis.git
   cd banking-dataset-analysis
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Typical `requirements.txt` might include:
   - pandas
   - numpy
   - scikit-learn
   - imblearn
   - xgboost
   - matplotlib
   - seaborn

3. **Add Data Files:** 
   Place `Train-Set.csv`, `Test-Set.csv`, and `sample_submission.csv` in the project directory.

4. **Run the Analysis:**
   - If using Jupyter: Open `banking_analysis.ipynb` and run the cells
   - Otherwise: 
     ```bash
     python main.py
     ```
   This will:
   - Load and preprocess the data
   - Perform undersampling to handle imbalanced classes
   - Train multiple models and print out their metrics
   - Generate a submission file

## Model Performance & Results

Each model's precision, recall, accuracy, and F1-score are reported. The F1-score is crucial in imbalanced scenarios as it balances both precision and recall.

### Models Evaluated
- Logistic Regression
- SVM
- Histogram Gradient Boosting Classifier (HGBC)
- Random Forest
- Bagging Classifier (Decision Tree base)
- XGBoost

Performance can be improved through:
- Hyperparameter tuning
- Additional feature engineering
- Advanced imbalanced data techniques (e.g., SMOTE)

## Exploratory Data Analysis (EDA)

The notebook includes various plots to visualize:
- Distribution of categorical variables
- Relationship between features and target variable

These visualizations help identify patterns and inform feature selection strategies.

## Contributing

Contributions welcome:
- Suggest new imbalance handling techniques
- Implement hyperparameter tuning strategies
- Add new visualizations or insights

To contribute, open an issue or submit a pull request.

## License

MIT License

## Contact

For questions or suggestions, please reach out to your-email@example.com or open a GitHub issue.
