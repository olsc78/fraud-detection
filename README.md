# E-Commerce Fraud Detection
Build and compare classification models to predict if an online transaction is fraudulent or not

## Dataset
- The synthetic dataset can be found here: data/Fraudulent_E-Commerce_Transaction_Data.csv and on Kaggle:  https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions
- It is an imbalanced, synthetic dataset of close to 1.5 million transactions, 5% of which are flagged as fraudulent
- The features are explanable, not anonymized and correspond to features one could find in real life.

## Notebook and Repository
- The data mining and the comparison between classifiers can be found in the "ecommerce-fraud-detection.ipynb" notebook. 
- Git Repository is located at https://github.com/olsc78/ecommerce-fraud-detection.git

## Summary of findings

### Data
The data exploration analysis revealed that the dataset was imbalanced but clean.
The main specificity of the dataset is that it is synthetic with a few simple fraud patterns hidden in the features.

### Models
Several models have been tested (Logistic regressions, K Nearest Neighbors, Decision Trees, Gradient Boosting Ensemble and Random Forest) with a grid search of various hyperparameters. SVC has been discarded as of now because of a lack of computing power.

### Findings
- Dataset is imbalanced and needed some strategies to circumvent: SMOTE, Random Under Sampler and class weights
- Metrics to focus on were PR AUC, ROC AUC and recall in order to find more true positives with the risk of increasing false positives
- Gradient boosting performed better, without being able to truly capture the true positives without generating a lot of false positives 
