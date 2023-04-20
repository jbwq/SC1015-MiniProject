# SC1015-MiniProject

## About 

This mini-project is part of module SC1015 (Introduction to Data Science & Artificial Intelligence). In  this project, the team explored a dataset taken from Heart Failure Prediction Dataset (https://www.kaggle.com/fedesoriano/heart-failure-prediction). This dataset composes of data retrieved from countries such as the United States of America, Switzerland and Hungary, comprising of risk factors that contributes to the risk of developing Cardiovascular Dieases or CVDs for short. This amounted to a total of 918 observations, inclusive continuous, categorical and binary data types that allowed the team to explore and apply various data handling and visualisation methods seen in the project's code. The large sample size is a key factor in improving the reliability and the credibility of the results obtained at the end of this project and the project team hopes that this would enable the usability for health institutions across different countries.

### So...What is Cardiovascular Disease?

Cardiovascular diseases refer to a number of health condition that affect the circulatory system (such as the heart and arteries) (Kohli, M.D., FACC & Felman, 2019).

Cardiovascular diseases (CVDs) are the leading cause of death worldwide, killing an estimated **17.9 million** people each year, accounting for 32% of all fatalities worldwide (World Health Organization (WHO), 2021). Out of these deaths, **Heart attacks and strokes** make up **85%** of them. Notably, 38% of premature deaths (under the age of 70) in 2019 were caused by CVDs. 

**Early detection and management of CVDs** are **crucial** for individuals with CVDs or those at high cardiovascular risk due to one or more risk factors, such as hypertension, diabetes, hyperlipidaemia, or pre-existing conditions. In such cases, **machine learning models** can be extremely beneficial. 

Thus, the project team embarked on this project, that entailed utilising knowledge attained from the module's materials in processes such as data extraction and visualisation, leading to exploratotry data analysis (EDA), following up with data cleaning and resampling. The team also explored beyond covered material during the model building process which led to promising results.

CVDs are a common cause of heart failure, and this dataset contained 11 variables that can be used to predict heart disease.

## Table of Contents

This project's code is sequenced as follows (Please review the code in the order listed for a better understanding):

1. Data Extraction and Visualisation
2. Exploratory Data Analysis
3. Data Cleaning 
4. Data Resampling and Train and Test Set Splitting 
5. Data Modelling (Use of Classifiers) 

## Contributors 
- @jbwq - Data Extraction, Data Visualisation, Data Classification (model building)
- @/kyle - Data Visualisation, Data Cleaning, Data Classification (model building) 
- @/yuquan - Data Extraction, Data Visualisation, Data Resampling, Data Classification (model building) 

## Problem Defintion 
- Which responses contribute to a higher chance of developing CVD?
- Which model predicts this more accurately?

## Libraries Used
- Data Visualisation: [seaborn](https://seaborn.pydata.org/), [matplotlib (pyplot)](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)
- Data Processing: [Numpy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
  - Pre-processing: sklearn [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html), [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
- Data Cleaning: sklearn [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- Data Modelling: sklearn
  - Metrics: confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, roc_auc_score, mean_squared_error, ConfusionMatrixDisplay, classification_report (https://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation)
  - Model selection: [train_test_split](), [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV), [GridSearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV), [StratifiedShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html), [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html), [cross_val_predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html)
  - Supervised Learning methods: [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html), [Random Forest Classifer](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier), [C-Support Vector Classifcation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC), [KNeighborsClassifer](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)

## Models Used
The team utilised 3 supervised learning models to experiment with classification methods. They are as follows:
- K-Nearest Neighbours 
- C-Support Vector 
- Random Forest 

## Conclusion

## Key Takeaways 

## References 
