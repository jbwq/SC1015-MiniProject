# SC1015-MiniProject

#### Built by: Boey Wen Qi, Lee Yu Quan, Lim Zhi Li Kyle 


## Introduction 

This mini-project is part of module SC1015 (Introduction to Data Science & Artificial Intelligence). In  this project, the team explored a dataset taken from [Heart Failure Prediction Dataset](https://www.kaggle.com/fedesoriano/heart-failure-prediction). This dataset composes of data retrieved from countries such as the United States of America, Switzerland and Hungary, comprising of risk factors that contributes to the risk of developing Cardiovascular Dieases or CVDs for short. This amounted to a total of 918 observations, inclusive continuous, categorical and binary data types that allowed the team to explore and apply various data handling and visualisation methods seen in the project's code. The large sample size is a key factor in improving the reliability and the credibility of the results obtained at the end of this project and the project team hopes that this would enable the usability for health institutions across different countries.

### So...What is Cardiovascular Disease?

Cardiovascular diseases refer to a number of health condition that affect the circulatory system (such as the heart and arteries) (Kohli, M.D., FACC & Felman, 2019).

Cardiovascular diseases (CVDs) are the leading cause of death worldwide, killing an estimated **17.9 million** people each year, accounting for 32% of all fatalities worldwide (World Health Organization (WHO), 2021). Out of these deaths, **Heart attacks and strokes** make up **85%** of them. Notably, 38% of premature deaths (under the age of 70) in 2019 were caused by CVDs. 

### What role does Machine Learning play in the prediction of CVDs?

**Early detection and management of CVDs** are **crucial** for individuals with CVDs or those at high cardiovascular risk due to one or more risk factors, such as hypertension, diabetes, hyperlipidaemia, or pre-existing conditions. In such cases, **machine learning models** can be extremely beneficial. 

In the context of Singapore, the rapidly ageing population would see an increase in expenditure of the government towards resources and subsidies to reduce the burden of healthcare costs on families - allowing accessibility towards medical services for the elderly (Chin , 2022). This is on top of the worrying statistic that 21 people die from cardiovascular disease daily, accounting for 32% of all deaths in 2021 (Singapore Heart Foundation, 2021). This brings forth the importance of machine learning, enabling early detection and management would hopefully see a reduction of medical costs in the long term. 

### What this project entails 

This project entailed utilising knowledge attained from the module's materials in processes such as data extraction and visualisation, leading to exploratotry data analysis (EDA), following up with data cleaning and resampling. The team also explored beyond covered material during the model building process which led to promising results.

CVDs are a common cause of heart failure, and this dataset contained 11 variables that can be used to predict heart disease.

## Problem Defintion 
Which brings us to our project defintion and goals. 
- Create a model for healthcare systems of different countries that allows accurate prediction of an individual having CVDs, allowing for early interventions.
- Our goal is to create a model that has 95% accuracy to predict the chances of developing CVDs.

## Table of Contents

This project's code is sequenced as follows (Please review the code in the order listed for a better understanding):

1. Data Extraction and Visualisation
2. Exploratory Data Analysis
3. Data Cleaning 
4. Data Resampling and Train and Test Set Splitting 
5. Data Modelling (Use of Classifiers) 

## Libraries Used
- Data Visualisation: [seaborn](https://seaborn.pydata.org/), [matplotlib (pyplot)](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)
- Data Processing: [Numpy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
  - Pre-processing: sklearn [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html), [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
- Data Cleaning: sklearn [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- Data Modelling: sklearn
  - Metrics: confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, roc_auc_score, mean_squared_error, ConfusionMatrixDisplay, classification_report (https://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation)
  - Model selection: [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html), [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV), [GridSearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV), [StratifiedShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html), [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html), [cross_val_predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html)
  - Supervised Learning methods: [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html), [Random Forest Classifer](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier), [C-Support Vector Classifcation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC), [KNeighborsClassifer](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)

## Models Used
The team utilised 3 supervised learning models to experiment with classification methods. They are as follows:
- K-Nearest Neighbours 
- C-Support Vector 
- Random Forest 

## Conclusion


## Key Takeaways 
- Explored various data visualisation. 
- Explored different methods during Data Cleaning (for cholesterol values: removing 0s, replacing with them median values and replacing with a *multivariate polynomial regression **predicted*** value. 
- Carrying out Data Resampling after *Data Cleaning* and identifying which *risk factors* are the better predictors for **HeartDiease**.
- Gaining a deeper understanding of metrics used to evaluate the accuracy of models such as F1 score, Precision, Accuracy and Recall and when to apply them according to our needs.
- Explored classification methods as alternatives to building models. By understanding their respective applications and their advantages and disadvantages.
- Technical and platform usage: Collaborating on Google Collab and Github.

## Contributors 
- @jbwq - Data Extraction, Data Visualisation, Data Classification (model building)
- @kylerlim - Data Visualisation, Data Cleaning, Data Classification (model building) 
- @yuquan1ee - Data Extraction, Data Visualisation, Data Resampling, Data Classification (model building) 

## References 
- https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)
- https://www.medicalnewstoday.com/articles/257484#statistics
- *heavy reference to this project done on kagle* https://www.kaggle.com/code/aletbm/cardiovascular-diseases-eda-modeling#%F0%9F%93%8A-EDA-and-data-wrangling
- https://www.ibm.com/topics/random-forest
- https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
- https://www.ibm.com/topics/knn
- https://www.myheart.org.sg/health/heart-disease-statistics/ 
- https://www.straitstimes.com/singapore/singapores-population-ageing-rapidly-184-of-citizens-are-65-years-and-older
