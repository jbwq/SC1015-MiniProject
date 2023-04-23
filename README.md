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

### What does this project entail?

This project entailed utilising knowledge attained from the module's materials in processes such as data extraction and visualisation, leading to exploratotry data analysis (EDA), following up with data cleaning and resampling. The team also explored beyond covered material during the model building process which led to promising results.

CVDs are a common cause of heart failure, and this dataset contained 11 variables that can be used to predict heart disease.

## Problem Defintion 
Which brings us to our project defintion and goals. 
- Create a model for healthcare systems of different countries that allows accurate prediction of an individual having CVDs, allowing for early interventions.
- Our goal is to create a model that has 95% accuracy to predict the chances of developing CVDs.

## Table of Contents

This project's code is sequenced as follows (Please review the code in the order listed for a better understanding):

1. Data Visualisation and Exploratory Data Analysis (EDA) - 1)_Data_Visualiation_and_EDA.ipynb
2. Data Cleaning - 2)_Data_Cleaning.ipynb
3. Data Resampling and further EDA - 3)_Data_resampling.ipynb
4. Data Modelling - 4)_Modelling.ipynb

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
- In conclusion, the model that yielded the highest accuracy (in terms of both the recall and F1-score metrics) is utilising the k-Nearest Neighbours classification method. 

## Key Takeaways 
- Explored various data visualisation. 
- Explored different methods during Data Cleaning (for cholesterol values: removing 0s, replacing with them median values and replacing with a *multivariate polynomial regression **predicted*** value. 
- Carrying out Data Resampling after *Data Cleaning* and identifying which *risk factors* are the better predictors for **HeartDiease**.
- Gaining a deeper understanding of metrics used to evaluate the accuracy of models such as F1 score, Precision, Accuracy and Recall and when to apply them according to our needs.
- Explored classification methods as alternatives to build accurate and reliable models. By understanding their respective applications and their advantages and disadvantages, this allowed us to see which type of algorithms are applicable and suited for different datasets. 
- Technical and platform usage: Collaborating on Google Collab and Github.

## Areas explored beyond syllabus materials:
- Multivariate Polynomial Regression
- Cross Validation to increase the accuracy of the model 
- Random Forest Classifer hyperparameter tuning
- Supervised learning methods: C-Support Vector, k-Nearest Neighbours

## Contributors 
- @jbwq - Data Extraction, Data Visualisation, Data Classification (model building)
- @kylerlim - Data Visualisation, Data Cleaning, Data Classification (model building) 
- @yuquan1ee - Data Extraction, Data Visualisation, Data Resampling, Data Classification (model building) 

## Articles referenced during this project that we think are good reference materials
- Article on cross validation and how to implement it: https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right
- Article on model metrics: https://koopingshung.com/blog/machine-learning-model-selection-accuracy-precision-recall-f1/
- Articles on k-Nearest Neighbours: 
  - Understanding k-Nearest Neighbours: https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning
- Articles on Random Forest Classifier: 
  - Understanding Random Forest Classifer: https://towardsdatascience.com/understanding-random-forest-58381e0602d2
  - Implementing Random Forest Classifier: https://towardsdatascience.com/machine-learning-basics-random-forest-classification-499279bac51e

## References 
International Business Machines Corporation (IBM). (n.d.). What is Random Forest? IBM. Retrieved April 22, 2023, from https://www.ibm.com/topics/random-forest 

International Business Machines Corporation (IBM). (n.d.). What is the K-nearest neighbors algorithm? IBM. Retrieved April 22, 2023, from https://www.ibm.com/topics/knn 

alaa2mahmoud. (2023, February 19). Heart disease: 6 classifiers. Kaggle. Retrieved April 22, 2023, from https://www.kaggle.com/code/alaa2mahmoud/heart-disease-6-classifiers#Heart-Failure-Classification-Project 

Aletbm. (2023, March 16). Cardiovascular diseases - EDA + modeling 🏥. Kaggle. Retrieved April 22, 2023, from https://www.kaggle.com/code/aletbm/cardiovascular-diseases-eda-modeling#%F0%9F%93%8A-EDA-and-data-wrangling 

Amazon Web Services. (n.d.). Docs.aws.amazon.com. Amazon Web Services. Retrieved April 22, 2023, from https://docs.aws.amazon.com/pdfs/machine-learning/latest/dg/machinelearning-dg.pdf 

Chin , S. F. (2022, September 27). S'pore's population ageing rapidly: Nearly 1 in 5 citizens is 65 years and older. The Straits Times. Retrieved April 21, 2023, from https://www.straitstimes.com/singapore/singapores-population-ageing-rapidly-184-of-citizens-are-65-years-and-older 

Fedesoriano. (2021, September). Heart failure prediction dataset. Kaggle. Retrieved April 23, 2023, from https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction 

Gandhi, R. (2018, June 8). Support Vector Machine - introduction to machine learning algorithms. Medium. Retrieved April 22, 2023, from https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47 

javatpoint. (2021). K-Nearest Neighbor(KNN) algorithm for Machine Learning - Javatpoint. www.javatpoint.com. Retrieved April 22, 2023, from https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning 

K, G. M. (2020, September 23). Machine learning basics: Random forest classification. Medium. Retrieved April 22, 2023, from https://towardsdatascience.com/machine-learning-basics-random-forest-classification-499279bac51e 

Kohli, M.D., FACC, D. P., &amp; Felman, A. (2019, July 26). Cardiovascular disease: Types, symptoms, prevention, and causes. Medical News Today. Retrieved April 20, 2023, from https://www.medicalnewstoday.com/articles/257484 

Lyashenko, V. (2023, April 21). Cross-validation in Machine Learning: How To Do It Right. neptune.ai. Retrieved April 22, 2023, from https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right 

Navlani, A. (2019, December). Scikit-learn SVM tutorial with Python (Support Vector Machines). DataCamp. Retrieved April 22, 2023, from https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python 

Saw Swee Hock School of Public Health. (2018, December 18). Cost of silent risk factors for cardiovascular disease in Asia. SPH.NUS.EDU.SG. Retrieved April 22, 2023, from https://sph.nus.edu.sg/2018/12/cost-of-silent-risk-factors-for-cardiovascular-disease-in-asia/#:~:text=CVDs%20contribute%20to%20approximately%20one,households%20and%20the%20public%20finances 

Shung, K. P. (2020, January 20). Model selection: Accuracy, precision, recall or F1? Building Intelligence Together. Retrieved April 22, 2023, from https://koopingshung.com/blog/machine-learning-model-selection-accuracy-precision-recall-f1/ 

Singapore Heart Foundation. (2021). Heart disease statistics. Singapore Heart Foundation. Retrieved April 21, 2023, from https://www.myheart.org.sg/health/heart-disease-statistics/ 

World Health Organization (WHO). (2021, June 11). Cardiovascular diseases (CVDs). World Health Organization. Retrieved April 20, 2023, from https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds) 

Yiu, T. (2019, June 12). Understanding random forest. Medium. Retrieved April 22, 2023, from https://towardsdatascience.com/understanding-random-forest-58381e0602d2 
