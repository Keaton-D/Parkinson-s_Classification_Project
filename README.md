# Parkinson's Disease Classification Project

This project analyzes the acoustics biomarkers from the [UCI Parkinson's dataset](https://archive.ics.uci.edu/dataset/174/parkinsons) to classify the presence of Parkinson's using machine learning models in Python.

## Backgorund and Purpose

Parkinson's disease is a degenerative disease of the nervous system that affects motor functions [1]. Symptoms progress slowly and can go unnoticed or become mistaken as random coincidences. Common symptoms include tremors, slow movement, poor posture/balance, and speech changes [1].

There is currently no cure for Parkinson's disease and its cause is unknown. However, lifestyle changes, medications and surgical procedures can help manage symptoms and improve quality of life [1]. Because early identification can help faciliate timely clinical evaluations and treatment, developing methods to identify Parkinson's disease cases is an important area of reserach.

The purpose of this project is to develop a machine learning model capable of predicting the presence of Parkinson's disease using acoustic biomarkers from patient voice recordings. By identifying patterns associated with Parkinson's disease, this model may provide a potential tool for supporting earlier identification and further clinical evaluation.

## Data

The dataset used for this project comes from the [UCI Parkinson's dataset](https://archive.ics.uci.edu/dataset/174/parkinsons) donated in 2008.

The dataset contains 22 voice-measurement features collected from 31 people, 23 of whom have Parkinson's disease. Each person contributed approximately 6 recordings, amounting to 195 observations.

To keep the list short, here are some important features in this project:

| Features | Description |
| ------------ | ------------ |
| status | Target varible (1 for Parkinson's, 0 for not)
| MDVP:Fo(Hz) | Average vocal fundamental frequency |
| MDVP:Fhi(Hz) | Maximum vocal fundamental frequency |
| MDVP:Shimmer | Measurement of variation in amplitude |
| DFA | Signal fractal scaling exponent |
| spread1, spread2 | Nonlinear measures of fundamental frequency variation |
| D2 | Nonlinear dynamical complexity measures |

For a more detailed description of all dataset features, see `data\parkinsons.names`


## Exploratory Data Analysis

### Distribution of Status

![alt text](image.png)
As previously mentioned, the dataset contains 23 indivduals with Parkinson's disease and 8 healthy individuals. At the recoridng level, the dataset contains 147 Parkinson's observations and 48 healthy observations, resulting in a 3:1 class imbalance. This imbalnce can cause classification models to favor majority class, potentially reducing their ability to correctly identify healthy observations.

To remedy the imbalnce, SMOTE (Synthetic Minority Over-sampling Technique) was employed. SMOTE creates synthetic data points based on data from the minority class. This method is prefered over undersampling, which removes exisitng samples from the majority class. Because the dataset is small, retaining as many original observations as possible is preferable to removing existing ones.

### Feature Correlation

![alt text](image-1.png)
Strong positive correlations between **status** and features such as **PPE** and **spread1** suggest that they will be significant predictors in the final model. It is important to note that *PPE* and *spread1* show strong multicollinearity with each other (r = 0.96), indicating they share highly redundant information. Additionally, features showing moderate correlation like **spread2**, **MDVP:Shimmer**, **MDVP:APQ**, **MDVP:Shimmer(dB)**, **Shimmer:APQ5**, **Shimmer:APQ3**, **Shimmer:DDA**, **D2**, **MDVP:Jitter(Abs)**, and **RPDE** may also add predictive value.

The dataset contains several features that measure similar acoustic properties, such as the shimmer and jitter metrics. Because these features share high multicollinearity, including multiple variations of the same feature can introduce redundancy and instability.

To address this, *Lasso Regression* will be used as a feature selection method to retain the most informative representative(s) from each feature group. Lasso drives the coefficents of weak or redundant predictors to 0, effectively removing them and leaving behind a reduced, optimized set of model features [2].

## Data Preprocessing

### Grouping
Each patient has approximately 6 recordings. The **name** column assigns a patient number and recording iteration. The issue behind this is that the model can memorize patient-specific vocal traits rather than developing disease patterns, causing performance metrics to be artificailly inflated.

The fix is to assign groups based on the **name** column to ensure that all recordings belonging to a specific patient stay exclusively in either the training or testing set. Using *StratifiedGroupKFold* addresses this issue while preserving the overall class distribution [3].

### Feature Selection
As anticipated, *Lasso* excluded one of the two highly correlated features (**PPE**), keeping **spread1** instead. **spread2** was also retained, likely due to its moderate collinearity with **spread1** (r = 0.65) and its moderate correlation to **status** (r = 0.45), suggesting it captures aspects of frequency variation different from **spread1**.

Among the three fundamental frequency features, **MDVP:Fo(Hz)** and **MDVP:Fhi(Hz)** were selected as predictors, whereas **MDVP:Flo(Hz)** was excluded. 

The exclusion likely stems from the moderate collinearity between **MDVP:Fo(Hz)** and **MDVP:Flo(Hz)** (r = 0.6) alongside their identical negative correlation with the target variable, **status** (r = -0.38).

Of the shimmer features, only **MDVP:Shimmer** was selected. This outcome was expected given the extreme multicollinearity across all shimmer measurements. **MDVP:Shimmer** was likely retained because it has the strongest correlation with **status** (r = 0.37).

Surprisingly, **DFA** was picked as a predictor despite having a low correlation to **status**. **DFA** was likely chosen because it has low multicollinearity with other predictors, providing a unique, non-redundant predictive signal that cannot be explained by any other feature.

The last selected feature was **D2**. Along with **RPDE**, both are nonlinear dynamical complexity features. While both share similar correlation profiles across remaining features and have a low mutual correlation with each other (r = 0.24), **D2** was likely prioritized because it has a slightly higher correlation with **status** (r = 0.34) compared to **RPDE** (r = 0.31).

### Train/Test Split
80% of the data was used to train the model to develop significant patterns, while the remaining 20% was used for testing to provide a reliable estiate of error.

### Standardization
*Standard Scaler* was used to scale all numerical variables on a standardized range. Models like SVM rely on calculating distances between points. Features that have a comparatively larger scale will dominate these calculations.

## Modeling

### Support Vector Machine (SVM)
SVM was selected as the primary model because it is well suited for small, numerical datasets. Additionally, through kernel functions, it can model nonlinear relationships between features are the target.

Three SVM kernels were evaluated: linear, polynomial, and radial basis function (RBF). The linear kernel performed comparatively worse than the polynomial and RBF kernels, while the polynomial and RBF kernels produced similar performance.

The hyperparameter search also tested different values of C: 0.1, 1, and 10. C controls the penalty placed on training errors, with larger values placing greater emphasis on correctly classifying the training data. The grid search selected C = 1 based on the highest average F1 score across the cross-validation folds.

### Cross-Validation
Five-fold cross-validation was used to evaluate model performance on unseen data during hyperparameter tuning. The training data was divided into five folds, with four folds used for training and the remaining fold for validation. This process was repeated five times, with each fold serving as the validation set once.

Because the dataset contains multiple recordings from the same individuals, StratifiedGroupKFold was used to ensure that recordings from the same individual remained within the same fold while maintaining the class distribution as closely as possible.

The average validation performance across the five folds was used by GridSearchCV to select the best-performing combination of SVM hyperparameters.

## Model Evaluation
![alt text](image-2.png)

| Metric    | Healthy (0) | Parkinson's (1) |
| --------- | ----------: | --------------: |
| Precision |        1.00 |            0.83 |
| Recall    |        0.40 |            1.00 |
| F1-score  |        0.57 |            0.91 |
| Support   |          10 |              29 |

| Overall Metric |     Score |
| -------------- | --------: |
| Accuracy       | **84.6%** |
| Macro F1       |      0.74 |
| Weighted F1    |      0.82 |

The final model achieved an accuracy of 84.6% on the test set. The model was able to identify all 29 cases of Parkinson's, resulting in a recall of 1.0 for the Parkinson's class. However, recall for the healthy class is substantially lower at 0.40, indicating that the model incorrectly classified several healthy observations as having Parkinson's disease.

Recall for the Parkinson's class was an important consideration because false negatives represent Parkinson's observations that the model fails to identify. Identifying potential Parkinson's cases is particularly relevant for this project because the goal is to explore whether voice measurements can be used as predictive features for Parkinson's disease.

F1-score was used as the primary scoring metric during hyperparameter tuning because it balances precision and recall. The final Parkinson's F1-score of 0.91 reflects the model's ability to identify Parkinson's observations while limiting false-positive classifications. However, the lower recall for the healthy class indicates that the model produced a notable number of false positives.

These results should be interpreted cautiously because the dataset is relatively small and contains multiple recordings from the same individuals. The model is intended as a machine-learning exploration rather than a clinical diagnostic tool.

## References

1. Mayo Clinic Staff. (2024, September 27). *Parkinson's disease - Symptoms and causes*. Mayo Clinic. https://www.mayoclinic.org/diseases-conditions/parkinsons-disease/symptoms-causes/syc-20376055

2. Edit later https://www.statisticalaid.com/lasso-regression/

3. Edit later https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

## Citations

If you use this project in your research, Please cite the following:

> **Dataset:** [UCI Machine Learning Repository: Parkinson's Dataset](https://archive.ics.uci.edu/dataset/174/parkinsons)