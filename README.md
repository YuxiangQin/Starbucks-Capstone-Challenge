# Starbucks-Capstone-Challenge
Project using Starbucks offer dataset

## Installation
packages needed to run the scripts:
- heapq
- sqlalchemy
- xgboost
- sklearn

## Instructions:
Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/portfolio.json data/profile.json data/transcript.json data/customer_offer_data.db data/customer_trans_agg.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/customer_offer_data.db models/classifier.pkl`

## Project Overview
This project revolves around the analysis of simulated data that mimics customer behavior on the Starbucks Rewards mobile app. Our primary objective is to construct a predictive model that can determine whether a customer is likely to respond positively to a specific offer. By harnessing the power of data science and machine learning techniques, we aspire to enhance the targeting efficiency of Starbucks' promotional offers, ultimately optimizing the user experience on the mobile app.

## Problem Statement
This project aims to develop a predictive model that identifies the demographic groups most responsive to specific offer types within the Starbucks Rewards mobile app. The objective is to empower Starbucks with insights to strategically target and personalize offers for different customer segments. By understanding and leveraging customer behaviors, the model will facilitate more effective promotional campaigns, contributing to an enhanced and tailored experience for Starbucks app users.

## Metrics
To gauge the effectiveness of our predictive model, we have defined the following metrics:

Accuracy Score: This metric will serve as an indicator of the overall correctness of our model's predictions. Achieving high accuracy is crucial to ensuring that the model reliably identifies customers likely to respond positively.

F-score: Given the nature of the problem, we will use the F-score to strike a balance between precision and recall. This is particularly important in scenarios where precision (delivering relevant offers) and recall (capturing all potential responders) need careful consideration.

By employing these metrics, we aim to refine Starbucks' promotional strategy, ensuring that offers are not only personalized but also highly relevant to individual customers. This, in turn, will contribute to a more engaging and rewarding experience on the Starbucks Rewards mobile app.

## Description of Input Data

The dataset used for this project is composed of three main components: `portfolio`, `profile`, and `transcript`. The data was loaded from the following sources:

- `portfolio`: Contains information about various offers, including their types, durations, and rewards.
- `profile`: Contains user demographic information such as age, income, and gender.
- `transcript`: Captures user activities related to offers and transactions.

The data was loaded in JSON format. Relevant details include:

- **portfolio:** Columns include `id`, `offer_type`, `difficulty`, `reward`, `duration`, and `channels`.
- **profile:** Columns include `id`, `gender`, `age`, `income`, and `became_member_on`.
- **transcript:** Columns include `person`, `event`, `value`, and `time`.

## EDA (Exploratory Data Analysis)

The exploratory data analysis revealed the following key findings:

- Noise rows with an age of 118 were removed from the `profile` data.
- Additional features such as `tenure_day`, `tenure_month`, and `tenure_year` were derived from the `became_member_on` column in the `profile` data.
- Channels in the `portfolio` data were transformed into separate columns.

## Data Preprocessing

### Cleaning

- Noise rows in the `profile` data were dropped.
- Date-related features were processed in the `profile` data.
- Channels in the `portfolio` data were separated into distinct columns.
- Features such as `offer_id` and `transaction_amount` were extracted from the `value` column in the `transcript` data.

### Feature Engineering

- Additional features such as `tenure_day`, `tenure_month`, and `tenure_year` were derived from the `became_member_on` column in the `profile` data.
- A Priority Queue (PQ) was used to manage the impact of offers on user transactions.

## Modeling
### Initial Solution
Used Random Forest Classifier without hyperparameter tune, result on test set:
Accuracy: 0.67
F1 Score: 0.59
### Model Improvements
Accuracy Socre for trainning set was 0.96, showing a strong overfitting, which is common in
Random Forest models, so leveraging hyperparameter tune to reduce overfitting impact,
also include more models to find best solution.
#### Use multiple models
Try more models and pick the best performer, three models were considered:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **XGBoost Classifier**

A pipeline was used for each model, including preprocessing steps such as scaling.

#### Hyperparameter Tuning

GridSearchCV was employed for hyperparameter tuning. The following parameters were tuned for each model:

- **Logistic Regression:** C
- **Random Forest Classifier:** `n_estimators`, `max_depth`, `min_samples_split`
- **XGBoost Classifier:** `n_estimators`, `max_depth`, `learning_rate`

## Results

Best estimators' for each model:
- Logistic Regression (C=1)
- Random Forest Classifier (n_estimators=150, max_depth=20, min_samples_split=10):
- XGBoost Classifier (n_estimators=200, max_depth=7, learning_rate=0.2):

### Comparison Table (metrics on test set)

| Model                   | Accuracy | F1 Score | ROC-AUC Score |
|-------------------------|----------|----------|---------------|
| Logistic Regression     | 0.65     | 0.54     | 0.63          |
| Random Forest Classifier| 0.69     | 0.61     | 0.67          |
| XGBoost Classifier      | 0.70     | 0.62     | 0.69          |

XGBoost performes the best, based on f1-score.
XGBoost and Random Forest are better than Logistic Regression, due to ensemble models take advantage of numbers of simple models, in general it could result in better results. However, it'll require more resource to train the models.
To determine tuning hyperparameter value list, created train set and test set f1-score change plot for each
hyperparameter, in this way we'll know a reasonable range to select best possible hyperparameter values.
n_estimators, max_depth and learning_rate all helped find a balance between variance and bias, with less estimators, less max_depth, larger learning_rate would reduce model's complexity.  
The F1 score and ROC-AUC score provide insights into the predictive capabilities and discrimination power of each model on the given test set.

## Conclusion

In conclusion, in order to better understand customers behavior when receiving offers, we determine to leverage existing data to build a binary classification model, to predict customer's respond on certain offer.

### Key Findings
The analysis revealed several key findings:
- Customers aged over 40 tend to be more responsive to offers, which means completing discount/buy one get one free offers or making transactions under the influence of informational offers.
- Comparing customer behavior with and without offers, individuals between 20 and 30 years old make more purchases when they receive offers.

### Extensive Data Cleaning
The data cleaning process was comprehensive, addressing various scenarios to ensure the dataset accurately reflects real-world situations. Noteworthy details include:

- Iterative Solution: Adopting an iterative approach to iterate through each row of the training data, resulting in a more accurate dataset.
- Offer Types: For each customer, we considered three types of offers:
  1. **Non-Informational Offer:** Straightforward to assess; a response is confirmed if `offer_complete_time >= offer_view_time`.
  2. **Informational Offer:** Required checking if a transaction occurred during the offer's impact duration. We also considered the presence of concurrent non-informational offers, assuming a transaction is more influenced by the non-informational offer in such cases.
  3. **No Offer:** Examined whether a customer's transaction occurred without any offer impact, providing insights into their behavior without offers.

## Improvements

Possible areas for improvement include:

- Further tuning hyperparameters to enhance model performance.
- Exploring additional feature engineering techniques for more informative features.
- Addressing any remaining limitations and challenges in the current approach.
