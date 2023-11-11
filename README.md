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

Three models were considered for binary classification:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **XGBoost Classifier**

A pipeline was used for each model, including preprocessing steps such as scaling.

## Hyperparameter Tuning

GridSearchCV was employed for hyperparameter tuning. The following parameters were tuned for each model:

- **Logistic Regression:** C
- **Random Forest Classifier:** `n_estimators`, `max_depth`, `min_samples_split`
- **XGBoost Classifier:** `n_estimators`, `max_depth`, `learning_rate`

## Results

Achieved metrics for each model on the test set:

### Logistic Regression:
- Accuracy: 0.65
- F1 Score: 0.54
- ROC-AUC Score: 0.63

### Random Forest Classifier:
- Accuracy: 0.69
- F1 Score: 0.61
- ROC-AUC Score: 0.67

### XGBoost Classifier:
- Accuracy: 0.70
- F1 Score: 0.62
- ROC-AUC Score: 0.69

The F1 score and ROC-AUC score provide insights into the predictive capabilities and discrimination power of each model on the given test set.

## Comparison Table

| Model                   | Accuracy | F1 Score | ROC-AUC Score |
|-------------------------|----------|----------|---------------|
| Logistic Regression     | 0.65     | 0.54     | 0.63          |
| Random Forest Classifier| 0.69     | 0.61     | 0.67          |
| XGBoost Classifier      | 0.70     | 0.62     | 0.69          |

## Conclusion

In conclusion, the XGBoost Classifier demonstrated the best performance based on the F1 score among the evaluated models. The project successfully addressed the binary classification problem, providing insights into user response to offers.

## Improvements

Possible areas for improvement include:

- Further tuning hyperparameters to enhance model performance.
- Exploring additional feature engineering techniques for more informative features.
- Addressing any remaining limitations and challenges in the current approach.
