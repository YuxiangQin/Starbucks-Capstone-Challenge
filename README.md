# Starbucks-Capstone-Challenge
Project using Starbucks offer dataset

# Installation
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

# Project Definition

## Project Overview
This project revolves around the analysis of simulated data that mimics customer behavior on the Starbucks Rewards mobile app. Our primary objective is to construct a predictive model that can determine whether a customer is likely to respond positively to a specific offer. By harnessing the power of data science and machine learning techniques, we aspire to enhance the targeting efficiency of Starbucks' promotional offers, ultimately optimizing the user experience on the mobile app.

## Problem Statement
This project aims to develop a predictive model that identifies the demographic groups most responsive to specific offer types within the Starbucks Rewards mobile app. The objective is to empower Starbucks with insights to strategically target and personalize offers for different customer segments. By understanding and leveraging customer behaviors, the model will facilitate more effective promotional campaigns, contributing to an enhanced and tailored experience for Starbucks app users.

## Metrics
To gauge the effectiveness of our predictive model, we have defined the following metrics:

Accuracy Score: This metric will serve as an indicator of the overall correctness of our model's predictions. Achieving high accuracy is crucial to ensuring that the model reliably identifies customers likely to respond positively.

F-score: Given the nature of the problem, we will use the F-score to strike a balance between precision and recall. This is particularly important in scenarios where precision (delivering relevant offers) and recall (capturing all potential responders) need careful consideration.

By employing these metrics, we aim to refine Starbucks' promotional strategy, ensuring that offers are not only personalized but also highly relevant to individual customers. This, in turn, will contribute to a more engaging and rewarding experience on the Starbucks Rewards mobile app.

# Analysis
Analysis contained in Starbucks_Capstone_notebook.ipynb
