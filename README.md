# Customer Churn Prediction with Machine Learning 

<div align='center'> 
    <img src="https://drive.google.com/uc?export=view&id=1UkZsE2-YDcOBSTEXOyFa3srA4MnVqBui"/>

</div>



This repository contains code and resources for predicting customer churn using machine learning algorithms. The goal is to identify customers who are likely to churn and take proactive measures to retain them.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Building and Evaluation](#model-building-and-evaluation)
- [Model Comparison](#model-comparison)
- [Model Fine-Tuning](#model-fine-tuning)
- [Conclusion](#conclusion)

## Introduction
Customer churn refers to the loss of customers or subscribers by a business. It is important for businesses to understand the factors that contribute to churn and predict which customers are at risk. In this project, we use various machine learning algorithms to build predictive models for customer churn.

Customer churn prediction offers several benefits for businesses, including:
- **Retention**: Identifying customers at risk of churn allows businesses to take proactive measures to retain them, such as offering personalized incentives or improving customer service.
- **Cost Reduction**: Acquiring new customers is often more expensive than retaining existing ones. By predicting churn, businesses can focus their resources on retaining high-value customers and reduce marketing costs.
- **Business Insights**: Analyzing the factors contributing to churn can provide valuable insights into customer behavior and help businesses improve their products and services.

## Dataset
The dataset used for this project contains customer information, such as demographics, usage patterns, and service details. It also includes a target variable indicating whether the customer has churned or not. The dataset is split into training and testing sets for model development and evaluation.

## Data Preprocessing
Before building the models, we perform data preprocessing steps to ensure the data is in a suitable format for training the machine learning algorithms. The preprocessing steps include:
- Handling missing values: We use techniques like imputation to fill in missing values or remove rows with missing data.
- Encoding categorical variables: Categorical variables are converted into numerical representations, such as one-hot encoding or label encoding.
- Scaling numerical features: Numerical features are scaled to a similar range to avoid biasing the models.

## Model Building and Evaluation
We train and evaluate several machine learning models on the preprocessed data. The models used in this project include Decision Tree Classifier, Random Forest Classifier, Logistic Regression Classifier, Support Vector Machine, and Gradient Boosting Classifier. For each model, we compute evaluation metrics such as accuracy, precision, recall, F1 score, and F2 score. We also generate a confusion matrix and a classification report for each model.

## Model Comparison
After evaluating the models, we compare their performance based on the evaluation metrics. The Gradient Boosting model is found to have the best accuracy, precision, and F2 score, indicating its effectiveness in predicting customer churn. Other models, such as Logistic Regression, Support Vector Machine, and Random Forest Classifier, also show good performance in terms of F1 score and accuracy.

## Model Fine-Tuning
To further improve the performance of the selected models, we fine-tune their hyperparameters using Bayesian optimization. The hyperparameters are tuned for the Gradient Boosting, Logistic Regression, Support Vector Machine, and Random Forest Classifier models. The fine-tuned models are then evaluated on the test data, and the evaluation metrics are recorded.

## Conclusion
In this project, we have demonstrated the use of machine learning algorithms for customer churn prediction. By training and evaluating different models, we identified the best performing model and fine-tuned its hyperparameters to achieve better accuracy in predicting customer churn. This information can help businesses take proactive measures to retain customers and improve customer satisfaction.

For detailed code and implementation

**Contact Information** <a name="contact"></a>

<table>
  <tr>
    <th>Name</th>
    <th>Twitter</th>
    <th>LinkedIn</th>
    <th>GitHub</th>
    <th>Hugging Face</th>
  </tr>
  <tr>
    <td>Bright Eshun</td>
    <td><a href="https://twitter.com/bright_eshun_">@bright_eshun_</a></td>
    <td><a href="https://www.linkedin.com/in/bright-eshun-9a8a51100/">@brighteshun</a></td>
    <td><a href="https://github.com/Bright136">@bright136</a></td>
    <td><a href="https://huggingface.co/bright1">@bright1</a></td>
  </tr>
</table>