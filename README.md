
# Optimizing an ML Pipeline in Azure

## Overview

This project is first part of the Udacity Azure ML Nanodegree. 
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

Here we are analyzing the bank marketing dataset and our goal is to train it so that we can determine if the customer will make a term deposit (yes) or not (no) i.e., a Classification problem.
We will clean the dataset using the clean_data function in train.py (data preparation script) then apply one hot encoding and then the data set is split into train and validation sets.
Our classification algorithm is Logistic Regression and the trained model is tested on the target column i.e., 'y' for predictions.

Hyperparameters are -C (Inverse of regularization strength. Smaller values cause stronger regularization) and
--max_iter (Maximum number of iterations to converge)

Azure Machine Learning supports three types of sampling:
Random sampling, Grid sampling and Bayesian sampling.

We have used Random sampling as Random sampling supports discrete and continuous hyperparameters. It supports early termination of low-performance runs and values are chosen randomly as a result of which it is not much compute intensive and works perfect for project

Grid sampling supports discrete hyperparameters. Use grid sampling if you can budget to exhaustively search over the search space. Supports early termination of low-performance runs but it is compute intensive as compared to Random sampling  

Bayesian sampling is based on the Bayesian optimization algorithm. It picks samples based on how previous samples did, so that new samples improve the primary metric.
Bayesian sampling is recommended if you have enough budget to explore the hyperparameter space. For best results, we recommend a maximum number of runs greater than or equal to 20 times the number of hyperparameters being tuned.
The number of concurrent runs has an impact on the effectiveness of the tuning process. A smaller number of concurrent runs may lead to better sampling convergence, since the smaller degree of parallelism increases the number of runs that benefit from previously completed runs.
Bayesian sampling only supports choice, uniform, and uniform distributions over the search space. So, this method is bit compute intensive for our project.

Azure Machine Learning supports four early termination policies - Bandit policy, Median stopping policy, Truncation selection policy and No termination policy.
We have used Bandit policy as bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run. This allows to compute to spend more time on the other hyper parameters that matters.

Best Run Id:  HD_82a26817-1a5f-48e9-a196-9fdc16830faa_1
Accuracy:  0.9111785533636824

## AutoML

The algorithm pipeline with highest accuracy is VotingEnsemble having accuracy: 0.916578148710167

## Pipeline comparison

Here AutoML is the winner.
Ensemble methods are techniques that create multiple models and then combine them to produce improved results. 
Ensemble methods usually produces more accurate solutions than a single model would that’s why AutoML result of ensemble method is much better as compared to Hyperdrive Logistic Regression model.


## Future work
In future we can work with imbalanced dataset and explore other ways of cleaning data. Try out other models for classification using hyperparameter.
Note: Lower C and high max_iter can cause overfitting so it is important that these variables are tuned carefully.

## Proof of cluster clean up
compute_target.delete()

