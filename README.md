# Supervised Learning Case Study

*by Connor, Cam, Joe, and Marc*

## Table of Contents
- [Introduction](#introduction)
  - [Case Study Goal](#case-study-goal)
  - [Minimum Viable Product](#minimum-viable-product)
  - [Background](#background)
  - [Datasets](#datasets)
- [Cross-validation](#cross-validation)
  - [Kfolds](#kfolds)
- [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Feature Categories](#feature-categories)
- [Regression Models](#regression-models)
  - [Ordinary Least Square](#ordinary-least-square)
  - [Lasso](#lasso)
- [Citation](#citation)

## Introduction

### Case Study Goal
1)	what factors are the best predictors for retention

### Minimum Viable Product

MVP   - 

MVP + -

### Background
A ride-sharing company (Company X) is interested in predicting rider retention. To help explore this question, we have provided a sample dataset of a cohort of users who signed up for an account in January 2014. The data was pulled on July 1, 2014; we consider a user retained if they were “active” (i.e. took a trip) in the preceding 30 days (from the day the data was pulled). In other words, a user is "active" if they have taken a trip since June 1, 2014. The data, churn.csv, is in the data folder. The data are split into train and test sets. You are encouraged to tune and estimate your model's performance on the train set, then see how it does on the unseen data in the test set at the end.

### Datasets  

The `merge_data.ipynb` notebook reads and merges most of the data in the `data` folder into one dataframe. We then pickled this dataframe and loaded it into reg_case_study.py, our main script. Feel free to execute this notebook cell-by-cell to gain a better understand of the data.

The `./data` folder contains data from three publically available sources.  Groups should feel free to supplement this data if they wish.
1. The largest collection of HIV and opioid data was obtained from the [opioid database](http://opioid.amfar.org/) maintained by the American Foundation for AIDS Research (amfAR).  
2. Demographic and economic data were obtained from the 5yr - American Community Survey which are available at the [US census bureau website](https://factfinder.census.gov/faces/nav/jsf/pages/searchresults.xhtml?refresh=t).
3. Estimates for the [MSM population](http://emorycamp.org/item.php?i=48) in each county were obtained from the Emory Coalition for Applied Modeling for Prevention (CAMP).

Data dictionaries that indicate what each column in the data means are included in the folder associated with each data set.


## Exploratory Data Analysis

- 37 columns representing health, 
 
       city: city this user signed up in phone: primary device for this user
      signup_date: date of account registration; in the form YYYYMMDD
      last_trip_date: the last time this user completed a trip; in the form YYYYMMDD
      avg_dist: the average distance (in miles) per trip taken in the first 30 days after signup
      avg_rating_by_driver: the rider’s average rating over all of their trips
      avg_rating_of_driver: the rider’s average rating of their drivers over all of their trips
      surge_pct: the percent of trips taken with surge multiplier > 1
      avg_surge: The average surge multiplier over all of this user’s trips
      trips_in_first_30_days: the number of trips this user took in the first 30 days after signing up
      luxury_car_user: TRUE if the user took a luxury car in their first 30 days; FALSE otherwise
      weekday_pct: the percent of the user’s trips occurring during a weekday
           
 - 3007 rows where each row is a unique account?
 
<p align="center">
  <img src="img/scatter_matrix_second_selection.png" width = 860>
</p>
 
### Feature Categories

#### First Selection 

Our first approach at selecting relevant and useful metrics was to 'eye-ball' the 37x37 scatter matrix. Along with this we carefully studied the column documentation dictionaries to find metrics that we thought would be useful. Below we've listed our results of seven metrics:

      'SA_fac','HIVprevalence','Population','drugdeaths', 'mme_percap','MSM12MTH', 'household_income' 

<p align="center">
  <img src="img/scatter_matrix_first_selection.png" width = 750>
  <img src="img/ols_summary_first_selection.png" width = 600>
</p>

## Cross-validation

### Train/Test

Data was already split

<p align="center">
  <img src="img/kfolds.png" width = 600>
</p>

## Learning Models

### Random Forest

Our first approach 

<p align="center">
  <img src="img/ols_summary_all.png" width = 600>
</p>



### Lasso

We next tried to apply a lasso regression to our train/test sub dataset. We tested lambda values of [1e-15,1e-10,1e-8,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1] to find what lambda value gives us the minimum mean squared error. In the plot below, notice how the mean square error changes with different lambda values:

<p align="center">
  <img src="img/lasso.png" width = 860>
</p>


We used a lambda of 0.01 to train our model on our training dataset. We applied the parameters learned from our training dataset to predict on our holdout dataset. Below you can see our Predicted HIV Incidence rate vs our

<p align="center">
  <img src="img/holdout.png" width = 860>
</p>

## Citation
This case study is based on [Supervised Learning Case-Study-Ride-Share](https://github.com/GalvanizeDataScience/supervised-learning-case-study/tree/Denver/ride-share).  




In this case study you should use non-parametric supervised learning models you were exposed to in class (such as kNN, decision trees, random forest, boosting) to create a predictive machine learning model.  It may be interesting to compare non-parametric to parametric (linear/logistic regression) results.  

Here are your case study options:  

* [Predict ride-sharing company churn](https://github.com/GalvanizeDataScience/supervised-learning-case-study/blob/Denver/ride-share/case_study_description.md)

* [Predict interview attendance](https://github.com/GalvanizeDataScience/supervised-learning-case-study/blob/Denver/interview/case_study_description.md)


Near the end of the day you'll be asked to present capstone results from your project README.    
Please present:  
* Your problem statement  
* A description of your data  
* EDA and feature engineering  
* Your modeling approach  
* How you decided on your final model  
* Model results on unseen data  
* Relevance to the real world  
* Areas for future work  

Notes: 
Be wary of data leakage - unwittingly providing more information about your target in your 
dataset than would be available at the time of prediction.  Read more [here.](https://www.kaggle.com/dansbecker/data-leakage)

To determine real-world relevance, you usually need to bring in your out-of-model knowledge to 
quantify the cost-benefit of performance metrics.  Recall your profit-curve assignment!

