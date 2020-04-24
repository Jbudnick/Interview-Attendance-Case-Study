# Talking Points PLEASE REMOVE BEFORE PRESENTATION
- talk about eda
   -  feature selection
   -  one hot encode
- descision to use random forest
   -  good accuracy
- feature importance plot
- how forest performed on data
# Talking Points PLEASE REMOVE BEFORE PRESENTATION


# Supervised Learning Case Study

*by Connor, Cam, Joe, and Marc*

## Table of Contents
- [Introduction](#introduction)
  - [Background](#background)
  - [Case Study Goal](#case-study-goal)
  - [Minimum Viable Product](#minimum-viable-product)
  - [Datasets](#datasets)
- [Cross-validation](#cross-validation)
  - [Train-Test Split](#train-test-split)
- [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Feature Categories](#feature-categories)
- [Learning Models](#learning-models)
  - [Random Forests](#random-forests)
    -[Training](#training)
    -[Results](#results)
- [Citation](#citation)

## **Introduction**

### **Background**
A ride-sharing company (Company X) is interested in predicting rider retention. To help explore this question, we have provided a sample dataset of a cohort of users who signed up for an account in January 2014. The data was pulled on July 1, 2014; we consider a user retained if they were “active” (i.e. took a trip) in the preceding 30 days (from the day the data was pulled). In other words, a user is "active" if they have taken a trip since June 1, 2014. The data, churn.csv, is in the data folder. The data are split into train and test sets. You are encouraged to tune and estimate your model's performance on the train set, then see how it does on the unseen data in the test set at the end.

### **Case Study Goal**

*What factors are the best predictors for retention*

In this case study we will use non-parametric supervised learning models to create a predictive machine learning model. It may be interesting to compare non-parametric to parametric (linear/logistic regression) results.

### **Minimum Viable Product**

**MVP**   : EDA, Describe our random forest classifier 

**MVP +** :

**MVP + +** : Compare non-parametric learning models to parametric ones



### **Datasets**  
Since we are interested in predicting retention we needed to determine wether or not a user is still active. We did this by checking if a user had taken a trip in the last month (from the day this data was pulled).

- `city`: city this user signed up in phone: primary device for this user
- `signup_date`: date of account registration; in the form `YYYYMMDD`
- `last_trip_date`: the last time this user completed a trip; in the form `YYYYMMDD`
- `avg_dist`: the average distance (in miles) per trip taken in the first 30 days after signup
- `avg_rating_by_driver`: the rider’s average rating over all of their trips 
- `avg_rating_of_driver`: the rider’s average rating of their drivers over all of their trips 
- `surge_pct`: the percent of trips taken with surge multiplier > 1 
- `avg_surge`: The average surge multiplier over all of this user’s trips 
- `trips_in_first_30_days`: the number of trips this user took in the first 30 days after signing up 
- `luxury_car_user`: TRUE if the user took a luxury car in their first 30 days; FALSE otherwise 
- `weekday_pct`: the percent of the user’s trips occurring during a weekday


## **Exploratory Data Analysis**
After some inital EDA we saw correlation between 'surge_pct' and 'avg_surge' so we decided to drop 'surge_pct'. We also noticed that drivers were almost always rating their passengers highly and determined that had little impact on retention. 

We originally one hot encoded the 3 different cities. After looking at some graphs and correlation values we notied that King's landing had a lower churn rate to the other cities.  

(city_churn.png)

Again looking at users phone OS we saw differences between Android and iPhone users. 
(phoe_churn.png)
encoding
correlations between android/iphone, winterfell/asaporn/stormwind
trips in the first 30 and luxury cars
non raters
outliers
 
<p align="center">
  <img src="../images/weekday_churn.png" width = 860>
</p>
 
### **Feature Categories**

#### **First Selection** 

Our first approach at selecting relevant and useful metrics was to 'eye-ball' the 37x37 scatter matrix. Along with this we carefully studied the column documentation dictionaries to find metrics that we thought would be useful. Below we've listed our results of seven metrics:

      'SA_fac','HIVprevalence','Population','drugdeaths', 'mme_percap','MSM12MTH', 'household_income' 

<p align="center">
  <img src="img/scatter_matrix_first_selection.png" width = 750>
  <img src="img/ols_summary_first_selection.png" width = 600>
</p>

## **Cross-validation**

### **Train-Test Split**

The `churn_train.csv` and `churn_test.csv` files are an 80:20 train-test-split of the `churn.csv` data.  For some friendly competition, students should only use the `churn_train.csv` file for model and hyperparameter selection, and then at the end of the day see how they do on the unseen `churn_test.csv.`

<p align="center">
  <img src="img/kfolds.png" width = 600>
</p>

## **Learning Models**

### **Random Forests**

We thought it would be a good approach to use a random forest because...

#### **Training**
Our first approach 

<p align="center">
  <img src="img/ols_summary_all.png" width = 600>
</p>

#### **Results**

When we tested on unseen data, our results were: prediction accuracy was...


## **Citation**
This case study is based on [Supervised Learning Case-Study-Ride-Share](https://github.com/GalvanizeDataScience/supervised-learning-case-study/tree/Denver/ride-share).  




  



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



