# Exploring Bikeshare Dynamics with Advanced Analytics and Predictive Modeling like Logistic Regression, KNN, SVM (linear & non-linear)  

This repository showcases a comprehensive analysis and predictive modeling project based on the Bikeshare data of Washington, D.C. for February, March, and April 2024. The project focuses on assessing bike and dock availability through pickup and drop-off counts and integrates local weather data to examine the impacts on bikeshare traffic. The core objective is to optimize bike allocations by predicting whether pickups exceed drop-offs using classification models. The analysis includes data merging, cleaning, and extensive visualization of bikeshare and weather data. The project progresses into predictive modeling where different machine learning models like K-Nearest Neighbors (KNN), Logistic Regression, and Support Vector Machines (SVM) are evaluated to find the best performer in predicting bikeshare traffic conditions. Logistic Regression, in particular, demonstrated the highest testing accuracy, indicating its superior generalization capabilities for this dataset. This repository serves as a valuable resource for anyone interested in data analytics within urban planning or transportation sectors, providing insights into handling, visualizing, and modeling complex datasets to inform operational decisions in real-time environments.

## Basic Information
**Names:** N M Emran Hussain  
**Email:** nmemranhussain2023@gmail.com  
**Date:** January 2025  
**Model Version:** 1.0.0  
**License:** [MIT License](LICENSE)

## Intended Use
- **Purpose:** This GitHub repository analyzes the impact of weather on bike-sharing in Washington, D.C., using Logistic Regression, KNN, SVM (linear & non-linear) models to predict bike usage based on variables like temperature and precipitation. It includes data preparation, model building, and visualization scripts, aimed at guiding operational and strategic decisions in urban planning.  
- **Intended Users:** Data Analysts, Data scientists, machine learning enthusiasts, educators.  
- **Out-of-scope Uses:** The model is not intended for production use in any critical applications or real-time decision-making systems.

## Training Data
- **Dataset Name:** Capital Bikeshare Data ('202402-capitalbikeshare-tripdata.csv', '202403-capitalbikeshare-tripdata.csv', '202404-capitalbikeshare-tripdata.csv' & 'DC_weather_2024.csv')  
- **Number of Samples:** 318689, 436947, 490266 & 367  
- **Features Used:** 'temp','precip','windspeed','uvindex'&'icon'
- **Target variable Used:** 'Number of pick-ups (PO_ct) & Number of Drop-offs (DO_ct)
- **Data Source:** [capitalbikeshare-data](https://s3.amazonaws.com/capitalbikeshare-data/index.html)

### Splitting the Data for logistic regression model
The dataset was divided into training and validation data as follows:
- **Training Data Split:** 60%
- **Validation Data Split:** 40%

## Data Dictionary

| Column Name     | Modeling Role  | Measurement Level  | Description                                                                                     |  
|-----------------|----------------|--------------------|-------------------------------------------------------------------------------------------------|
| PU_ct	          | Dependent	     | Ratio	            | Represents the count of bike pickups; used as a response variable in regression models.         |  
| DO_ct	          | Dependent	     | Ratio	            | Represents the count of bike drop-offs; used as a response variable in regression models.       |  
| temp	          | Independent	   | Interval	          | Temperature in degrees Celsius; used to predict bike usage based on weather conditions.         |  
| precip	        | Independent	   | Ratio	            | Precipitation in millimeters; used to assess the impact of rain on bike usage.                  |  
| windspeed	      | Independent	   | Ratio	            | Wind speed in kilometers per hour; considered to study its effect on biking comfort and safety. |  
| uvindex	        | Independent	   | Ordinal	          | UV index, categorized from low to high; used to determine the impact of sun exposure on biking. |  
| icon	          | Independent	   | Nominal	          | Weather condition icon (e.g., sunny, cloudy, rain); used to categorize daily weather visually.  |  

### Differences Between Training and Test Data
- The training data includes the target variables (Number of Pick ups (PO_ct)) and (Number of Drop-offs (DO_ct)) and the independent variables ('temp','precip','windspeed','uvindex' & 'icon'), allowing us to train and evaluate the model, while the test data, based on the split (40%), is used solely for generating predictions to assess model performance on unseen data.

## Model Details
### Architecture  
- This model card utilizes only linear model such as **K-Nearest Neighbors, Logistic Regression, and Support Vector Machines**.

### Evaluation Metrics  
- **Train Accuracy:** This refers to the performance measure of the classification models (like K-Nearest Neighbors, Logistic Regression, and Support Vector Machines) on the same dataset that was used to train the model. It indicates how well the model has learned to predict the outcome (whether pickups exceed drop-offs) from the training data itself. High train accuracy suggests that the model fits the training data well, but it's also important to check against overfitting, where the model performs well only on the training data but not on new, unseen data.
  
- **Test Accuracy:** This measures the performance of the models on a separate set of data that was not used during the training phase. It's used to assess how well the model generalizes to new, unseen data. In this project, test accuracy is crucial for determining which model predicts bikeshare traffic conditions most effectively outside of the training context. High test accuracy implies that the model can reliably predict outcomes in real-world scenarios, which is vital for operational decision-making in urban planning or transportation management.



| Evaluation Matrix                          |       Model                              |   Values    |
|--------------------------------------------|------------------------------------------|-------------|
| Train Accuracy                             | KNN classifier model with K=5            | 0.7593      |
| Test Accuracy                              | KNN classifier model with K=5            | 0.4444      |
| Training Accuracy                          | KNN classifier model with K= 1 to 15     

### Software Used to Implement the Model
- **Software:** Python (with libraries such as Pandas, Scikit-learn, seaborn & matplotlib)

### Version of the Modeling Software: 
- **'pandas'**: '2.2.2',
- **'scikit-learn'**: '1.4.2',
- **'seaborn'**: '0.13.2',
- **'matplotlib'**: '3.8.4**
