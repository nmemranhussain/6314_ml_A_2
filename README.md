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

### Software Used to Implement the Model
- **Software:** Python (with libraries such as Pandas, Scikit-learn, seaborn & matplotlib)

### Version of the Modeling Software: 
- **'pandas'**: '2.2.2',
- **'scikit-learn'**: '1.4.2',
- **'seaborn'**: '0.13.2',
- **'matplotlib'**: '3.8.4**

## Quantitative Analysis

### K-Nearest Neighbors (KNN) classifier Model:

![Plot of Train & Test Accuracy Vs. Number of Neighbors in KNN](ml_6314_A_2) 

**Description**: The plot displays the training and testing accuracy of a K-Nearest Neighbors (KNN) classifier across different values of K, the number of neighbors. The training accuracy starts off high at lower values of K, indicating overfitting, and declines as K increases, suggesting a reduction in model complexity. Conversely, the testing accuracy initially rises as K increases from 2 to 8, highlighting improved model generalization, before stabilizing and showing minor fluctuations beyond K=8, which indicates that further increases in K do not significantly enhance model performance. The peak in testing accuracy around K=8 suggests this as the optimal number of neighbors for balancing model accuracy and generalization, demonstrating a typical trade-off between bias and variance in machine learning model tuning.

| Evaluation Matrix                          |       Model                              |   Values    |
|--------------------------------------------|------------------------------------------|-------------|
| Train Accuracy                             | KNN classifier model with K=5            | 0.7593      |
| Test Accuracy                              | KNN classifier model with K=5            | 0.4444      |
| Optimal Number of K                        | Using NumPy's **np.argmax()**            | 9           |
| Test Accuracy                              | Using NumPy's **np.argmax()**            | 0.5556      |
| Best hyperparameters (Optimal Number of K) | Using GridSearchCV                       | 8           |
| Test accuracy                              | Using GridSearchCV                       | 0.472222222 |

### Logisitic Regression Classifier Model:

| Evaluation Matrix                          |   Values     |
|--------------------------------------------|--------------|
| Train Accuracy                             |  0.7222      |
| Test Accuracy                              |  0.5278      |
| Intercept:                                 | -0.22399632  |
| Coefficients of 'temp'                     | -0.081528    |
| Coefficients of 'percip'                   | 1.351205     |
| Coefficients of 'windspeed'                | -0.040697    |
| Coefficients of 'uvindex'                  | 0.252599     |
| Coefficients of 'icon_partly-cloudy-day'   | 0.779979     |
| Coefficients of 'icon_rain'                | 0.543769     |
| Coefficients of 'icon_snow'                | -0.247574    |

In first Sample, the probability that **PU_ct > DO_ct** in the **first test sample:** 0.4691

### SVM (linear, C = 10) Classifier Model:
| Evaluation Matrix                          |  Values     |
|--------------------------------------------|-------------|
| Linear SVC Training Accuracy               |  0.6852     |
| Linear SVC Testing Accuracy                |  0.5000     |

### SVM (Non-linear, using rbf kernel, C = 10) Classifier Model:

| Evaluation Matrix                          |  Values     |
|--------------------------------------------|-------------|
| Nonlinear SVC Training Accuracy            | 0.8148      |
| Nonlinear SVC Testing Accuracy             | 0.5000      |

## Among the KNN, Logisitc Regression, linear SVC, nonlinear SVC with RBF Kernel, Logistic Regression is the best.  
When evaluating the performance of the different models, the best-performing model can be determined based on both the training and testing accuracies. It's important to look for a model that not only performs well on the training data but also generalizes effectively to new, unseen data (testing data). Here's the summarized result:

- K-Nearest Neighbors (KNN): Optimal K is 8 with a **Testing Accuracy of 0.4722**
- Logistic Regression: Training Accuracy of 0.6667 and **Testing Accuracy of 0.5278**
- Linear SVC: Training Accuracy of 0.6667 and **Testing Accuracy of 0.5000**
- Nonlinear SVC: Training Accuracy of 0.5741 and **Testing Accuracy of 0.5000**
  
Among the models listed—K-Nearest Neighbors (KNN), Logistic Regression, Linear SVC (Support Vector Classifier), and Nonlinear SVC with RBF (Radial Basis Function) kernel—the Logistic Regression performs the best based on the testing accuracy. The Logistic Regression model has a testing accuracy of 0.5278, which is the highest among the four models . Testing accuracy is a critical measure as it indicates how well the model can generalize to new, unseen data. In this case, while the Logistic Regression model does not achieve a very high accuracy, it still outperforms the other models, which have lower testing accuracies (K-Nearest Neighbors (KNN) at 0.4722, Linear SVC at 0.5000, and Nonlinear SVC at 0.5000). This suggests that the Logistic Regression model is more effective in capturing the underlying patterns in the data without overfitting compared to the other models.





