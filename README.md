# Project: Machine Learning Engineer with Microsoft Azure Capstone

# Table of Contents
<!--ts-->
- [Dataset](#dataset)
  * [Overview](#overview)
  * [Task](#task)
  * [Access](#access)
- [Automated ML](#automated-ml)
  * [Overview of AutoML Settings](#overview-of-automl-settings)
  * [Results](#results)
  * [RunDetails Widget](#rundetails-widget)
  * [Best Model](#best-model)
- [Hyperparameter Tuning](#hyperparameter-tuning)
  * [Overview of Hyperparameter Tuning Settings](#overview-of-hyperparameter-tuning-settings)
  * [Results](#results)
  * [RunDetails Widget](#rundetails-widget)
  * [Best Model](#best-model)
- [Model Deployment](#model-deployment)
  * [Overview of Deployed Model](#overview-of-deployed-model)
  * [Endpoint](#endpoint)
  * [Endpoint Query](#endpoint-query)  
- [Screen Recording](#screen-recording)
- [Suggestions to Improve](#suggestions-to-improve)
 
<!--te-->  

## Dataset

### Overview

The dataset used for this captone project comes from Davide Chicco, Giuseppe Jurman: "Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone". BMC Medical Informatics and Decision Making 20, 16 (2020)

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide. Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.

Twelve (12) clinical features would be included for training:

age: age of the patient (years)
anaemia: decrease of red blood cells or hemoglobin (boolean)
high blood pressure: if the patient has hypertension (boolean)
creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
diabetes: if the patient has diabetes (boolean)
ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
platelets: platelets in the blood (kiloplatelets/mL)
sex: woman or man (binary)
serum creatinine: level of serum creatinine in the blood (mg/dL)
serum sodium: level of serum sodium in the blood (mEq/L)
smoking: if the patient smokes or not (boolean)
time: follow-up period (days)
Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

### Task

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

For this task, Azure auto machine learning will be performed to make accurate prediction on the death event based on patient's 12 clinical features.

### Access

The data (csv file) was downloaded from a Kaggle webesite and uploaded into the Azure Machine Learning via Dataset console as a registered Tabular dataset with a self-defined key "heart-failure".
In the workplace, this registered dataset could be accessed via the key: "dataset = ws.datasets[key]"

## Automated ML

### Overview of AutoML Settings 

With the registered heart failure dataset, a classification automated machine learning training was performed with a criteria of "AUC_weighted". 
After training, the best model was generated as of "VotingEnsemble" with the best metric as of 0.9148. The best run and the working model were retrieved, and the working was registered and deployed as a web service, i.e., real-time endpoint. 
The prediction could be made by running the provided "endpoint.py" file, inside which containes the data points with 12 features. 

For AutoML settings, several critical parameters have to be configured, including primary_metric (i.e., accuracy, AUC_weighted, spearman_correlation, r2-score), the main matric to be used for best model selection), experiment_timeout_minutes (i.e., exit criteria), max_concurrent_iterations (i.e., multiple child runs on clusters), computer_target (i.e., computing cluster), task (i.e., classification, regression, or time series forcasting), training_data (i.e., heart failure dataset), label_column_name (i.e., "DEATH_EVENT"), enable_onnx_compatible (i.e, a onnx model to be used for various applications), enable_early_stopping (i.e., whether to enable early termination if the score is not imprving in the short term in order to improve training efficiency), and featurization (i.e., 'auto" indicates data is automatically scaled and normaized). 

### Results

The best model after training was "VotingEnsemble" with the AUC_weighed matric as of 0.9148.

### RunDetails Widget
![RunDetails Widget](https://github.com/tanglijhu/nd00333-capstone-submission/blob/main/img/automal_RunDetails%20widget.PNG?raw=true)

### Best Model
![best model 1](https://github.com/tanglijhu/nd00333-capstone-submission/blob/main/img/automal_best_model.PNG?raw=true)
![best model 2](https://github.com/tanglijhu/nd00333-capstone-submission/blob/main/img/automal_best_model_run_id-1.PNG?raw=true)
![best model 3](https://github.com/tanglijhu/nd00333-capstone-submission/blob/main/img/automal_best_model-details.PNG?raw=true)


## Hyperparameter Tuning

### Overview of Hyperparameter Tuning Settings 

The classification problem could also be tackled with a Logistic Regression algorithm training from sklearn. Logistis Regression is a go-to binary classification algorithm with the benifits of: 1) easier to use and interpret, 2) very efficient to train, 3) making no assumptions about distributions of classes in feature space.

Hyperparameter tuning was performed by using HyperDriveConfig with specified regression training transcript and hyperparameters. Especially, "C" parameter controls the penality strength (i.e., inverse penlity so smaller value specifies more penality and strong regularization) and "max_iter" parameter means the maximum number of iterations taken for the solvers to converge. In this case, the search space for C parameter was uniformed distributed in 0.1 to 1 and for max_iter was chosen with discreate value as of 50, 100, 150, 200 (100 as default).


### Results

The best metrics after HyperDriveConfig include 'Regularization Strength:': 0.19046682857549246, 'Max iterations:': 200, and 'Accuracy': 0.7833333333333333.
The corresponding parameters include "C": '0.19046682857549246' and 'max_iter': '200'.

### RunDetails Widget
![RunDetails Widget](https://github.com/tanglijhu/nd00333-capstone-submission/blob/main/img/hyperdrive%20RunDetails%20widget.PNG?raw=true)

### Best Model
![best model](https://github.com/tanglijhu/nd00333-capstone-submission/blob/main/img/hyperdrive_best_model_run_id.PNG?raw=true)

## Model Deployment

### Overview of Deployed Model 

The working model was deployed as a web service on Azure Container Instance (ACI). The entry_script used as an input to InferenceConfig was "score.py", which is the best working model generated from AutoML training. Some of the critical deploy configuration parameters include "cpu_cores", "memort_gb", "tags", "auth_enabled", and "primary_key". 

To deploy a working model as a web servce (i.e., endpoint), the workspace name, endpoint name, the registered model, inference_config, and deploy configuration were provided. 

After successful deployment, a REST endpoing with a scoring url was generated to be used for predictions as shown below: 

### Endpoint
![Endpoint](https://github.com/tanglijhu/nd00333-capstone-submission/blob/main/img/model_endpoint.PNG?raw=true)

### Endpoint Query

In order to use the endpoint to make predictions, 'endpoint.py' containing two data series with 12 features for each was implemented and the results were generated. The details could be visualized in the following screen recording. 

## Screen Recording

A [screen recording](https://youtu.be/jsxS3OFomd8) of the project is provided to demonstrate the following steps: 

* a working model
* demo of the deployed model
* demo of a sample request sent to the endpont and its response 

## Suggestions to Improve

* To perform feature engineering, for example, dimension reduction using PCA. PCA enables to represent a multivariate data (i.e., high dimension) as smaller set of variables (i.e., small dimenstion) in order to observe trends, clusters, and outliers. This could uncover the relationships between observations and variables and among the variables.
