## End to End Solar-Flare-Regression-Model

**The purpose of this model is to predict the expected/mean number of solar flares of common, moderate and severe intensity in a 24H period, from data about the sunspot being studied.**

*Throughout the process, I have consulted a wide range of online resources and AI tools as most of the concepts covered were new to me, and I used these tools to investigate the challenges I faced, as well as to seek clarity about what certain results truly mean.*

The final model uses **Poisson Regression** to predict values for each flare class. 

# Tech stack: 

- **Pandas and Scikit-learn** for model building and Exploratory Data Analysis (Matplotlib and Seaborn were also used for support in visualisation but are omitted from the final notebook)
- **Joblib** model saving/pickling
- **Streamlit** UI/frontend
- **FastAPI, Pydantic, Uvicorn and Requests** backend 
- **Vercel** Deployment 

(My original intentions were to containerise the model using *Docker*, as this would be an added learning opportunity. However, due to my system's repeated compatibility issues with Docker, I had to skip this step. )

# Model Development

- **Exploratory Data Analysis**

The data used is from the *UCI Machine Learning Repository*, describing the various features of a sunspot and how these translated to **common**, **moderate** and **severe** solar flares. Having 3 target classes was a point of challenge, which I was able to acknowledge whilst having to choose models to train. The model contained features that were not numerical - modified Zurich class, largest spot size and spot distribution, classed by letter according to scientific classifications. Initially, I decided to map the values to corresponding integers, as these categories seemed somewhat ranked in intensity of a solar flare. However, after training several models, the scores produced were questionable, as models tend to see mapped integers as having a linear relationship, which was not exactly true for sunspot classification. Hence, I switched my approach to using one-hot encoding, despite the increase in the number of columns.

-  **Model Training and Testing**

My initial thoughts were to compare the performance of three sklearn models to be able to choose the most accurate one, after hyperparameter tuning. I had chosen the *RandomForestRegressor*, *RidgeRegression* and *KNearestRegressor* models, due to their capabilities in handling multi-output regression. I used *GridSearchCV* for RandomForest and KNearestResgressor, and RidgeCV for RidgeRegression to tune and cross validate for the best set of hyperparameters for each model. One challenge faced here was the presence of memory issues, where not every hyperparameter was tuned, and kernal crashes meant that I had to leave out some hyperparameters. These were scored based on the negative mean squared error. I fit each tuned model to the training data. Surprisingly, all three models had a very similar mean squared error of *-0.48 to -0.51*. This level of similarity was also the case when testing the model on the test dataset, with both the mean squared error and mean absolute error (results in notebook). I decided to choose the RidgeRegressor due to it having a very slightly lower MAE than the others, choosing it over RandomForest due to its speed. 

-  **Streamlit frontend**

I decided to use Streamlit for the frontend interface due to its simplicity and ability to integrate to a FastAPI backend. I used dropdown boxes for user input to prevent validation errors, ensuring that the input is consistent with the formats expected by the model. 

-  **FastAPI backend**

I used a *POST* request to send data from the frontend to the backend for the model to use. This involved various challenges such as formatting errors with column names, leading to the use of aliases to match variable names in the Pydantic Basemodel. Another source of challenge was being able to return the predicted DataFrame from the endpoint, where the format had to be manipulated to return the details to the user. 

This process exposed a flaw in the model, where negative answers were being returned for the number of solar flare counts, which is not possible. I then came across the use of the PoissonRegressor, specially used for count data, which meant that I had to revert to the EDA. After tuning and testing the model, the scores were in the same range as the previous three, which is why I decided to go with this in the end. After using this model, the negative scores were no longer there (as a result of the log link function incorporated into the Poisson model), and I proceeded with this. 


# Evaluation 

The model itself had a mean squared error of 0.5234404893970696 on testing data. This would be an 'average' mean across the three classes, with each having its own range of values. The 'common flares class' ranged from 0-8, whilst the 'severe flares' class ranged from 0-2, which shows that the error would be relatively greater in the latter. I would consider separately evaluating each target class rather than using a single statistic and grouping them together to be able to look at how the model performs individually. 

Model tuning, especially in the earlier stages where I was comparing three regression models, was timetaking, especially with RandomForestRegressor, an already computationally expensive model being tested on thousands of combinations of hyperparameters. In fact, due to computational limitations, I was not able to tune each and every single parameter for each model, meaning, opportunities for further optimisation of the models to ensure better error scores were missed. Also, it would have been of greater interest to see how the scores fluctuated across multiple runs of testing. Libraries like Optuna and MLFlow, as seen in other projects, could be used to handle tuning and saving of each model instance's metrics. 

The model itself returns float values for the expected/mean number of solar flares in a 24 hour period, could seem counterintuitive. However, this is the premise of poisson regression, and generally, regression models in ML. I decided to stay with this model as poisson regression is statistically used to count discrete events in a set time period. Moreover, I felt that classification models would have limited the values returned to the user to what was originally in the training dataset, even if it would have returned integers. 

Zurich classifications for sunspots include the classes A, B, C, D, E, F and H. However, the training data did not have any sunspots of class A, which meant that I had to leave it out. Moreover, the class imbalance in the dataset could be handled by different approaches like trying different sampling methods. 


### Key online sources and references which were especially helpful throughout - not an exhaustive list: 
https://www.youtube.com/watch?v=luJ64trcCwc - Reference to overall structure of an end to end ML project 

https://www.kaggle.com/datasets/stealthtechnologies/solar-flares-dataset - Dataset - source - UCI ML Repo 

Details for variable information - https://archive.ics.uci.edu/dataset/89/solar+flare - according to the authors, the dataset is intended for the purpose of predicting the number of each type of solar flare in a particular 24H period. 

https://www.youtube.com/watch?v=xi0vhXFPegw - EDA walkthrough 

https://www.aavso.org/zurich-classification-system-sunspot-groups - Zurich Classification of sunspot groups 

https://www.stce.be/educational/classification - McIntosh classification system for sunspots 

Documentation: Pandas, Scikit-Learn, Streamlit, FastAPI

https://www.geeksforgeeks.org/pandas/pandas-replace-multiple-values-in-python/ - Mapping data in df

https://medium.com/@fraidoonomarzai99/hyperparameters-tunning-and-cross-validation-in-depth-d0918b62d986 - CV and Hyperparameter tuning 

https://www.youtube.com/watch?v=wwfCZz3VKlY - Gridsearch

https://www.geeksforgeeks.org/machine-learning/multioutput-regression-in-machine-learning/ - Multi output regression

https://stackoverflow.com/questions/71276813/difference-between-ridgecv-and-gridsearchcv - Ridge CV vs grid search CV
 
https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall ML metrics 

https://medium.com/@faheemsiddiqi789/how-can-i-determine-if-my-data-is-balanced-or-imbalanced-080819af408c - Class 
imbalance

https://stackoverflow.com/questions/54953967/your-session-crashed-after-using-all-available-ram-in-google-collab Kernel/RAM crashes

https://stackoverflow.com/questions/72101295/python-gridsearchcv-taking-too-long-to-finish-running - Time taken for GridSearch

https://scikit-learn.org/stable/model_persistence.html - Model persistence

https://www.youtube.com/watch?v=-WfuEJfItjY - Joblib walkthrough

https://www.youtube.com/watch?v=c1n5iCMzr9E - Streamlit walkthrough

https://www.youtube.com/watch?v=ZnDmTGgYMn0 - FastAPI walkthrough

https://datascience.stackexchange.com/questions/124959/use-prediction-after-using-get-dummies-in-pandas - Pandas get dummies

https://www.geeksforgeeks.org/machine-learning/deploying-ml-models-as-api-using-fastapi/ - FastAPI

https://testdriven.io/blog/fastapi-streamlit/ - FastAPI 

https://stackoverflow.com/questions/73326689/fastapi-post-error-422-detail-locbody-file-msgfield-required - Error messages

https://stats.stackexchange.com/questions/203872/what-to-do-when-a-linear-regression-gives-negative-estimates-which-are-not-possi - Linear regression estimations

https://stats.stackexchange.com/questions/160180/regression-models-to-only-predict-integers-instead-of-floating-point-numbers  - Integers vs floats in regression

https://www.statology.org/a-beginners-guide-to-generalized-linear-models-glms/ - GLMs

https://stackoverflow.com/questions/43532811/gridsearch-over-multioutputregressor - Gridsearch over Multi output regression

https://stackoverflow.com/questions/49416697/statsmodel-poisson-prediction-return-floats-instead-of-whole-numbers - Poisson floats vs integers

https://medium.com/@hannah.hj.do/interpreting-poisson-regression-125f016c1aa6 - Interpreting poisson models

https://www.youtube.com/watch?v=u49-v0HxKoU - Poisson regression interpretation

https://www.youtube.com/watch?v=d5Q-St5e6WM - Poisson regression

https://stackoverflow.com/questions/79867833/what-does-poissonregression-predict-actually-return-in-sklearn/79867987#79867987 - I asked a question on SO about what is being returned by the poisson reg model, and if I can directly use the output values. 

https://stackoverflow.com/questions/62658215/convergencewarning-lbfgs-failed-to-converge-status-1-stop-total-no-of-iter - Poisson convergence 

https://www.geeksforgeeks.org/pandas/add-column-names-to-dataframe-in-pandas/ - Columns in df

https://leapcell.io/blog/how-to-use-python-requests-for-post-requests - Interpret FastAPI output

https://stackoverflow.com/questions/79870602/docker-workaround-for-macos-12 - Question I posted about Docker workarounds

https://www.geeksforgeeks.org/machine-learning/how-to-handle-imbalanced-classes-in-machine-learning/ - Strategies to fix class imbalance