## End to End Machine Learning Project

To view the Project : [https://student-performance-4l3x.onrender.com](https://student-performance-4l3x.onrender.com)

## Student Performane Checker

1. Understanding problem statement
2. Data collection
3. Performing data checks
4. Data analysis
5. Pre-processing Data 
6. Model training
7. Choose the best model


-  problrm statement
    - The projects shows how the student's performance is affected by other variables such as gender, parental level education, ethnicity, race , luch etc..

-  Data collection and Packages
    - Datset consists of 8 rows and 1000 columns
    E:\PROJECTS\ML_Project\notebook\data\student_performance.csv
    - Import Pandas, Numpy, Matplotlib, Seaborn and warings library
    - DATASET INFO
        - gender : sex of students -> (Male/female)
        - race/ethnicity : ethnicity of students -> (Group A, B,C, D,E)
        - parental level of education : parents' final education ->(bachelor's degree,some college,master's degree,associate's degree,high school)
        - lunch : having lunch before test (standard or free/reduced)
        - test preparation course : complete or not complete before test
        - math score
        - reading score
        - writing score

    
-  Checking all the missing values
    - missing values, duplicates, data type, no of unique values 
    - check statistics of dataset, categories present in various forms


-  Analyszing the data using Visualization techniques
    - Comparing between gender and avearge
    - comparing between gender and test score
    - Plotting graghs for statistical data
    - plotting graphs for each columns 
    - comparing the values of columns between each other


-  Training the models
    - import basic libraries
    - import models such as
        - mse, mae, r2_score
        - KNN
        - Decision tree
        - RF, AdaBoost
        - SVR
        - Linearregression, Ridge, Lasso
        - randomizedsearchCV
        - CarBoost
        - XGB
    - create X and y variables from the dataset
        - remove math_score from dataset and assign to y variable
    - Column wise pre-processing for ml model
        - Numerical column - standard scaling technique
        - categorical column - oneHotEncoding techinique
        - ColumnTransformer - combining both

        - prepare seperate numeric and categorical column
        - import transformers
        - Define Individual transformer
        - Combine using ColumnTransformer
    - Preprocessor in a pipeline
    - Preparing dataset of training and testing from train_test_split 
    - Creating a evaluation function fo the model using mae, mse, rmse, r2_score


-  Comparing the performance of the multiplr regression model on the datset
    - using metrics like mae, mse, rmse, r2_score
    - training multiple regression model on X_train and X_test
    - types of models
        - "Linear Regression": LinearRegression(),
        - "Lasso": Lasso(),
        - "Ridge": Ridge(),
        - "K-Neighbors Regressor": KNeighborsRegressor(),
        - "Decision Tree": DecisionTreeRegressor(),
        - "Random Forest Regressor": RandomForestRegressor(),
        - "XGBRegressor": XGBRegressor(), 
        - "CatBoosting Regressor": CatBoostRegressor(verbose=False),
        - "AdaBoost Regressor": AdaBoostRegressor()
    - To pick a model - model = list(models.values())[i]
    - To train the model - model.fit(X_train, y_train)
    - predict and evaluate the predictions 
    - Later the model performace and r2-score are evaluated in a decreasing order

- Linear Regression model is choosed based on above workings
    - y_pred and y_test are plotted
    - Differnece between the actual and prdicted values are seen

- This the working of the whole model and even ypou can add more features for the dataset...
