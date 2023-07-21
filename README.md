# Wine_Quality_Chemical_Prediction
This code develops machine learning workflow for predicting the quality of wine based on its chemical properties. It uses various regression models, including Linear Regression, Random Forest, Support Vector Regression, Gradient Boosting Regression, and Ridge Regression. Additionally, it incorporates a Neural Network using TensorFlow/Keras for the same task.

Let's break down the code step by step:

1. The first line is a comment, indicating that the Seaborn library is being installed using pip.

2. Importing necessary libraries:
   - `numpy` (as np): A library for numerical operations in Python.
   - `pandas` (as pd): A library for data manipulation and analysis, particularly using data frames.
   - `matplotlib.pyplot` (as plt): A library for creating visualizations in Python.
   - `seaborn` (as sns): A powerful library for data visualization based on Matplotlib, which adds additional functionality and aesthetics.

3. Importing machine learning-related libraries:
   - `train_test_split`: A function from scikit-learn for splitting data into training and testing sets.
   - Various regression models: `LinearRegression`, `RandomForestRegressor`, `SVR` (Support Vector Regression), `GradientBoostingRegressor`, and `Ridge`.
   - `mean_squared_error` and `r2_score`: Evaluation metrics for regression models.
   - `StandardScaler`: A preprocessing module from scikit-learn used for feature scaling.
   - `GridSearchCV`: A function for performing hyperparameter tuning using cross-validation from scikit-learn.
   - `tensorflow.keras` and `layers`: Libraries for building and training neural networks using TensorFlow and Keras.

4. The code loads the wine quality dataset from a CSV file called 'winequality-red.csv' and performs Exploratory Data Analysis (EDA) to understand the data better. It displays a summary of the dataset, including information about data types and missing values, and shows descriptive statistics. Furthermore, it creates a heatmap of the correlation matrix and a pair plot of features colored by wine quality to visualize relationships between variables.

5. The code prepares the data for machine learning. It separates the features (X) from the target variable (y) and then normalizes the features using `StandardScaler`.

6. Next, the data is split into training and testing sets using `train_test_split`.

7. The code initializes and trains several regression models (Linear Regression, Random Forest, SVR, Gradient Boosting Regression, and Ridge Regression). For each model, it performs the following steps:
   - Fits the model on the training data.
   - Makes predictions on the test data.
   - Evaluates the model using mean squared error (MSE) and R-squared (R2) metrics.
   - Performs cross-validation and calculates the mean R-squared across the folds.

8. After evaluating different models, the code performs hyperparameter tuning for the Random Forest model using `GridSearchCV`. It searches for the best combination of hyperparameters (n_estimators, max_depth, min_samples_split, and min_samples_leaf) that yield the highest R-squared score.

9. Finally, the code creates a Neural Network using TensorFlow/Keras. The network architecture consists of three layers: two hidden layers with ReLU activation and one output layer. The model is compiled with the Adam optimizer and the mean squared error loss function. It is then trained on the training data for 100 epochs.

10. The performance of the Neural Network model is evaluated using the mean squared error and R-squared metrics.
