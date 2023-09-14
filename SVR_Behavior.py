from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd

df = pd.read_csv("data.csv")

def svr_scores(df: pd.DataFrame) -> tuple:
    """
    Calculate R-squared scores and Least Squares Method (LSM) for each column in the DataFrame.
    
    Arguments:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        tuple: A tuple containing two dictionaries. The first dictionary contains the R-squared scores
               for each column, and the second dictionary contains the LSM scores for each column.
    """
    
    scores = {}  # Store the R-squared scores for each column
    lsm = {}  # Store the LSM scores for each column
    
    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Split the data into features (X) and target variable (y)
        X = df.drop(column, axis=1)
        y = df[column]
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        # Create and fit the SVR model
        svr = SVR()
        svr.fit(X_train, y_train)
        
        # Predict the target variable for the test set
        y_pred = svr.predict(X_test)
        
        # Calculate the R-squared score
        r2 = r2_score(y_test, y_pred)
        
        # Calculate the Least Squares Method (LSM) from the hyperplane
        mse = mean_squared_error(y_test, y_pred)
        lsm_score = np.sqrt(mse)
        
        # Store the scores for the column
        scores[column] = r2
        lsm[column] = lsm_score    
        
    return scores, lsm

