import numpy as np
import csv
import json


def read_dataset_csv(csv_file_path,delimiter=','):
    """ Read dataset from CSV file and return as numpy array 
    
    # Input arguments:
    csv_file_path:  Path to CSV file (str or pathlib.Path)
    delimiter:      Delimiter used in file. ',' (default), ';' and ' '
                    are common choices. 

    # Returns:
    header:         List of column headers (strings)
                    Length (n_features,)
    X:              Numpy matrix, shape (n_samples, n_features),
                    with string datatype (e.g '<U6' for strings max. 6 chars long)        
    
    # Notes:
    - The function assumes that the first line of the file contains column
    headers (text), delimited in the same style as the data itself.
    """
    pass


def encode_category_numeric(string_array,dtype=int):
    """ Encode category strings as numbers
    
    # Input arguments:
    string_array:   NumPy array of strings representing categories in dataset
                    (e.g. ['dog','cat','dog','giraffe'])
    dtype:          Data type of numeric output array. 
                    Examples: int (default) or float. 
    
    # Returns:
    num_array:      Array of integers representing categories in string_array.
                    Each unique category is encoded with a single integer.
                    Integers are assigned according to the alphabetical ordering
                    of the (unique) categories, starting from 0.
                    For string_array = ['dog','cat','dog','giraffe'], num_array = [1,0,1,2]
    category_index: Dictionary with keys correspoding to category names (strings)
                    and values corresponding to category numbers.
                    For example above, category_index =
                    {'cat':0,'dog':1,'giraffe':2} 

    # Notes:
    - NumPy function np.unique() could be useful.
    """
    pass


def convert_text_dataset_to_numeric(X_str):
    """ Convert NumPy array of strings to numeric data 
    
    # Input arguments:
    X_str:      NumPy matrix, shape (n_samples, n_features),
                with string datatype (e.g '<U6' for strings max. 6 chars long) 

    Returns:
    X_num:      NumPy matrix corresponding to X_str, but with string data
                represented as numbers (float).
                Strings which represent numbers are converted directly 
                (e.g. '3.14' or '42' are converted to 3.14 and 42.0)
                Strings representing categories are converted to 
                whole numbers, but are represented in floating point.
                Example: ['dog','cat','dog','giraffe'] is converted to 
                [1.0, 0.0, 1.0, 2.0] (see encode_category_numeric()).
                The distinction between numeric and category is inferred
                from the content of the first data row. 
    is_category         Boolean vector indicating which columns of X_num 
                        correspond to categories.
    category_indices:   List of category indices (see encode_category_as_int()),
                        one index per column with categories. 
    """
    pass
    

def nobel_laureates(json_file_path, category):
    """ Return full name of all Nobel prize winners (laureates) in given category 
    
    # Input arguments:
    json_file_path:     Path to JSON file (str or pathlib.Path) with Nobel prizes
    category:           String indicating Nobel prize category
                        (e.g. 'medicine' or 'literature')

    # Returns:
    winners:    List of full names for every Nobel prize winner
                in given category, ordered chronologically from first to last.
                In years with multiple winners, winners are listed in the same order
                as they are listed in the JSON file.  
                Example for literature: (['Sully Prudhomme', ..., 'Annie Ernaux']) 
    """
    pass

def normalize_data_zscore(X,norm_col):
    """ Normalize data columns by subtracting mean and dividing by std.dev. ("z-score")
    
    # Input arguments:
    X           Numpy array (matrix) with numerical data,
                shape (n_samples,n_features)
    norm_col    Boolean vector, shape (n_features).
                Set norm_col[i] = True to indicate that column i should be normalized.

    # Returns:
    X_norm      Normalized version of X. In columns indicated by norm_col,
                the mean column value is first subtracted, and the resulting 
                value is scaled by dividing with the column standard deviation.

    # Notes:
    - Numpy methods like np.copy(), np.mean() and np.std() could be useful.
    - See also https://en.wikipedia.org/wiki/Standard_score 
    """
    pass


def remove_missing_data_rows(X):
    """ Detect and remove rows in data matrix with missing values (NaN)
    
    # Input arguments:
    X:          NumPy matrix with data, shape (n_samples, n_features)
                Elements of X with missing data are represented by
                the constant np.nan ("not a number").

    # Returns:
    X_clean     Version of X with all rows with missing data removed

    # Notes:
    - Numpy method np.isnan() could be useful.
    """
    pass


def impute_missing_data(X):
    """ Fill in missing values in data matrix using per-column median value 
    
    # Input arguments:
    X:          NumPy matrix with data, shape (n_samples, n_features)
                Elements of X with missing data are represented by
                the constant np.nan ("not a number").

    # Returns:
    X_clean:    Copy of X with nan values replaced with per-column
                median values.

    # Notes:
    - Numpy methods np.copy(), np.isnan() and np.median() could be useful.
    """
    pass


def train_test_split(X, y, train_frac):
    """ Shuffle and split dataset into separate parts for training and testing 
    
    # Input arguments:
    X:          NumPy matrix with data, shape (n_samples, n_features)
    y:          Numpy vector with values to be predicted. Shape (n_samples,)
    train_frac: Fraction of samples to be placed in training dataset
                (between 0 and 1)

    # Returns:
    train_data: Tuple (X_train,y_train), corresponding to training "split"
                of original data in X and y.
                The number of samples in train_data is equal to
                round(n_samples*train_frac)
    test_data:  Tuple (X_train,y_train), corresponding to training "split"
                of original data in X and y.
                The samples contained in test_data corresponds to the elments 
                of X and y not included in train_data.

    # Notes:
    - The samples in X and y are randomly shuffled before the split is performed. 
    The shuffling is exactly the same for X and y (each row in X still has the 
    same corresponding value in y after both are shuffled).
    - A NumPy random generator can be used for shuffling: 
    https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.permutation.html 
    """
    pass
