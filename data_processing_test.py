import pytest
import numpy as np
import data_processing as dp

##############
#  FIXTURES  #
##############

@pytest.fixture
def string_num_dataset_1():
    X_str = np.array([['3','cat','3.14','healthy'],
                  ['6','dog','2.72','sick'],
                  ['4','giraffe','1.41','healthy']])
    X_num = np.array([  [3.0,0.0,3.14,0.0],
                        [6.0,1.0,2.72,1.0],
                        [4.0,2.0,1.41,0.0]])
    is_category = np.array([False,True,False,True])
    category_indices = [{'cat':0.0,'dog':1.0,'giraffe':2.0},
                                {'healthy':0.0,'sick':1.0}]
    return (X_str,X_num,is_category,category_indices)


@pytest.fixture
def string_num_dataset_2():
    X_str = np.array([  ['female','26','169','59','Writer'],
                    ['female','41','160','62','Doctor'],
                    ['male','38','178','79','Lawyer'],
                    ['female','29','165','58','Graphic Designer'],
                    ['male','52','179','83','CEO'],
                    ['female','31','168','61','Project Manager'],
                    ['male','36','177','76','Engineer'],
                    ['female','24','162','53','Accountant'],
                    ['male','33','178','82','Teacher'],
                    ['male','28','182','75','Engineer']])
    X_num = np.array([  [0,26,169,59,8],
                        [0,41,160,62,2],
                        [1,38,178,79,5],
                        [0,29,165,58,4],
                        [1,52,179,83,1],
                        [0,31,168,61,6],
                        [1,36,177,76,3],
                        [0,24,162,53,0],
                        [1,33,178,82,7],
                        [1,28,182,75,3]],dtype=float) 
    is_category = np.array([True,False,False,False,True])
    gender_index = {'female':0.0, 'male':1.0}
    job_index = {'Accountant':0.0,'CEO':1.0,'Doctor':2.0,'Engineer':3.0,
                 'Graphic Designer':4.0,'Lawyer':5.0,'Project Manager':6.0,
                 'Teacher':7.0,'Writer':8.0}   
    return(X_str,X_num,is_category,[gender_index,job_index])


@pytest.fixture
def numeric_dataset():
    X = np.array([  [73., 61., 18.],
                    [27., 76., 63.],
                    [22., 88., 56.],
                    [70., 43., 83.]])
    return X


@pytest.fixture
def numeric_dataset_with_nans():
    X = np.array([[73., 61., np.nan],
        [27., 76., 63.],
        [22., np.nan, 56.],
        [70., 43., 83.]])
    return X


@pytest.fixture
def X_y_persons_dataset():
    X = np.array([['26','169','59','Writer'],
        ['41','160','62','Doctor'],
        ['38','178','79','Lawyer'],
        ['29','165','58','Graphic Designer'],
        ['52','179','83','CEO'],
        ['31','168','61','Project Manager'],
        ['36','177','76','Engineer'],
        ['24','162','53','Accountant'],
        ['33','178','82','Teacher'],
        ['28','182','75','Engineer']])
    y = np.array(['female', 'female', 'male','female','male','female','male','female','male','male'])
    return (X,y)


###########
#  TESTS  #
###########

def test_read_dataset_csv_1(): # 2p
    """ Read data from CSV file, check correct format and check random sample """
    (header,X) = dp.read_dataset_csv('datasets/palmer_penguins.csv')
    assert header == ['species', 'island', 'culmen_length_mm', 'culmen_depth_mm',
                      'flipper_length_mm', 'body_mass_g', 'sex']
    assert X.shape == (344, 7)
    assert (X.dtype.kind == 'U') or (X.dtype.kind == 'S')
    assert np.all(X[42] == np.array(['Adelie', 'Dream', '36', '18.5', '186', '3100', 'FEMALE']))

def test_read_dataset_csv_2(): # 2p
    """ Read data from CSV file with ; as delimiter """
    (header,X) = dp.read_dataset_csv('datasets/winequality-white.csv',delimiter=';')
    assert header == ['fixed acidity','volatile acidity','citric acid','residual sugar',
                      'chlorides','free sulfur dioxide','total sulfur dioxide',
                      'density','pH','sulphates','alcohol','quality']
    assert X.shape == (4898, 12)
    assert (X.dtype.kind == 'U') or (X.dtype.kind == 'S')
    assert np.all(X[127] == np.array(['6.5','0.24','0.32','7.6','0.038','48','203','0.9958','3.45','0.54','9.7','7']))

def test_encode_category_numeric(): # 2p
    """ Test encoding of text classes as numbers """
    string_array = np.array(['dog', 'cat', 'dog', 'giraffe', 'cat', 'cat'])
    int_array,category_index = dp.encode_category_numeric(string_array)
    assert np.all(int_array == np.array([1,0,1,2,0,0]))
    assert category_index == {'cat':0,'dog':1,'giraffe':2}

def test_convert_text_dataset_to_numeric_1(string_num_dataset_1): # 3p
    """ Test encoding of text dataset as numeric dataset """
    X_num, is_category, category_indices = dp.convert_text_dataset_to_numeric(string_num_dataset_1[0])
    assert np.all(X_num == string_num_dataset_1[1])
    assert np.all(is_category == string_num_dataset_1[2])
    assert category_indices == string_num_dataset_1[3]
    
def test_convert_text_dataset_to_numeric_2(string_num_dataset_2): # 2p
    """ Test encoding of text dataset as numeric dataset """
    X_num, is_category, category_indices = dp.convert_text_dataset_to_numeric(string_num_dataset_2[0])
    assert np.all(X_num == string_num_dataset_2[1])
    assert np.all(is_category == string_num_dataset_2[2])
    assert category_indices == string_num_dataset_2[3]

def test_nobel_laureates_1(): # 2p
    """ Test search of JSON file on Nobel prizes to get list of winners """
    names = dp.nobel_laureates('datasets/nobel_prize.json','literature')
    assert len(names) == 119
    assert names[0] == 'Sully Prudhomme'
    assert names[19] == 'Knut Hamsun'
    assert names[-1] == 'Annie Ernaux'

def test_nobel_laureates_2():  # 1p
    """ Test search of JSON file on Nobel prizes to get list of winners """
    names = dp.nobel_laureates('datasets/nobel_prize.json','physics')
    assert len(names) == 222
    assert names[0] == 'Wilhelm Conrad Röntgen'
    assert names[5] == 'Marie Curie'
    assert names[-1] == 'Anton Zeilinger'

def test_normalize_data_zscore(numeric_dataset): # 3p
    """ Test Z-score normalization of specific columns in dataset """
    X = numeric_dataset
    col_norm = np.array([True,False,True])
    X_norm = dp.normalize_data_zscore(X,col_norm)
    assert np.allclose(X_norm,np.array([
        [ 1.05975976,  61., -1.57127047],
        [-0.8901982 ,  76.,  0.33973416],
        [-1.10215015,  88.,  0.04246677],
        [ 0.93258859,  43.,  1.18906955]])) 
    
def test_remove_missing_data_rows(numeric_dataset_with_nans):  # 2p
    """ Test removal of rows with nans """
    X = numeric_dataset_with_nans
    X_clean = dp.remove_missing_data_rows(X)
    assert np.all(X_clean ==  np.array([[27., 76., 63.],
                                        [70., 43., 83.]]))

def test_impute_missing_data(numeric_dataset_with_nans):      # 3p
    """ Test imputation of missing data with median values """
    X = numeric_dataset_with_nans
    X_clean = dp.impute_missing_data(X)
    assert np.all(X_clean ==  np.array([[73., 61., 63.],
                                        [27., 76., 63.],
                                        [22., 61., 56.],
                                        [70., 43., 83.]]))

def test_train_test_split(X_y_persons_dataset):  # 4p
    """ Test splitting of data into training and test datasets """
    X,y = X_y_persons_dataset
    train_frac = 0.7
    ((X_train,y_train),(X_test,y_test)) = dp.train_test_split(X,y,train_frac)
    
    # Check that split datasets have the correct shape
    assert X_train.shape == (7,4)
    assert y_train.shape == (7,)
    assert X_test.shape == (3,4)
    assert y_test.shape == (3,)

    # Check that elements have been shuffled
    assert not (np.all(X_train == X[0:7]) or np.all(X_test == X[0:3]))
    
    # Concatenate X and y and convert to lists for easier comparison of rows,
    # using expand_dims(y,axis=0) to make y a single-column matrix  
    Xy_list = np.concatenate((X,np.expand_dims(y,axis=1)),axis=1).tolist()
    Xy_train_list = np.concatenate((X_train,np.expand_dims(y_train,axis=1)),axis=1).tolist()
    Xy_test_list = np.concatenate((X_test,np.expand_dims(y_test,axis=1)),axis=1).tolist()

    # Check that data in each split is part of original dataset
    assert all([(row in Xy_list) for row in Xy_train_list])
    assert all([(row in Xy_list) for row in Xy_test_list])

    # Check that there is no overlap between train and test datasets
    assert not(any([(row in Xy_train_list) for row in Xy_test_list]))
