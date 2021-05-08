"""
---------------------------------------------------
-------------- MODULE: Transformers ---------------
---------------------------------------------------
This useful module allocates custom classes built 
for making the whole data preparation pipeline 
easier for the user. With transformations wrote
as python classes with BaseEstimator and 
TransformerMixin from sklearn package, the objects
inherits the fit_transform() method that can be
applied on complete data preparation pipelines
from easiest to hardest.

Table of Contents
---------------------------------------------------
1. Initial setup
    1.1 Importing libraries
2. Custom Transformers
    2.1 Initial pipelines
    2.2 Data preparation pipelines
    2.3 Pipelines for model consuption
---------------------------------------------------
"""

# Author: Thiago Panini
# Date: 13/04/2021


"""
---------------------------------------------------
---------------- 1. INITIAL SETUP -----------------
             1.1 Importing libraries
---------------------------------------------------
"""

# Standard libraries
import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


"""
---------------------------------------------------
------------ 2. CUSTOM TRANSFORMERS ---------------
             2.1 Initial pipelines
---------------------------------------------------
"""

# Formatting columns on a DataFrame
class ColumnFormatter(BaseEstimator, TransformerMixin):
    """
    Applies a custom column formatting process in a pandas DataFrame
    in order to standardize the column names through the string functions
    lower(), strip() and replace()

    Parameters
    ----------
    None

    Return
    ------
    :return: df: DataFrame after column formatter process [type: pd.DataFrame]

    Application
    -----------
    cols_formatter = ColumnFormatter()
    df_custom = cols_formatter.fit_transform(df_old)
    """

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
        return df

# Selecting a set of columns in a DataFrame
class ColumnSelection(BaseEstimator, TransformerMixin):
    """
    Filters columns in a DataFrame based on a list passed as a class attribute

    Parameters
    ----------
    :param features: column list to be filtered [type: list]

    Return
    ------
    :return: df: DataFrame after filtering process [type: pd.DataFrame]

    Application
    -----------
    selector = ColumnSelection(features=initial_features)
    df_filtered = selector.fit_transform(df)
    """

    def __init__(self, features):
        self.features = features

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        return df[self.features]

# Transforming a target column
class BinaryTargetMapping(BaseEstimator, TransformerMixin):
    """
    Transforms a raw target column in a binary one composed of 1s and 0s
    based on a positive class passed as a class atribute

    Parameters
    ----------
    :param target_col: target column reference [type: string]
    :param pos_class: categorical target entry to be considered as positive class [type: string]
    :param new_target_name: new target column name after mapping [type: string, default='target']

    Return
    ------
    :return: df: DataFrame after target mapping [pd.DataFrame]

    Application
    -----------
    target_prep = BinaryTargetMapping(target_col='original_target', pos_class='DETRATOR')
    df = target_prep.fit_transform(df)
    """

    def __init__(self, target_col, pos_class, new_target_name='target'):
        self.target_col = target_col
        self.pos_class = pos_class
        self.new_target_name = new_target_name

        # Sanity check: new_target_name must be different from source target_col
        if self.target_col == self.new_target_name:
            self.flag_equal = 1
        else:
            self.flag_equal = 0

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        # Applying mapping based on positive class categorical entry
        df[self.new_target_name] = df[self.target_col].apply(lambda x: 1 if x == self.pos_class else 0)

        # Dropping source target column if applicable
        if self.flag_equal:
            return df
        else:
            return df.drop(self.target_col, axis=1)

# Dropping duplicated data
class DropDuplicates(BaseEstimator, TransformerMixin):
    """
    Drops duplicates on a pandas DataFrame

    Parameters
    ----------
    None

    Return
    ------
    :return: df: DataFrame without duplicates [type: pd.DataFrame]

    Aplicação
    ---------
    dup_dropper = DropDuplicates()
    df_nodup = dup_dropper.fit_transform(df)
    """

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        return df.drop_duplicates()

# Splitting data into train and test
class DataSplitter(BaseEstimator, TransformerMixin):
    """
    Applies train_test_split in order to generated one dataset for training and another one for testing

    Parameters
    ----------
    :param target: target reference column on dataset [type: string]
    :param test_size: test_size argument from train_test_split sklearn function [type: float, default: .20]
    :param random_state: random seed [type: int, default: 42]

    Additional Hints
    ----------------
    X_: inner class attribute that allocates the dataset features before split [1]
    y_: inner class attribute that allocates the dataset target before split [1]
        [1] The X_ and y_ attributes are initialized just before the split and can
            be returned outside this class on project script

    Return
    ------
    :return: X_train: features of training data [type: pd.DataFrame]
             X_test: features of testing data [type: pd.DataFrame]
             y_train: target array of training data [type: np.array]
             y_test: target array of testing data [type: np.array]

    Application
    -----------
    splitter = DataSplitter(target='target')
    X_train, X_test, y_train, y_test = splitter.fit_transform(df)
    """

    def __init__(self, target, test_size=.20, random_state=42):
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        # Setting X_ and y_ attributes
        self.X_ = df.drop(self.target, axis=1)
        self.y_ = df[self.target].values

        return train_test_split(self.X_, self.y_, test_size=self.test_size, random_state=self.random_state)

# Limiting categorical entries
class CategoricalLimitter(BaseEstimator, TransformerMixin):
    """
    Limits categorical columns to have a certain number of entries defined by "n_cat" attribute

    Parameters
    ----------
    :param features: list of columns where this class will be applied [type: list]
    :param n_cat: categorical entries limit [1] [type: int, default=5]
        [1] Any column with more than "n_cat" entries will have its entries grouped as 
            specified on "other_tag" parameter. The order will be defined by quantity (value_counts)
    :param other_tag: tag for all entries that exceed the limit defined by "n_cat" [type: string, default='Other']

    Return
    ------
    :return: df: DataFrame after categorical grouping [type: pandas.DataFrame]

    Application
    -----------
    cat_agrup = CategoricalLimitter(features=['colA', 'colB'], n_cat=3)
    df_prep = cat_agrup.fit_transform(df)
    """
    
    def __init__(self, features, n_cat=5, other_tag='Other'):
        self.features = features
        self.n_cat = n_cat
        self.other_tag = other_tag
        
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        # Iterating over each feature
        for feature in self.features:
            # Generating a list with entries that exceeds the n_cat limit (from n_cat to :)
            other_list = list(df[feature].value_counts().index[self.n_cat:])

            # Overwritting source categorical feature with new group limited by n_cat
            df[feature] = df[feature].apply(lambda x: x if x not in other_list else self.other_tag)
            
        return df

# Applying a mapping on categorical entries based on a dictionary
class CategoricalMapper(BaseEstimator, TransformerMixin):
    """
    Applies a categorical mapper on categorical entries based on a dictionary attribute

    Parameters
    ----------
    :param cat_dict: dictionary with dataset categorical columns on keys and its entries on values [type: dict]
    :param other_tag: tag for categorial entries not in dictionary values [type: string, default='Other']

    Return
    ------
    :return: df: DataFrame after categorical mapper [type: pandas.DataFrame]

    Application
    -----------
    cat_agrup = CategoricalMapper(cat_dict=cat_dict, other_tag=OTHER_TAG)
    df_prep = cat_agrup.fit_transform(df)
    """
    
    def __init__(self, cat_dict, other_tag='Other'):
        self.cat_dict = cat_dict
        self.other_tag = other_tag
        
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        # Iterating over categorical dictionary and transforming columns
        for col, cats in self.cat_dict.items():
            df[col] = df[col].apply(lambda x: x if x in cats else self.other_tag)
            
        return df

# Changing data types
class DtypeModifier(BaseEstimator, TransformerMixin):
    """
    Changes data types from a dataset based on a "mod_dict" dictionary attribute

    Parameters
    ----------
    :param mod_dict: dictionary with dataset columns on keys and its dtype on values [type: dict]
        *example: {'feature_A': str, 'feature_B': int, 'feature_C': float}

    Return
    ------
    :return: df_mod: DataFrame after dtype transformation [type: pandas.DataFrame]

    Application
    -----------
    mod_dict = {'feature_A': str}
    dtype_mod = DtypeModifier(mod_dict=mod_dict)
    df_mod = dtype_mod.fit_transform(df)
    """
    
    def __init__(self, mod_dict):
        self.mod_dict = mod_dict
        
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        # Iterating over dictionary and applying dtype changes
        for col, dtype in self.mod_dict.items():
            df[col] = df[col].astype(dtype)
            
        return df


"""
---------------------------------------------------
------------ 2. CUSTOM TRANSFORMERS ---------------
         2.2 Data preparation pipelines
---------------------------------------------------
"""

# Encoding categorical data
class DummiesEncoding(BaseEstimator, TransformerMixin):
    """
    Applies the encoding based on an Ordinary Encoding process taken by get_dummies pandas method.
    One useful feature of this class is the possibility for returning the final features after the 
    encoding process through the "features_after_encoding" class attribute.
    
    This class should be used on categorical pipelines.

    Additional Hints
    ----------------
    features_after_encoding: class attribute with resultant features after the encoding process [1]
    cat_features_ori: class attribute with source categorical features before encoding [1]
        [1] those features can be returned outside this class for composing the overall model features

    Parameters
    ----------
    :param dummy_na: flag for encoding null values [type: bool, default=True]
    :param cat_features_ori: list of source categorical features of the given dataset [type: list]
    :param cat_features_final: list of expected features for handling with missing encoded features [type: list]
        *with this attribute, the class transform() method can automatically deal with
        encoded columns that aren't on the expected final list. Very useful.

    Return
    ------
    :return: X_dum: Dataframe with categorical attributes after encoding [type: pd.DataFrame]

    Application
    -----------
    encoder = DummiesEncoding(dummy_na=True)
    X_encoded = encoder.fit_transform(df[cat_features])
    """

    def __init__(self, dummy_na=True, cat_features_ori=None, cat_features_final=None):
        self.dummy_na = dummy_na
        self.cat_features_ori = cat_features_ori
        self.cat_features_final = cat_features_final

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        # Saving source categorical features on a class attribute
        if self.cat_features_ori is None:
            try:
                self.cat_features_ori = list(X.columns)
            except AttributeError as ae:
                print('X is a numpy array and this makes impossible saving "cat_features_ori" attribute.')
                print(f'Please set "cat_fatures_ori" attribute directly on the object call')
                print(f'Exception: {ae}')
                return
        else:
            X = pd.DataFrame(X, columns=self.cat_features_ori)

        # Applying encoding
        X_cat_dum = pd.get_dummies(X, dummy_na=self.dummy_na)

        # Joining datasets and dropping source columns
        X_dum = X.join(X_cat_dum)
        X_dum = X_dum.drop(self.cat_features_ori, axis=1)

        # Saving features after encoding on a class attribute
        self.features_after_encoding = list(X_dum.columns)
        
        # Dealing with possible missing categorical features using "cat_features_final"
        if self.cat_features_final is not None:
            missing_features = [col for col in self.cat_features_final if col not in self.features_after_encoding]
            # For any column not encoded, creates one with zero flag
            for col in missing_features:
                X_dum[col] = 0
            
            # Sorting the columns for keeping the original structure
            X_dum = X_dum.loc[:, self.cat_features_final]

        return X_dum

# Filling null data
class FillNullData(BaseEstimator, TransformerMixin):
    """
    Fills null data on a given dataset.

    This class should be used on categorical pipelines.

    Parameters
    ----------
    :param cols_to_fill: list of column for which nulls will be filled [type: list, default: None]
        *set None for all columns
    :param value_fill: value to be filled on null data [type: int, default: 0]

    Return
    ------
    :return X: DataFrame with null data filled dados nulos preenchidos [type: pd.DataFrame]

    Aplicação
    ---------
    filler = FillNullData(cols_to_fill=['colA', 'colB', 'colC'], value_fill=-999)
    X_filled = filler.fit_transform(X)
    """

    def __init__(self, cols_to_fill=None, value_fill=0):
        self.cols_to_fill = cols_to_fill
        self.value_fill = value_fill

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Preenchendo dados nulos
        if self.cols_to_fill is not None:
            X[self.cols_to_fill] = X[self.cols_to_fill].fillna(value=self.value_fill)
            return X
        else:
            return X.fillna(value=self.value_fill)

# Dropping null data
class DropNullData(BaseEstimator, TransformerMixin):
    """
    Drops null data from dropna() pandas method

    Parameters
    ----------
    :param cols_dropna: list of column for which nulls will be dropped [type: list, default: None]
        *set None for all columns

    Retorno
    -------
    :return: X: DataFrame sem dados nulos [type: pd.DataFrame]

    Application
    -----------
    null_dropper = EliminaDadosNulos(cols_dropna=['colA', 'colB', 'colC'])
    X = null_dropper.fit_transform(X)
    """

    def __init__(self, cols_dropna=None):
        self.cols_dropna = cols_dropna

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Dropping null data
        if self.cols_dropna is not None:
            X[self.cols_dropna] = X[self.cols_dropna].dropna()
            return X
        else:
            return X.dropna()

# Selecting top features based on a importance rule
class FeatureSelection(BaseEstimator, TransformerMixin):
    """
    Selects the top k features based on a sorted list. This class can be used to select
    the most important features returned for a trained model

    Parameters
    ----------
    :param feature_importance: feature importance array or a sorted list [np.array]
    :param k: used to select the top k features from the array [type: int]

    Return
    ------
    :return: DataFrame filtered by top k features [pd.DataFrame]

    Application
    -----------
    feature_selector = FeatureSelection(feature_importance, k=10)
    X_selected = feature_selector.fit_transform(X)
    """

    def __init__(self, feature_importance, k):
        self.feature_importance = feature_importance
        self.k = k

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Extract indexes for the top k features and selects the DataFrame
        indices = np.sort(np.argpartition(np.array(self.feature_importance), -self.k)[-self.k:])
        if type(X) is DataFrame:
            return X.iloc[:, indices]
        else:
            return X[:, indices]

# Applying logarithm transformation on features
class LogTransformation(BaseEstimator, TransformerMixin):
    """
    Applies log transformation on numerical features

    Parameters
    ----------
    :param cols_to_log: list of columns to be transformed [type: list]

    Return
    ------
    :return np.log1p(X): DataFrame after log application [type: pd.DataFrame]

    Application
    -----------
    log_transf = LogTransformation()
    X_log = log_transf.fit_transform(X_num)
    """

    def __init__(self, cols_to_log=None):
        self.cols_to_log = cols_to_log

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Applying transformation in just some columns
        if self.cols_to_log is not None:
            if type(X) is DataFrame:
                X[self.cols_to_log] = np.log1p(X[self.cols_to_log])
                return X
            elif type(X) is ndarray:
                X[:, self.cols_to_log] = np.log1p(X[:, self.cols_to_log])
                return X
        # Applying tranformation on all columns
        else:
            return np.log1p(X)

# Applying logarithm transformation with a selection flag that can be tunned
class DynamicLogTransformation(BaseEstimator, TransformerMixin):
    """
    Applies log transformation on numerical features. This class has an "application"
    boolean flag that guides the log transformation and it can be tunned on pipelines

    Parameters
    ----------
    :param num_features: numerical features list [type: list]
    :param cols_to_log: list of columns to be transformed [type: list]
    :param application: flag for applying or not the class transform method [type: bool]

    Return
    ------
    :return np.log1p(X): DataFrame after log application [type: pd.DataFrame]

    Application
    -----------
    log_transf = LogTransformation()
    X_log = log_transf.fit_transform(X_num)
    """
    
    def __init__(self, num_features, cols_to_log, application=True):
        self.num_features = num_features
        self.cols_to_log = cols_to_log
        self.application = application
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Verifying application flag
        if not self.application:
            return X
        else:
            # Extracting columns indexes from num_features list
            cols_idx = [self.num_features.index(col) for col in self.cols_to_log]
            
            # Applying transformation in case of data on array or dataframe type
            if type(X) is ndarray:
                X[:, cols_idx] = np.log1p(X[:, cols_idx])
            elif type(X) is DataFrame:
                X.iloc[:, cols_idx] = np.log1p(X.iloc[:, cols_idx])
            return X   

# Applying scaling transformation with a selection flag that can be tunned
class DynamicScaler(BaseEstimator, TransformerMixin):
    """
    Applies a scaling process on numerical features. If the "scaler_type" class
    attribute is set for None, there is no transformation applied. This 
    attribute can be further tunned on built pipelines

    Parameters
    ----------
    :param scaler_type: normalization type [type: string]
        *options: [None, 'Standard', 'MinMax']

    Return
    ------
    :return X_scaled: array after normalization

    Application
    -----------
    scaler = DynamicScaler(scaler_type='Standard')
    X_scaled = scaler.fit_transform(X)
    """
    
    def __init__(self, scaler_type='Standard'):
        self.scaler_type = scaler_type
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Applying normalization based on scaler_type attribute
        if self.scaler_type == 'Standard':
            scaler = StandardScaler()
            return scaler.fit_transform(X)
        elif self.scaler_type == 'MinMax':
            scaler = MinMaxScaler
            return scaler.fit_transform(X)
        else:
            return X                 


"""
---------------------------------------------------
------------ 2. CUSTOM TRANSFORMERS ---------------
        2.3 Pipelines for model consuption
---------------------------------------------------
"""

# Using a trained model and creating new prediction columns on a source dataset
class ModelResults(BaseEstimator, TransformerMixin):
    """
    Executes the prediction methods of a trained model and appends prediction results on new columns of the source dataset

    Parameters
    ----------
    :param model: trained estimator to be used on predictions [type: estimator]
    :param features: model features extracted after the preparation process [type: list]

    Return
    ------
    :return df_pred: DataFrame with model features and model prediction results [type: pd.DataFrame]

    Application
    -----------
    model = trainer._get_estimator(model_name='RandomForest')
    model_exec = ModelResults(model=model, features=MODEL_FEATURES)
    df_pred = model_exec.fit_transform()
    """
    
    def __init__(self, model, features):
        self.model = model
        self.features = features
        
    def fit(self, X):
        return self
    
    def transform(self, X):
        # Creating a DataFrame with model features
        df_final = pd.DataFrame(X, columns=self.features)
        
        # MAking predictions and append new columns
        df_final['y_pred'] = self.model.predict(X)
        df_final['y_scores'] = self.model.predict_proba(X)[:, 1]
        
        return df_final        

        