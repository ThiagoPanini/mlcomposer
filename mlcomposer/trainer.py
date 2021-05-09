"""
---------------------------------------------------
----------------- MODULE: Trainer -----------------
---------------------------------------------------
This must-use module has excellent classes and
methods for complete training and evaluating 
Machine Learning models with just few lines of code.
It really encapsulates almost all the hard work of
data scientists and data analysts by presenting
classes with methods for training, optimizing
models hyperparemeters, generating performance
reports, plotting feature importances, confusion
matrix, roc curves and much more!

Table of Contents
---------------------------------------------------
1. Initial setup
    1.1 Importing libraries
    1.2 Setting up logging
    1.3 Functions for formatting charts
    1.4 Functions for saving objects
2. Classification
    2.1 Binary classification
    2.2 Multiclass classification
3. Regression
    2.1 Linear regression
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
import numpy as np
import time
from datetime import datetime
import itertools
from math import ceil
import os
from os import makedirs, getcwd
from os.path import isdir, join

# Machine Learning 
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict, learning_curve
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, \
                            f1_score, confusion_matrix, roc_curve, mean_absolute_error, \
                            mean_squared_error, r2_score
from sklearn.metrics import classification_report
from sklearn.exceptions import NotFittedError
import shap

# Viz
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.axes import Axes
import seaborn as sns

# For AnnotateBars class
from dataclasses import dataclass
from typing import *

# Logging
import logging


"""
---------------------------------------------------
---------------- 1. INITIAL SETUP -----------------
              1.2 Setting up logging
---------------------------------------------------
"""

# Function for a useful log configuration
def log_config(logger, level=logging.DEBUG, 
               log_format='%(levelname)s;%(asctime)s;%(filename)s;%(module)s;%(lineno)d;%(message)s',
               log_filepath=os.path.join(os.getcwd(), 'exec_log/execution_log.log'),
               flag_file_handler=False, flag_stream_handler=True, filemode='a'):
    """
    Uses a logging object for applying basic configuration on it

    Parameters
    ----------
    :param logger: logger object created on module scope [type: logging.getLogger()]
    :param level: logger object level [type: level, default=logging.DEBUG]
    :param log_format: logger format to be stored [type: string]
    :param log_filepath: path where .log file will be stored [type: string, default='log/application_log.log']
    :param flag_file_handler: flag for saving .log file [type: bool, default=False]
    :param flag_stream_handler: flag for log verbosity on cmd [type: bool, default=True]
    :param filemode: write type on the log file [type: string, default='a' (append)]

    Return
    ------
    :return logger: pre-configured logger object
    """

    # Setting level for the logger object
    logger.setLevel(level)

    # Creating a formatter
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

    # Creating handlers
    if flag_file_handler:
        log_path = '/'.join(log_filepath.split('/')[:-1])
        if not isdir(log_path):
            makedirs(log_path)

        # Adding file_handler
        file_handler = logging.FileHandler(log_filepath, mode=filemode, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if flag_stream_handler:
        # Adding stream_handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)    
        logger.addHandler(stream_handler)

    return logger

# Setting up logger object
logger = logging.getLogger(__file__)
logger = log_config(logger)


"""
---------------------------------------------------
---------------- 1. INITIAL SETUP -----------------
        1.3 Functions for formatting charts
---------------------------------------------------
"""

# Formatting spines in a matplotlib plot
def format_spines(ax, right_border=False):
    """
    Modify borders and axis colors of matplotlib figures

    Parameters
    ----------
    :param ax: figura axis created using matplotlib [type: matplotlib.pyplot.axes]
    :param right_border: boolean flag for hiding right border [type: bool, default=False]

    Return
    ------
    This functions has no return besides the customization of matplotlib axis

    Example
    -------
    fig, ax = plt.subplots()
    format_spines(ax=ax, right_border=False)
    """

    # Setting colors on the axis
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['top'].set_visible(False)

    # Right border formatting
    if right_border:
        ax.spines['right'].set_color('#CCCCCC')
    else:
        ax.spines['right'].set_color('#FFFFFF')
    ax.patch.set_facecolor('#FFFFFF')


# Reference: https://towardsdatascience.com/annotating-bar-charts-and-other-matplolib-techniques-cecb54315015
# Creating allias
#Patch = matplotlib.patches.Patch
PosVal = Tuple[float, Tuple[float, float]]
#Axis = matplotlib.axes.Axes
Axis = Axes
PosValFunc = Callable[[Patch], PosVal]

@dataclass
class AnnotateBars:
    font_size: int = 10
    color: str = "black"
    n_dec: int = 2
    def horizontal(self, ax: Axis, centered=False):
        def get_vals(p: Patch) -> PosVal:
            value = p.get_width()
            div = 2 if centered else 1
            pos = (
                p.get_x() + p.get_width() / div,
                p.get_y() + p.get_height() / 2,
            )
            return value, pos
        ha = "center" if centered else  "left"
        self._annotate(ax, get_vals, ha=ha, va="center")
    def vertical(self, ax: Axis, centered:bool=False):
        def get_vals(p: Patch) -> PosVal:
            value = p.get_height()
            div = 2 if centered else 1
            pos = (p.get_x() + p.get_width() / 2,
                   p.get_y() + p.get_height() / div
            )
            return value, pos
        va = "center" if centered else "bottom"
        self._annotate(ax, get_vals, ha="center", va=va)
    def _annotate(self, ax, func: PosValFunc, **kwargs):
        cfg = {"color": self.color,
               "fontsize": self.font_size, **kwargs}
        for p in ax.patches:
            value, pos = func(p)
            ax.annotate(f"{value:.{self.n_dec}f}", pos, **cfg)


"""
---------------------------------------------------
---------------- 1. INITIAL SETUP -----------------
         1.3 Functions for saving objects
---------------------------------------------------
"""

# Saving DataFrames on csv format
def save_data(data, output_path, filename):
    """
    Método responsável por salvar objetos DataFrame em formato csv.

    Parameters
    ----------
    :param data: data to be saved [type: pd.DataFrame]
    :param output_path: path reference for the file [type: string]
    :param filename: filename for the file with .csv extension [type: string]

    Return
    ------
    There is no return besides the file saved on local machine

    Application
    -----------
    df = file_generator_method()
    save_result(df, output_path=OUTPUT_PATH, filename='arquivo.csv')
    """

    # Searching if directory exists
    if not os.path.isdir(output_path):
        logger.warning(f'Path {output_path} not exists. Creating directory on the path')
        try:
            os.makedirs(output_path)
        except Exception as e:
            logger.error(f'Error on training to create directory {output_path}. Exception: {e}')
            return

    logger.debug(f'Saving file on directory')
    try:
        output_file = os.path.join(output_path, filename)
        data.to_csv(output_file, index=False)
    except Exception as e:
        logger.error(f'Error on saving file {filename}. Exception: {e}')

# Saving model in pkl format
def save_model(model, output_path, filename):
    """
    Saves trained model and pipelines on pkl format

    Parameter
    ---------
    :param model: model object to be saved [type: model]
    :param output_path: path reference for the file [type: string]
    :param filename: filename for the file with .csv extension [type: string]

    Return
    ------
    There is no return besides the object saved on local machine

    Application
    -----------
    model = classifiers['estimator']
    save_model(model, output_path=OUTPUT_PATH, filename='model.pkl')
    """

    # Searching if directory exists
    if not os.path.isdir(output_path):
        logger.warning(f'Path {output_path} not exists. Creating directory on the path')
        try:
            os.makedirs(output_path)
        except Exception as e:
            logger.error(f'Error on training to create directory {output_path}. Exception: {e}')
            return

    logger.debug(f'Saving pkl file on directory')
    try:
        output_file = os.path.join(output_path, filename)
        joblib.dump(model, output_file)
    except Exception as e:
        logger.error(f'Error on saving model {filename}. Exception: {e}')

# Saving figures generated from matplotlib
def save_fig(fig, output_path, img_name, tight_layout=True, dpi=300):
    """
    Saves figures created from matplotlib/seaborn

    Parameters
    ----------
    :param fig: figure created using matplotlib [type: plt.figure]
    :param output_file: path for image to be saved (path + filename in png format) [type: string]
    :param tight_layout: flag for tighting figure layout before saving it [type: bool, default=True]
    :param dpi: image resolution [type: int, default=300]

    Return
    ------
    This function returns nothing besides the image saved on the given path

    Application
    ---------
    fig, ax = plt.subplots()
    save_fig(fig, output_file='image.png')
    """

    # Searching for the existance of the directory
    if not os.path.isdir(output_path):
        logger.warning(f'Directory {output_path} not exists. Creating a directory on the given path')
        try:
            os.makedirs(output_path)
        except Exception as e:
            logger.error(f'Error on creating the directory {output_path}. Exception: {e}')
            return
    
    # Tighting layout
    if tight_layout:
        fig.tight_layout()
    
    logger.debug('Saving image on directory')
    try:
        output_file = os.path.join(output_path, img_name)
        fig.savefig(output_file, dpi=300)
        logger.info(f'Image succesfully saved in {output_file}')
    except Exception as e:
        logger.error(f'Error on saving image. Exception: {e}')
        return



"""
---------------------------------------------------
---------------- 2. CLASSIFICATION ----------------
             2.1 Binary Classification
---------------------------------------------------
"""

class BinaryClassifier:
    """
    Trains and evaluate binary classification models.
    The methods of this class enable a complete management of
    binary classification tasks in every step of the development
    workflow
    """

    def __init__(self):
        """
        Método construtor inicializa dicionário de informações dos modelos treinados
        """
        self.classifiers_info = {}      

    def fit(self, set_classifiers, X_train, y_train, **kwargs):
        """
        Trains each classifier in set_classifiers dictionary through a defined setup

        Parameters
        ----------
        :param set_classifiers: contains the setup for training the models [type: dict]
            set_classifiers = {
                'model_name': {
                    'model': __estimator__,
                    'params': __estimator_params__
                }
            }
        :param X_train: features for training data [type: np.array]
        :param y_train: target array for training data [type: np.array]
        :param **kwargs: additional parameters
            :arg approach: sufix string for identifying a different approach for models training [type: string, default='']
            :arg random_search: boolean flag for applying RandomizedSearchCV on training [type: bool, default=False]
            :arg scoring: optimization metric for RandomizedSearchCV (if applicable) [type: string, default='accuracy']
            :arg cv: K-folds used on cross validation evaluation on RandomizerSearchCV [type: int, default=5]
            :arg verbose: verbosity configured on hyperparameters search [type: int, default=-1]
            :arg n_jobs: CPUs vcores to be used during hyperparameters search [type: int, default=-1]
            :arg save: flag for saving pkl/joblib files for trained models on local disk [type: bool, default=True]
            :arg output_path: folder path for pkl/joblib files to be saved [type: string, default=cwd() + 'output/models']
            :arg model_ext: extension for model files (pkl or joblib) without point "." [type: string, default='pkl']

        Return
        ------
        This method doesn't return anything but the set of self.classifiers_info class attribute with useful info

        Application
        -----------
        # Initializing object and training models
        trainer = BinaryClassifier()
        trainer.fit(set_classifiers, X_train_prep, y_train)
        """

        # Extracting approach from kwargs dictionary
        approach = kwargs['approach'] if 'approach' in kwargs else ''

        # Iterating over each model on set_classifiers dictionary
        try:
            for model_name, model_info in set_classifiers.items():
                # Defining a custom key for the further classifiers_info class attribute dictionary
                model_key = model_name + approach
                logger.debug(f'Training model {model_key}')
                model = model_info['model']

                # Creating an empty dictionary for storing model's info
                self.classifiers_info[model_key] = {}

                # Validating the application of random search for hyperparameter tunning
                try:
                    if 'random_search' in kwargs and bool(kwargs['random_search']):
                        params = model_info['params']
                        
                        # Returning additional parameters from kwargs dictionary
                        scoring = kwargs['scoring'] if 'scoring' in kwargs else 'accuracy'
                        cv = kwargs['cv'] if 'cv' in kwargs else 5
                        verbose = kwargs['verbose'] if 'verbose' in kwargs else -1
                        n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else -1
                        
                        # Preparing and applying search
                        rnd_search = RandomizedSearchCV(model, params, scoring=scoring, cv=cv,
                                                        verbose=verbose, random_state=42, n_jobs=n_jobs)
                        logger.debug('Applying RandomizedSearchCV')
                        rnd_search.fit(X_train, y_train)

                        # Saving the best model on classifiers_info class dictionary
                        self.classifiers_info[model_key]['estimator'] = rnd_search.best_estimator_
                    else:
                        # Training model without searching for best hyperparameters
                        self.classifiers_info[model_key]['estimator'] = model.fit(X_train, y_train)
                except TypeError as te:
                    logger.error(f'Error when trying RandomizedSearch. Exception: {te}')
                    return

                # Saving pkl files if applicable
                if 'save' in kwargs and bool(kwargs['save']):
                    model_ext = kwargs['model_ext'] if 'model_ext' in kwargs else 'pkl'
                    logger.debug(f'Saving model file for {model_name} on {model_ext} format')
                    model = self.classifiers_info[model_key]['estimator']
                    output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/models')

                    anomesdia = datetime.now().strftime('%Y%m%d')
                    filename = model_name.lower() + '_' +  anomesdia + '.' + model_ext
                    save_model(model, output_path=output_path, filename=filename)

        except AttributeError as ae:
            logger.error(f'Error when training models. Exception: {ae}')

    def compute_train_performance(self, model_name, estimator, X, y, cv=5):
        """
        Applies cross validation for returning the main binary classification metrics for trained models.
        In practice, this method is usually applied on a top layer of the class, or in other words, it is usually
        executed by another method for extracting metrics on training and validating data

        Parameters
        ----------
        :param model_name: model key on self.classifiers_info class attribute [type: string]
        :param estimator: model estimator to be evaluated [type: object]
        :param X: model features for training data [type: np.array]
        :param y: target array for training data [type: np.array]
        :param cv: K-folds used on cross validation step [type: int, default=5]

        Return
        ------
        :return train_performance: dataset with metrics computed using cross validation on training set [type: pd.DataFrame]

        Application
        -----------
        # Initializing and training models
        trainer = ClassificadorBinario()
        trainer.fit(model, X_train, y_train)
        train_performance = trainer.compute_train_performance(model_name, estimator, X_train, y_train)
        """

        # Computing metrics using cross validation
        logger.debug(f'Computing metrics on {model_name} using cross validation with {cv} K-folds')
        try:
            t0 = time.time()
            accuracy = cross_val_score(estimator, X, y, cv=cv, scoring='accuracy').mean()
            precision = cross_val_score(estimator, X, y, cv=cv, scoring='precision').mean()
            recall = cross_val_score(estimator, X, y, cv=cv, scoring='recall').mean()
            f1 = cross_val_score(estimator, X, y, cv=cv, scoring='f1').mean()

            # Computing probabilities for AUC metrics
            try:
                y_scores = cross_val_predict(estimator, X, y, cv=cv, method='decision_function')
            except:
                # Tree models don't have decision_function() method but predict_proba()
                y_probas = cross_val_predict(estimator, X, y, cv=cv, method='predict_proba')
                y_scores = y_probas[:, 1]
            auc = roc_auc_score(y, y_scores)

            # Saving metrics on self.classifiers_info class attribute
            self.classifiers_info[model_name]['train_scores'] = y_scores

            # Creating a DataFrame with performance result
            t1 = time.time()
            delta_time = t1 - t0
            train_performance = {}
            train_performance['model'] = model_name
            train_performance['approach'] = f'Train {cv} K-folds'
            train_performance['acc'] = round(accuracy, 4)
            train_performance['precision'] = round(precision, 4)
            train_performance['recall'] = round(recall, 4)
            train_performance['f1'] = round(f1, 4)
            train_performance['auc'] = round(auc, 4)
            train_performance['total_time'] = round(delta_time, 3)
            logger.info(f'Sucessfully computed metrics on training data in {round(delta_time, 3)} seconds')

            return pd.DataFrame(train_performance, index=train_performance.keys()).reset_index(drop=True).loc[:0, :]

        except Exception as e:
            logger.error(f'Error on computing metrics. Exception: {e}')    

    def compute_val_performance(self, model_name, estimator, X, y):
        """
        Computes metrics on validation datasets for binary classifiers.
        In practice, this method is usually applied on a top layer of the class, or in other words, it is usually
        executed by another method for extracting metrics on training and validating data

        Parameters
        ----------
        :param model_name: model key on self.classifiers_info class attribute [type: string]
        :param estimator: model estimator to be evaluated [type: object]
        :param X: model features for validation data [type: np.array]
        :param y: target array for validation data [type: np.array]

        Return
        ------
        :return val_performance: dataset with metrics computed using cross validation on training set [type: pd.DataFrame]

        Application
        -----------
        # Initializing and training models
        trainer = BinaryClassifier()
        trainer.fit(model, X_train, y_train)
        val_performance = trainer.compute_val_performance(model_name, estimator, X_val, y_val)
        """

        # Computing metrics using cross validation
        logger.debug(f'Computing metrics on {model_name} using validation data')
        try:
            t0 = time.time()
            y_pred = estimator.predict(X)
            y_proba = estimator.predict_proba(X)
            y_scores = y_proba[:, 1]

            # Retrieving metrics using validation data
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            auc = roc_auc_score(y, y_scores)

            # Saving probabilities on treined classifiers dictionary
            self.classifiers_info[model_name]['val_scores'] = y_scores

            # Creating a DataFrame with metrics
            t1 = time.time()
            delta_time = t1 - t0
            test_performance = {}
            test_performance['model'] = model_name
            test_performance['approach'] = f'Validation'
            test_performance['acc'] = round(accuracy, 4)
            test_performance['precision'] = round(precision, 4)
            test_performance['recall'] = round(recall, 4)
            test_performance['f1'] = round(f1, 4)
            test_performance['auc'] = round(auc, 4)
            test_performance['total_time'] = round(delta_time, 3)
            logger.info(f'Sucesfully computed metrics using validation data for {model_name} on {round(delta_time, 3)} seconds')

            return pd.DataFrame(test_performance, index=test_performance.keys()).reset_index(drop=True).loc[:0, :]

        except Exception as e:
            logger.error(f'Error on computing metrics. Exception: {e}')

    def evaluate_performance(self, X_train, y_train, X_val, y_val, cv=5, **kwargs):
        """
        Executes and computes classification metrics for training and validation data
        
        Parameters
        ----------
        :param X_train: model features for training data [type: np.array]
        :param y_train: target array for training data [type: np.array]
        :param X_val: model features for validation data [type: np.array]
        :param y_val: target array for validation data [type: np.array]
        :param cv: K-folds used on cross validation step [type: int, default=5]
        :param **kwargs: additional parameters
            :arg save: boolean flag for saving files on locak disk [type: bool, default=True]
            :arg output_path: path for files to be saved [type: string, default=cwd() + 'output/metrics']
            :arg output_filename: name of csv file to be saved [type: string, default='metrics.csv']
        
        Return
        ------
        :return df_performances: dataset with metrics obtained using training and validation data [type: pd.DataFrame]

        Application
        -----------
        # Training model and evaluating performance on training and validation sets
        trainer = BinaryClassifier()
        trainer.fit(estimator, X_train, y_train)

        # Generating a performance dataset
        df_performance = trainer.evaluate_performance(X_train, y_train, X_val, y_val)
        """

        # Creating an empty DataFrame for storing metrics
        df_performances = pd.DataFrame({})

        # Iterating over the trained classifiers on the class attribute dictionary
        for model_name, model_info in self.classifiers_info.items():

            # Verifying if the model was already trained (model_info dict will have the key 'train_performance')
            if 'train_performance' in model_info.keys():
                df_performances = df_performances.append(model_info['train_performance'])
                df_performances = df_performances.append(model_info['val_performance'])
                continue

            # Returning the model to be evaluated
            try:
                estimator = model_info['estimator']
            except KeyError as e:
                logger.error(f'Error on returning the key "estimator" from model_info dict. Model {model_name} was not trained')
                continue

            # Computing performance on training and validation sets
            train_performance = self.compute_train_performance(model_name, estimator, X_train, y_train, cv=cv)
            val_performance = self.compute_val_performance(model_name, estimator, X_val, y_val)

            # Setting up results on classifiers_info class dict
            self.classifiers_info[model_name]['train_performance'] = train_performance
            self.classifiers_info[model_name]['val_performance'] = val_performance

            # Building a DataFrame with model metrics
            model_performance = train_performance.append(val_performance)
            df_performances = df_performances.append(model_performance)
            df_performances['anomesdia_datetime'] = datetime.now()

            # Saving some attributes on classifiers_info for maybe retrieving in the future
            model_data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val
            }
            model_info['model_data'] = model_data

        # Saving results if applicable
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/metrics')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'metrics.csv'
            save_data(df_performances, output_path=output_path, filename=output_filename)

        return df_performances.reset_index(drop=True)

    def feature_importance(self, features, top_n=-1, **kwargs):
        """
        Extracts the feature importance method from trained models
        
        Parameters
        ----------
        :param features: list of features considered on training step [type: list]
        :param top_n: parameter for filtering just top N features most important [type: int, default=-1]
            *obs: when this parameter is equal to -1, all features are considered
        :param **kwargs: additional parameters
            :arg save: boolean flag for saving files on locak disk [type: bool, default=True]
            :arg output_path: path for files to be saved [type: string, default=cwd() + 'output/metrics']
            :arg output_filename: name of csv file to be saved [type: string, default='top_features.csv']

        Return
        ------
        :return: all_feat_imp: pandas DataFrame com a análise de feature importance dos modelos [type: pd.DataFrame]

        Application
        -----------
        # Training models
        trainer = BinaryClassifier()
        trainer.fit(estimator, X_train, y_train)

        # Returning a feature importance dataset for all models at once
        feat_imp = trainer.feature_importance(features=MODEL_FEATURES, top_n=20)
        """

        # Creating an empty DataFrame for storing feature importance analysis
        feat_imp = pd.DataFrame({})
        all_feat_imp = pd.DataFrame({})

        # Iterating over models in the class
        for model_name, model_info in self.classifiers_info.items():
            
            # Extracting feature importance from models
            logger.debug(f'Extracting feature importances from the model {model_name}')
            try:
                importances = model_info['estimator'].feature_importances_
            except KeyError as ke:
                logger.warning(f'Model {model_name} was not trained yet, so it is impossible use the method feature_importances_')
                continue
            except AttributeError as ae:
                logger.warning(f'Model {model_name} do not have feature_importances_ method')
                continue

            # Preparing dataset for storing the info
            feat_imp['feature'] = features
            feat_imp['importance'] = importances
            feat_imp['model'] = model_name
            feat_imp['anomesdia_datetime'] = datetime.now()
            feat_imp.sort_values(by='importance', ascending=False, inplace=True)
            feat_imp = feat_imp.loc[:, ['model', 'feature', 'importance', 'anomesdia_datetime']]

            # Saving feature importance info on class attribute dictionary classifiers_info
            self.classifiers_info[model_name]['feature_importances'] = feat_imp
            all_feat_imp = all_feat_imp.append(feat_imp)
            logger.info(f'Feature importances extracted succesfully for the model {model_name}')

        # Saving results if applicable
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/metrics')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'top_features.csv'
            save_data(all_feat_imp, output_path=output_path, filename=output_filename)
        
        return all_feat_imp

    def training_flow(self, set_classifiers, X_train, y_train, X_val, y_val, features, **kwargs):
        """
        This method consolidates all the steps needed for trainign, evaluating and extracting useful
        information for machine learning models given specific input arguments. When executed, this
        method sequencially applies the fit(), evaluate_performance() and feature_importance() methods
        of this given class, saving results if applicable.
        
        This is a good choice for doing all the things at once. The tradeoff is that it's important to
        input a set of parameters needed for all individual methods.

        Parameters
        ----------
        :param set_classifiers: contains the setup for training the models [type: dict]
            set_classifiers = {
                'model_name': {
                    'model': __estimator__,
                    'params': __estimator_params__
                }
            }
        :param X_train: features for training data [type: np.array]
        :param y_train: target array for training data [type: np.array]
        :param X_val: model features for validation data [type: np.array]
        :param y_val: target array for validation data [type: np.array]
        :param features: list of features considered on training step [type: list]
        :param **kwargs: additional parameters
            :arg approach: sufix string for identifying a different approach for models training [type: string, default='']
            :arg random_search: boolean flag for applying RandomizedSearchCV on training [type: bool, default=False]
            :arg scoring: optimization metric for RandomizedSearchCV (if applicable) [type: string, default='accuracy']
            :arg cv: K-folds used on cross validation evaluation [type: int, default=5]
            :arg verbose: verbosity configured on hyperparameters search [type: int, default=-1]
            :arg n_jobs: CPUs vcores to be used during hyperparameters search [type: int, default=-1]
            :arg save: boolean flag for saving files on locak disk [type: bool, default=True]
            :arg models_output_path: path for saving model pkl files [type: string, default=cwd() + 'output/models']
            :arg metrics_output_path: path for saving performance metrics dataset [type: string, default=cwd() + 'output/metrics']
            :arg metrics_output_filename: filename for metrics dataset csv file to be saved [type: string, default='metrics.csv']
            :arg featimp_output_filename: filename for feature importance csv file to be saved [type: string, default='top_features.csv']
            :arg top_n_featimp: :param top_n: parameter for filtering just top N features most important [type: int, default=-1]
                *obs: when this parameter is equal to -1, all features are considered

        Return
        ------
        This method don't return anything but the complete training and evaluating flow

        Application
        -----------
        # Initializing object and executing training steps through the method
        trainer = ClassificadorBinario()
        trainer.training_flow(set_classifiers, X_train, y_train, X_val, y_val, features)
        """

        # Extracting additional parameters from kwargs dictionary
        approach = kwargs['approach'] if 'approach' in kwargs else ''
        random_search = kwargs['random_search'] if 'random_search' in kwargs else False
        scoring = kwargs['scoring'] if 'scoring' in kwargs else 'accuracy'
        cv = kwargs['cv'] if 'cv' in kwargs else 5
        verbose = kwargs['verbose'] if 'verbose' in kwargs else -1
        n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else -1
        save = bool(kwargs['save']) if 'save' in kwargs else True
        models_output_path = kwargs['models_output_path'] if 'models_output_path' in kwargs else os.path.join(os.getcwd(), 'output/models')
        metrics_output_path = kwargs['metrics_output_path'] if 'metrics_output_path' in kwargs else os.path.join(os.getcwd(), 'output/metrics')
        metrics_output_filename = kwargs['metrics_output_filename'] if 'metrics_output_filename' in kwargs else 'metrics.csv'
        featimp_output_filename = kwargs['featimp_output_filename'] if 'featimp_output_filename' in kwargs else 'top_features.csv'
        top_n_featimp = kwargs['top_n_featimp'] if 'top_n_featimp' in kwargs else -1

        # Training models
        self.fit(set_classifiers, X_train, y_train, approach=approach, random_search=random_search, scoring=scoring,
                 cv=cv, verbose=verbose, n_jobs=n_jobs, save=save, output_path=models_output_path)

        # Evaluating models
        self.evaluate_performance(X_train, y_train, X_val, y_val, save=save, output_path=metrics_output_path, 
                                  output_filename=metrics_output_filename)

        # Extracting feature importance from models
        self.feature_importance(features, top_n=top_n_featimp, save=save, output_path=metrics_output_path, 
                                output_filename=featimp_output_filename)

    def plot_metrics(self, figsize=(16, 10), palette='rainbow', cv=5, **kwargs):
        """
        Plots metrics results for all trained models using training and validation data

        Parameter
        ---------
        :param figsize: figure size [type: tuple, default=(16, 10)]
        :param palette: matplotlib colormap for the chart [type: string, default='rainbow']
        :param cv: K-folds used on cross validation evaluation [type: int, default=5]
        :param **kwargs: additional parameters
            :arg save: boolean flag for saving files on locak disk [type: bool, default=True]
            :arg output_path: path for files to be saved [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: name of csv file to be saved [type: string, default='metrics_comparison.png']

        Return
        ------
        This method don't return anything but the custom metrics chart

        Application
        -----------
        # Training models
        trainer = BinaryClassifier()
        trainer.fit(estimator, X_train, y_train)

        # Visualizing performance through a custom chart
        trainer.plot_metrics()
        """

        # Initializing plot
        logger.debug(f'Initializing plot for visual evaluation of classifiers')
        metrics = pd.DataFrame()
        for model_name, model_info in self.classifiers_info.items():
            
            logger.debug(f'Returning metrics through cross validation for {model_name}')
            try:
                # Returning classifier variables from classifiers_info class dict attribute
                metrics_tmp = pd.DataFrame()
                estimator = model_info['estimator']
                X_train = model_info['model_data']['X_train']
                y_train = model_info['model_data']['y_train']

                # Computing metrics using cross validation
                accuracy = cross_val_score(estimator, X_train, y_train, cv=cv, scoring='accuracy')
                precision = cross_val_score(estimator, X_train, y_train, cv=cv, scoring='precision')
                recall = cross_val_score(estimator, X_train, y_train, cv=cv, scoring='recall')
                f1 = cross_val_score(estimator, X_train, y_train, cv=cv, scoring='f1')

                # Adding up into the empty DataFrame metrics
                metrics_tmp['accuracy'] = accuracy
                metrics_tmp['precision'] = precision
                metrics_tmp['recall'] = recall
                metrics_tmp['f1'] = f1
                metrics_tmp['model'] = model_name

                # Appending metrics for each model
                metrics = metrics.append(metrics_tmp)
            except Exception as e:
                logger.warning(f'Error on returning metrics for {model_name}. Exception: {e}')
                continue
        
        logger.debug(f'Transforming metrics DataFrame for applying a visual plot')
        try:
            # Pivotting metrics (boxplot)
            index_cols = ['model']
            metrics_cols = ['accuracy', 'precision', 'recall', 'f1']
            df_metrics = pd.melt(metrics, id_vars=index_cols, value_vars=metrics_cols)

            # Grouping metrics (bars)
            metrics_group = df_metrics.groupby(by=['model', 'variable'], as_index=False).mean()
        except Exception as e:
            logger.error(f'Error on trying to pivot the DataFrame. Exception: {e}')
            return

        logger.debug(f'Visualizing metrics for trained models')
        try:
            # Plotting charts
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=figsize)
            sns.boxplot(x='variable', y='value', data=df_metrics.sort_values(by='model'), hue='model', ax=axs[0], palette=palette)
            sns.barplot(x='variable', y='value', data=metrics_group, hue='model', ax=axs[1], palette=palette, order=metrics_cols)

            # Customizing axis
            axs[0].set_title(f'Metrics distribution using cross validation on training data with {cv} K-folds', size=14, pad=15)
            axs[1].set_title(f'Average of each metric obtained on cross validation', size=14, pad=15)
            format_spines(axs[0], right_border=False)
            format_spines(axs[1], right_border=False)
            axs[1].get_legend().set_visible(False)
            AnnotateBars(n_dec=3, color='black', font_size=12).vertical(axs[1])
        except Exception as e:
            logger.error(f'Error when plotting charts for metrics. Exception: {e}')
            return

        # Tighting layout
        plt.tight_layout()

        # Saving figure if applicable
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'metrics_comparison.png'
            save_fig(fig, output_path=output_path, img_name=output_filename)      

    def plot_feature_importance(self, features, top_n=20, palette='viridis', **kwargs):
        """
        Plots a chart for visualizing features most important for each trained model on the class

        Parameters
        ----------
        :param features: list of features considered on training step [type: list]
        :param top_n: parameter for filtering just top N features most important [type: int, default=20]
            *obs: when this parameter is equal to -1, all features are considered
        :param palette: matplotlib colormap for the chart [type: string, default='viridis']
        :param **kwargs: additional parameters
            :arg save: boolean flag for saving files on locak disk [type: bool, default=True]
            :arg output_path: path for files to be saved [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: name of png file to be saved [type: string, default='feature_importances.png']

        Return
        ------
        This method don't return anything but the custom chart for feature importances analysis

        Application
        -----------
        # Training models
        trainer = BinaryClassifier()
        trainer.fit(estimator, X_train, y_train)

        # Visualizing performance through a custom chart
        trainer.plot_feature_importance()
        """

        # Definindo parâmetros de plotagem
        logger.debug('Inicializando plotagem das features mais importantes para os modelos')
        feat_imp = pd.DataFrame({})
        i = 0
        ax_del = 0
        nrows = len(self.classifiers_info.keys())
        fig, axs = plt.subplots(nrows=nrows, figsize=(16, nrows * 6))
        sns.set(style='white', palette='muted', color_codes=True)
        
        # Iterando sobre os modelos presentes na classe
        for model_name, model_info in self.classifiers_info.items():
            # Validando possibilidade de extrair a importância das features do modelo
            logger.debug(f'Extraindo importância das features para o modelo {model_name}')
            try:
                importances = model_info['estimator'].feature_importances_
            except:
                logger.warning(f'Modelo {model_name} não possui o método feature_importances_')
                ax_del += 1
                continue
            
            # Preparando o dataset para armazenamento das informações
            feat_imp['feature'] = features
            feat_imp['importance'] = importances
            feat_imp.sort_values(by='importance', ascending=False, inplace=True)

            logger.debug(f'Plotando gráfico de importância das features para o modelo {model_name}')
            try:
                # Plotando feature importance
                sns.barplot(x='importance', y='feature', data=feat_imp.iloc[:top_n, :], ax=axs[i], palette=palette)

                # Customizando gráfico
                axs[i].set_title(f'Features Mais Importantes: {model_name}', size=14)
                format_spines(axs[i], right_border=False)
                i += 1
  
                logger.info(f'Gráfico de importância das features plotado com sucesso para o modelo {model_name}')
            except Exception as e:
                logger.error(f'Erro ao gerar gráfico de importância das features para o modelo {model_name}. Exception lançada: {e}')
                continue

        # Deletando eixos sobressalentes (se aplicável)
        if ax_del > 0:
            logger.debug('Deletando eixos referentes a análises não realizadas')
            try:
                for i in range(-1, -(ax_del+1), -1):
                    fig.delaxes(axs[i])
            except Exception as e:
                logger.error(f'Erro ao deletar eixo. Exception lançada: {e}')
        
        # Alinhando figura
        plt.tight_layout()

        # Salvando imagem
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'feature_importances.png'
            save_fig(fig, output_path=output_path, img_name=output_filename)   

    def custom_confusion_matrix(self, model_name, y_true, y_pred, classes, cmap, normalize=False):
        """
        Plots a custom confusion matrix for only one model. In practive this method is called in a top layer
        through another method that iterates over all trained models on the class. This was a good way for
        keep the organization on the class by centralizing all confusion matrix char modifications in
        one specific method

        Parameters
        ----------
        :param model_name: model key on self.classifiers_info class attribute [type: string]
        :param y_true: target array for source data [type: np.array]
        :param y_pred: predictions array generated by a predict method [type: np.array]
        :param classes: name for classes to be put on the matrix [type: list]
        :param cmap: matplotlib colormap for the matrix chart [type: matplotlib.colormap]
        :param normalize: flag for normalizing cells on the matrix [type: bool, default=False]

        Return
        -------
        This method don't return anything but the customization of confusion matrix

        Application
        -----------
        This method is not usually executed by users outside the class.
        Please take a look at the self.plot_confusion_matrix() method.
        """

        # Returning confusion matrix through the sklearn's function
        conf_mx = confusion_matrix(y_true, y_pred)

        # Plotting the matrix
        plt.imshow(conf_mx, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        tick_marks = np.arange(len(classes))

        # Customizing axis
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Customizing entries
        fmt = '.2f' if normalize else 'd'
        thresh = conf_mx.max() / 2.
        for i, j in itertools.product(range(conf_mx.shape[0]), range(conf_mx.shape[1])):
            plt.text(j, i, format(conf_mx[i, j]),
                     horizontalalignment='center',
                     color='white' if conf_mx[i, j] > thresh else 'black')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'{model_name}\nConfusion Matrix', size=12)
    
    def plot_confusion_matrix(self, cmap=plt.cm.Blues, normalize=False, **kwargs):
        """
        Iterates over the dictionary of trained models and builds a custom conf matrix for each one
        using training and validation data

        Parameters
        ----------
        :param cmap: matplotlib colormap for the matrix chart [type: matplotlib.colormap, default=]
        :param normalize: flag for normalizing cells on the matrix [type: bool, default=False]
        :param **kwargs: additional parameters
            :arg save: boolean flag for saving files on locak disk [type: bool, default=True]
            :arg output_path: path for files to be saved [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: name of png file to be saved [type: string, default='confusion_matrix.png']
        
        Return
        ------
        This method don't return anything but the plot of custom confusion matrix for trained models

        Application
        -----------
        trainer = BinaryClassifier()
        trainer.training_flow(set_classifiers, X_train, y_train, X_val, y_val, features)
        trainer.plot_confusion_matrix(output_path=OUTPUT_PATH)
        """

        # Setting up parameters
        logger.debug('Initializing confusion matrix plotting for the models')
        k = 1
        nrows = len(self.classifiers_info.keys())
        fig = plt.figure(figsize=(10, nrows * 4))
        sns.set(style='white', palette='muted', color_codes=True)

        # Iterating over each trained model on classifiers_info class attribute
        for model_name, model_info in self.classifiers_info.items():
            logger.debug(f'Returning training and validation data for {model_name}')
            try:
                # Returning data for the model
                X_train = model_info['model_data']['X_train']
                y_train = model_info['model_data']['y_train']
                X_val = model_info['model_data']['X_val']
                y_val = model_info['model_data']['y_val']
                classes = np.unique(y_train)
            except Exception as e:
                logger.error(f'Error when returning data already saved for {model_name}. Exception: {e}')
                continue

            # Making predictions for training (cross validation) and validation data
            logger.debug(f'Making predictions on training and validation data for {model_name}')
            try:
                train_pred = cross_val_predict(model_info['estimator'], X_train, y_train, cv=5)
                val_pred = model_info['estimator'].predict(X_val)
            except Exception as e:
                logger.error(f'Error on making predictions for {model_name}. Exception: {e}')
                continue

            logger.debug(f'Creating a confusion matrix for {model_name}')
            try:
                # Plotting the matrix using training data
                plt.subplot(nrows, 2, k)
                self.custom_confusion_matrix(model_name + ' Train', y_train, train_pred, classes=classes, 
                                             cmap=cmap, normalize=normalize)
                k += 1

                # Plotting the matrix using validation data
                plt.subplot(nrows, 2, k)
                self.custom_confusion_matrix(model_name + ' Validation', y_val, val_pred, classes=classes, 
                                             cmap=plt.cm.Greens, normalize=normalize)
                k += 1
                logger.info(f'Confusion matrix succesfully plotted for {model_name}')
            except Exception as e:
                logger.error(f'Error when generating confusion matrix for {model_name}. Exception: {e}')
                continue

        # Tighting layout
        plt.tight_layout()

        # Saving image if applicable
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'confusion_matrix.png'
            save_fig(fig, output_path=output_path, img_name=output_filename)   

    def plot_roc_curve(self, figsize=(16, 6), **kwargs):
        """
        Plots a custom ROC Curve for each trained model on dictionary class attribute
        for training and validation data

        Parameters
        ----------
        :param figsize: figure size [type: tuple, default=(16, 6)]
        :param **kwargs: additional parameters
            :arg save: boolean flag for saving files on locak disk [type: bool, default=True]
            :arg output_path: path for files to be saved [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: name of png file to be saved [type: string, default='roc_curve.png']

        Return
        ------
        This method don't return anything but a custom chart for the ROC Curve

        Application
        -----------
        trainer = BinaryClassifier()
        trainer.training_flow(set_classifiers, X_train, y_train, X_val, y_val, features)
        trainer.plot_roc_curve(output_path=OUTPUT_PATH)
        """

        # Creating figure
        logger.debug('Initializing a ROC Curve analysis for trained models')
        fig, axs = plt.subplots(ncols=2, figsize=figsize)

        # Iterating over trained models on class attribute
        for model_name, model_info in self.classifiers_info.items():

            logger.debug(f'Returning labels and training and validation scores for {model_name}')
            try:
                # Returning labels for training and validation
                y_train = model_info['model_data']['y_train']
                y_val = model_info['model_data']['y_val']

                # Returning scores already computed on performance evaluation method
                train_scores = model_info['train_scores']
                val_scores = model_info['val_scores']
            except Exception as e:
                logger.error(f'Error on returning parameters for {model_name}. Exception: {e}')
                continue

            logger.debug(f'Computing FPR, TPR and AUC on training and validation for {model_name}')
            try:
                # Computing false positive rate and true positive rate
                train_fpr, train_tpr, train_thresholds = roc_curve(y_train, train_scores)
                test_fpr, test_tpr, test_thresholds = roc_curve(y_val, val_scores)

                # Returning AUC already computed on performance evaluation method
                train_auc = model_info['train_performance']['auc'].values[0]
                test_auc = model_info['val_performance']['auc'].values[0]
            except Exception as e:
                logger.error(f'Error when computing parameters for {model_name}. Exception: {e}')
                continue

            logger.debug(f'Plotting the ROC Curves for {model_name}')
            try:
                # Plotting ROC Curve (training)
                plt.subplot(1, 2, 1)
                plt.plot(train_fpr, train_tpr, linewidth=2, label=f'{model_name} auc={train_auc}')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.axis([-0.02, 1.02, -0.02, 1.02])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - Train Data')
                plt.legend()

                # Plotting ROC Curve (training)
                plt.subplot(1, 2, 2)
                plt.plot(test_fpr, test_tpr, linewidth=2, label=f'{model_name} auc={test_auc}')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.axis([-0.02, 1.02, -0.02, 1.02])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - Validation Data', size=12)
                plt.legend()
            except Exception as e:
                logger.error(f'Error on plotting ROC Curve for {model_name}. Exception: {e}')
                continue

        # Tighting laout
        plt.tight_layout()

        # Saving image if applicable
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'roc_curve.png'
            save_fig(fig, output_path=output_path, img_name=output_filename)   
    
    def plot_score_distribution(self, shade=True, **kwargs):
        """
        Plots useful charts for analysing the score distribution of a model through a kdeplot.
        When executed, this method builds up two charts: one for training and another for validation
        where each one is given by two curves for each target class
        
        Parameters
        ----------
        :param shade: flag for filling down the area under the distribution curve [type: bool, default=True]
        :param **kwargs: additional parameters
            :arg save: boolean flag for saving files on locak disk [type: bool, default=True]
            :arg output_path: path for files to be saved [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: name of png file to be saved [type: string, default='score_distribution.png']

        Return
        ------
        This method don't return anything but the score distribution plot
        
        Application
        -----------
        trainer = BinaryClassifier()
        trainer.training_flow(set_classifiers, X_train, y_train, X_val, y_val, features)
        trainer.plot_score_distribution(output_path=OUTPUT_PATH)
        """

        # Creating figure
        logger.debug('Initializing distribution score analysis for the models')
        i = 0
        nrows = len(self.classifiers_info.keys())
        fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(16, nrows * 4))
        sns.set(style='white', palette='muted', color_codes=True)

        # Iterating over trained classifiers on the class attribute
        for model_name, model_info in self.classifiers_info.items():

            logger.debug(f'Returning labels and trainind and validation score for {model_name}')
            try:
                # Returning training and validation target labels label de treino e de teste
                y_train = model_info['model_data']['y_train']
                y_val = model_info['model_data']['y_val']

                # Returning scores that were already computed on evaluate_performance() method
                train_scores = model_info['train_scores']
                test_scores = model_info['val_scores']
            except Exception as e:
                logger.error(f'Error on returning parameters for {model_name}. Exception: {e}')
                continue

            logger.debug(f'Plotting the score distribution chart for {model_name}')
            try:
                # Building distribution chart for training data
                sns.kdeplot(train_scores[y_train == 1], ax=axs[i, 0], label='y=1', shade=shade, color='crimson')
                sns.kdeplot(train_scores[y_train == 0], ax=axs[i, 0], label='y=0', shade=shade, color='darkslateblue')
                axs[i, 0].set_title(f'Score Distribution for {model_name} - Training')
                axs[i, 0].legend()
                axs[i, 0].set_xlabel('Score')
                format_spines(axs[i, 0], right_border=False)

                # Building distribution chart for validation data
                sns.kdeplot(test_scores[y_val == 1], ax=axs[i, 1], label='y=1', shade=shade, color='crimson')
                sns.kdeplot(test_scores[y_val == 0], ax=axs[i, 1], label='y=0', shade=shade, color='darkslateblue')
                axs[i, 1].set_title(f'Score Distribution for {model_name} - Validation')
                axs[i, 1].legend()
                axs[i, 1].set_xlabel('Score')
                format_spines(axs[i, 1], right_border=False)
                i += 1
            except Exception as e:
                logger.error(f'Error on returning curve for {model_name}. Exception: {e}')
                continue

        # Tighting layout
        plt.tight_layout()

        # Saving image if applicable
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'score_distribution.png'
            save_fig(fig, output_path=output_path, img_name=output_filename)   

    def plot_score_bins(self, bin_range=.20, **kwargs):
        """
        Plots a distribution score analysis splitted on categorical bins.

        Parameters
        ----------
        :param bin_range: range for score bins [type: float, default=.20]
        :param **kwargs: additional parameters
            :arg save: boolean flag for saving files on locak disk [type: bool, default=True]
            :arg output_path: path for files to be saved [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: name of png file to be saved [type: string, default='score_bins.png']

        Return
        ------
        This method don't return anything but a custom chart or visualizing scores at different bins
        
        Application
        -----------
        trainer = BinaryClassifier()
        trainer.training_flow(set_classifiers, X_train, y_train, X_val, y_val, features)
        trainer.plot_score_distribution(output_path=OUTPUT_PATH)
        """

        logger.debug('Initializing score analysis on categorical bins for trained models')
        i = 0
        nrows = len(self.classifiers_info.keys())
        fig1, axs1 = plt.subplots(nrows=nrows, ncols=2, figsize=(16, nrows * 4))
        fig2, axs2 = plt.subplots(nrows=nrows, ncols=2, figsize=(16, nrows * 4))

        # Creating a list of bins
        bins = np.arange(0, 1.01, bin_range)
        bins_labels = [str(round(list(bins)[i - 1], 2)) + ' a ' + str(round(list(bins)[i], 2)) for i in range(len(bins)) if i > 0]

        # Iterating ver trained models on class attribute
        for model_name, model_info in self.classifiers_info.items():

            logger.debug(f'Returning parameters for {model_name}')
            try:
                # Retrieving the train scores and creating a DataFrame
                train_scores = model_info['train_scores']
                y_train = model_info['model_data']['y_train']
                df_train_scores = pd.DataFrame({})
                df_train_scores['scores'] = train_scores
                df_train_scores['target'] = y_train
                df_train_scores['faixa'] = pd.cut(train_scores, bins, labels=bins_labels)

                # Computing the distribution for each bin
                df_train_rate = pd.crosstab(df_train_scores['faixa'], df_train_scores['target'])
                df_train_percent = df_train_rate.div(df_train_rate.sum(1).astype(float), axis=0)

                # Retrieving val scores and creating a DataFrame
                val_scores = model_info['val_scores']
                y_val = model_info['model_data']['y_val']
                df_val_scores = pd.DataFrame({})
                df_val_scores['scores'] = val_scores
                df_val_scores['target'] = y_val
                df_val_scores['faixa'] = pd.cut(val_scores, bins, labels=bins_labels)

                # Computing the distribution for each bin
                df_val_rate = pd.crosstab(df_val_scores['faixa'], df_val_scores['target'])
                df_val_percent = df_val_rate.div(df_val_rate.sum(1).astype(float), axis=0)
            except Exception as e:
                logger.error(f'Error on returning and computing parameters for {model_name}. Exception: {e}')
                continue

            logger.debug(f'Plotting score distribution on bins for {model_name}')
            try:
                sns.countplot(x='faixa', data=df_train_scores, ax=axs1[i, 0], hue='target', palette=['darkslateblue', 'crimson'])
                sns.countplot(x='faixa', data=df_val_scores, ax=axs1[i, 1], hue='target', palette=['darkslateblue', 'crimson'])

                # Formatting legend and titles
                axs1[i, 0].legend(loc='upper right')
                axs1[i, 1].legend(loc='upper right')
                axs1[i, 0].set_title(f'Score Distribution on Bins for {model_name} - Training', size=14)
                axs1[i, 1].set_title(f'Score Distribution on Bins for {model_name} - Validation', size=14)

                # Adding up data labels
                AnnotateBars(n_dec=0, color='black', font_size=12).vertical(axs1[i, 0])
                AnnotateBars(n_dec=0, color='black', font_size=12).vertical(axs1[i, 1])

                # Formatting axis
                format_spines(axs1[i, 0], right_border=False)
                format_spines(axs1[i, 1], right_border=False)

                logger.debug(f'Plotting percentual analysis on bins for {model_name}')
                for df_percent, ax in zip([df_train_percent, df_val_percent], [axs2[i, 0], axs2[i, 1]]):
                    df_percent.plot(kind='bar', ax=ax, stacked=True, color=['darkslateblue', 'crimson'], width=0.6)

                    for p in ax.patches:
                        # Colecting parameters for adding data labels
                        height = p.get_height()
                        width = p.get_width()
                        x = p.get_x()
                        y = p.get_y()

                        # Formatting parameters
                        label_text = f'{round(100 * height, 1)}%'
                        label_x = x + width - 0.30
                        label_y = y + height / 2
                        ax.text(label_x, label_y, label_text, ha='center', va='center', color='white',
                                fontweight='bold', size=10)
                    format_spines(ax, right_border=False)

                    # Formatting legend and title
                    axs2[i, 0].set_title(f'Score Distribution on Bins (Percent) for {model_name} - Training', size=14)
                    axs2[i, 1].set_title(f'Score Distribution on Bins (Percent) for {model_name} - Validation', size=14)
                i += 1

            except Exception as e:
                logger.error(f'Error on plotting score distribution on bins for {model_name}. Exception: {e}')
                continue

        # Tighting layout
        fig1.tight_layout()
        fig2.tight_layout()

        # Saving image
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            save_fig(fig1, output_path, img_name='score_bins.png')
            save_fig(fig2, output_path, img_name='score_bins_percent.png')        

    def plot_learning_curve(self, ylim=None, cv=5, n_jobs=3, train_sizes=np.linspace(.1, 1.0, 10), **kwargs):
        """
        Plots an excellent chart for analysing a learning curve for trained models.
        
        Parameters
        ----------
        :param ylim: vertical axis limit [type: int, default=None]
        :param cv: K-folds used on cross validation [type: int, default=5]
        :param n_jobs: CPUs vcores for processing [type: int, default=3]
        :param train_sizes: array with steps for measuring performance [type: np.array, default=np.linspace(.1, 1.0, 10)]
        :param **kwargs: additional parameters
            :arg save: boolean flag for saving files on locak disk [type: bool, default=True]
            :arg output_path: path for files to be saved [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: name of png file to be saved [type: string, default='learning_curve.png']

        Return
        ------
        This method don't return anything but the learning curve chart

        Application
        -----------
        trainer = BinaryClassifier()
        trainer.training_flow(set_classifiers, X_train, y_train, X_val, y_val, features)
        trainer.plot_learning_curve()
        """

        logger.debug(f'Initializing plots for learning curves for trained models')
        i = 0
        nrows = len(self.classifiers_info.keys())
        fig, axs = plt.subplots(nrows=nrows, figsize=(16, nrows * 6))

        # Iterating over each model in class attribute
        for model_name, model_info in self.classifiers_info.items():
            ax = axs[i]
            logger.debug(f'Returning parameters for {model_name} and applying learning_curve method')
            try:
                model = model_info['estimator']
                X_train = model_info['model_data']['X_train']
                y_train = model_info['model_data']['y_train']

                # Calling learning_curve function for returning scores for training and validation
                train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

                # Computando médias e desvio padrão (treino e validação)
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                val_scores_mean = np.mean(val_scores, axis=1)
                val_scores_std = np.std(val_scores, axis=1)
            except Exception as e:
                logger.error(f'Error on returning parameters and applying learning curve for {model_name}. Exception: {e}')
                continue

            logger.debug(f'Plotting learning curves for training and validation data for {model_name}')
            try:
                # Results on training data
                ax.plot(train_sizes, train_scores_mean, 'o-', color='navy', label='Training Score')
                ax.fill_between(train_sizes, (train_scores_mean - train_scores_std), (train_scores_mean + train_scores_std),
                                alpha=0.1, color='blue')

                # Results on validation data
                ax.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Cross Val Score')
                ax.fill_between(train_sizes, (val_scores_mean - val_scores_std), (val_scores_mean + val_scores_std),
                                alpha=0.1, color='crimson')

                # Customizando plotagem
                ax.set_title(f'Model {model_name} - Learning Curve', size=14)
                ax.set_xlabel('Training size (m)')
                ax.set_ylabel('Score')
                ax.grid(True)
                ax.legend(loc='best')
            except Exception as e:
                logger.error(f'Error on plotting learning curve for {model_name}. Exception: {e}')
                continue
            i += 1
        
        # Tighting layout
        plt.tight_layout()

        # Saving image
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'learning_curve.png'
            save_fig(fig, output_path=output_path, img_name=output_filename) 

    def plot_shap_analysis(self, model_name, features, figsize=(16, 10), **kwargs):
        """
        Plots an useful shap analysis for interpreting a specific model
        
        Parameters
        ----------
        :param model_name: a key for extracting an estimator from classifier info dict class attribute [type: string]
        :param features: list of features used on training the model [type: list]
        :param figsize: figure size [type: tuple, default=(16, 10)]
        :param **kwargs: additional parameters
            :arg save: boolean flag for saving files on locak disk [type: bool, default=True]
            :arg output_path: path for files to be saved [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: name of png file to be saved [type: string, default='learning_curve.png']

        Return
        ------
        This method don't return anything but the plot of shap analysis (violin)

        Application
        -----------
        trainer = BinaryClassifier()
        trainer.training_flow(set_classifiers, X_train, y_train, X_val, y_val, features)
        trainer.plot_shap_analysis(model_name='LightGBM', features=MODEL_FEATURES)
        """

        logger.debug(f'Explaining {model_name} through a violin plot on shap analysis')
        try:
            model_info = self.classifiers_info[model_name]
            model = model_info['estimator']
        except Exception as e:
            logger.error(f'Model key {model_name} not exists or model was not trained. Available options: {list(self.classifiers_info.keys())}')
            return

        logger.debug(f'Returning paramteres for {model_name}')
        try:
            # Returning model parameters
            X_train = model_info['model_data']['X_train']
            X_val = model_info['model_data']['X_val']
            df_train = pd.DataFrame(X_train, columns=features)
            df_val = pd.DataFrame(X_val, columns=features)
        except Exception as e:
            logger.error(f'Error on returning parameters for {model_name}. Exception: {e}')

        logger.debug(f'Creating a explainer and generating shap values for {model_name}')
        try:
            explainer = shap.TreeExplainer(model, df_train)
            shap_values = explainer.shap_values(df_val)
        except Exception as e:
            try:
                logger.warning(f'TreeExplainer does not fit on {model_name}. Trying LinearExplainer')
                explainer = shap.LinearExplainer(model, df_train)
                shap_values = explainer.shap_values(df_val, check_additivity=False)
            except Exception as e:
                logger.error(f'Error on returning parameters for {model_name}. Exception: {e}')
                return

        logger.debug(f'Making a shap analysis violin plot for {model_name}')
        try:
            fig, ax = plt.subplots(figsize=figsize)
            try:
                shap.summary_plot(shap_values, df_val, plot_type='violin', show=False)
            except Exception as e:
                shap.summary_plot(shap_values[1], df_val, plot_type='violin', show=False)
            plt.title(f'Shap Analysis (violin) for {model_name}')
            if 'save' in kwargs and bool(kwargs['save']):
                output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
                output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else f'shap_analysis_{model_name}.png'
                save_fig(fig, output_path, img_name=output_filename)
        except Exception as e:
            logger.error(f'Error on plotting shap analysis for {model_name}. Exception: {e}')
            return 

    def visual_analysis(self, features, metrics=True, feat_imp=True, cfmx=True, roc=True, score_dist=True, score_bins=True, 
                        learn_curve=True, model_shap=None, show=False, save=True, output_path=os.path.join(os.getcwd(), 'output/imgs')):
        """
        Makes a complete visual analysis for trained models by executing all individual graphic functions
        squencially, passing arguments as needed

        Parameters
        ----------
        :param features: features list used on models training [type: list]
        :param metrics: flag for executing plot_metrics() method [type: bool, default=True]
        :param feat_imp: flag for executing plot_feature_importance() method [type: bool, default=True]
        :param cfmx: flag for executing plot_confusion_matrix() method [type: bool, default=True]
        :param roc: flag for executing plot_roc_curve() method [type: bool, default=True]
        :param score_dist: flag for executing plot_score_distribution() method [type: bool, default=True]
        :param score_bins: flag for executing plot_score_bins() method [type: bool, default=True]
        :param learn_curve: flag for executing plot_learning_curve() method [type: bool, default=True]
        :param model_shap: key string for selecting a model for applying shap analysis [type: string, default=None]
        :param show: flag for showing up the figures on screen or jupyter notebook cel [type: bool, default=False]
        :param save: flag for saving figures on local machine [type: bool, default=True]
        :param output_path: path for saving files [type: string, default=cwd() + 'output/imgs']        

        Return
        ------
        This method don't return anything but the generation of plots following arguments configuration

        Application
        -----------
        trainer = BinaryClassifier()
        trainer.training_flow(set_classifiers, X_train, y_train, X_val, y_val, features)
        trainer.visual_analysis(features=MODEL_FEATURES)   
        """

        # Verifying parameter for showing up figs
        backend_ = mpl.get_backend()
        if not show:
            mpl.use('Agg')

        logger.debug(f'Initializing visual analysis for trained models')
        try:
            # Plotting metrics
            if metrics:
                self.plot_metrics(save=save, output_path=output_path)

            # Plotting feature importances
            if feat_imp:
                self.plot_feature_importance(features=features, save=save, output_path=output_path)

            # Plotting confusion matrix
            if cfmx:
                self.plot_confusion_matrix(save=save, output_path=output_path)
            
            # Plotting ROC curve
            if roc:
                self.plot_roc_curve(save=save, output_path=output_path)

            # Plotting score distribution
            if score_dist:
                self.plot_score_distribution(save=save, output_path=output_path)

            # Plotting score distribution on bins
            if score_bins:
                self.plot_score_bins(save=save, output_path=output_path)

            # Plotting learning curve
            if learn_curve:
                self.plot_learning_curve(save=save, output_path=output_path)

            # Plotting shap analysis
            if model_shap is not None:
                self.plot_shap_analysis(save=save, model_name=model_shap, features=features, output_path=output_path)

        except Exception as e:
            logger.error(f'Error on plotting visual analysis for models. Exception: {e}')

        # Reseting configuration
        mpl.use(backend_)

    def get_estimator(self, model_name):
        """
        Returns the estimator of a selected model

        Parameters
        ----------
        :param model_name: key string for extracting the model from classifiers_info class attribute [type: string]

        Return
        ------
        :return model: model estimator stored on class attribute [type: estimator]

        Application
        -----------
        model = trainer.get_estimator(model_name='RandomForestClassifier')
        """

        logger.debug(f'Returning estimator for model {model_name} stored on class attribute')
        try:
            model_info = self.classifiers_info[model_name]
            return model_info['estimator']
        except Exception as e:
            logger.error(f'Key string {model_name} does not exists or was not trained. Options: {list(self.classifiers_info.keys())}')
            return

    def get_metrics(self, model_name):
        """
        Returns metrics computed for a specific model

        Parameters
        ----------
        :param model_name: key string for extracting the model from classifiers_info class attribute [type: string]

        Return
        ------
        :return metrics: metrics dataset for a specific model [type: DataFrame]

        Application
        -----------
        metrics = trainer.get_metrics(model_name='RandomForestClassifier')
        """

        logger.debug(f'Returning metrics computed for {model_name}')
        try:
            # Returning dictionary class attribute with stored information of model
            model_info = self.classifiers_info[model_name]
            train_performance = model_info['train_performance']
            val_performance = model_info['val_performance']
            model_performance = train_performance.append(val_performance)
            model_performance.reset_index(drop=True, inplace=True)

            return model_performance
        except Exception as e:
            logger.error(f'Error on returning metrics for {model_name}. Exception: {e}')

    def get_model_info(self, model_name):
        """
        Returns a complete dictionary with all information for models stored on class attribute

        Parameters
        ----------
        :param model_name: key string for extracting the model from classifiers_info class attribute [type: string]

        Return
        ------
        :return model_info: dictionary with stored model's informations [type: dict]
            model_info = {
                'estimator': model,
                'train_scores': np.array,
                'test_scores': np.array,
                'train_performance': pd.DataFrame,
                'test_performance': pd.DataFrame,
                'model_data': {
                    'X_train': np.array,
                    'y_train': np.array,
                    'X_val': np.array,
                    'y_val': np.array,
                'feature_importances': pd.DataFrame
                }
            }

        Application
        -----------
        metrics = trainer.get_model_info(model_name='RandomForestClassifier')
        """

        logger.debug(f'Returning all information for {model_name}')
        try:
            # Retornando dicionário do modelo
            return self.classifiers_info[model_name]
        except Exception as e:
            logger.error(f'Error on returning informations for {model_name}. Exception  {e}')

    def get_classifiers_info(self):
        """
        Returns the class attribute classifiers_info with all information for all models

        Parameters
        ----------
        None

        Return
        ------
        :return classifiers_info: dictionary with information for all models
            classifiers_info ={
                'model_name': model_info = {
                                'estimator': model,
                                'train_scores': np.array,
                                'test_scores': np.array,
                                'train_performance': pd.DataFrame,
                                'test_performance': pd.DataFrame,
                                'model_data': {
                                    'X_train': np.array,
                                    'y_train': np.array,
                                    'X_val': np.array,
                                    'y_val': np.array,
                                'feature_importances': pd.DataFrame
                                }
                            }
        """

        return self.classifiers_info


"""
---------------------------------------------------
---------------- 2. CLASSIFICATION ----------------
           2.2 Multiclass Classification
---------------------------------------------------
"""

class ClassificadorMulticlasse:
    """
    Classe responsável por consolidar métodos úteis para o treinamento
    e avaliação de modelos de classificação multiclasse em um contexto de
    aprendizado supervisionado
    """

    def __init__(self, encoded_target=False):
        """
        Método construtor inicializa dicionário de informações dos modelos treinados
        """
        self.classifiers_info = {}
        self.encoded_target = encoded_target

    def fit(self, set_classifiers, X_train, y_train, **kwargs):
        """
        Método responsável por treinar cada um dos classificadores contidos no dicionário
        set_classifiers através da aplicação das regras estabelecidas pelos argumentos do método

        Parâmetros
        ----------
        :param set_classifiers: dicionário contendo informações dos modelos a serem treinados [type: dict]
            set_classifiers = {
                'model_name': {
                    'model': __estimator__,
                    'params': __estimator_params__
                }
            }
        :param X_train: features do modelo a ser treinado [type: np.array]
        :param y_train: array contendo variável target do modelo [type: np.array]
        :param **kwargs: argumentos adicionais do método
            :arg approach: indicativo de sufixo para armazenamento no atributo classifiers_info [type: string, default: '']
            :arg random_search: flag para aplicação do RandomizedSearchCV [type: bool, default: False]
            :arg scoring: métrica a ser otimizada pelo RandomizedSearchCV [type: string, default: 'accuracy']
            :arg cv: K-folds utiliados na validação cruzada [type: int, default: 5]
            :arg verbose: nível de verbosity da busca aleatória [type: int, default: -1]
            :arg n_jobs: quantidade de jobs aplicados durante a busca dos hiperparâmetros [type: int, default: -1]
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento de objetos do modelo [type: string, default=cwd() + 'output/models']
            :arg model_ext: extensão do objeto gerado (pkl ou joblib) - sem o ponto [type: string, default='pkl']

        Retorno
        -------
        Este método não retorna nada além do preenchimento de informações do treinamento no atributo self.classifiers_info

        Aplicação
        ---------
        # Instanciando objeto
        trainer = ClassificadorBinario()
        trainer.fit(set_classifiers, X_train_prep, y_train)
        """

        # Referenciando argumentos adicionais
        approach = kwargs['approach'] if 'approach' in kwargs else ''

        # Iterando sobre os modelos presentes no dicionário de classificadores
        try:
            for model_name, model_info in set_classifiers.items():
                # Definindo chave do classificador para o dicionário classifiers_info
                model_key = model_name + approach
                logger.debug(f'Treinando modelo {model_key}')
                model = model_info['model']

                # Criando dicionário vazio para armazenar dados do modelo
                self.classifiers_info[model_key] = {}

                # Validando aplicação da busca aleatória pelos melhores hiperparâmetros
                try:
                    if 'random_search' in kwargs and bool(kwargs['random_search']):
                        params = model_info['params']
                        
                        # Retornando parâmetros em kwargs
                        scoring = kwargs['scoring'] if 'scoring' in kwargs else 'accuracy'
                        cv = kwargs['cv'] if 'cv' in kwargs else 5
                        verbose = kwargs['verbose'] if 'verbose' in kwargs else -1
                        n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else -1
                        
                        # Preparando e aplicando busca
                        rnd_search = RandomizedSearchCV(model, params, scoring=scoring, cv=cv,
                                                        verbose=verbose, random_state=42, n_jobs=n_jobs)
                        logger.debug('Aplicando RandomizedSearchCV')
                        rnd_search.fit(X_train, y_train)

                        # Salvando melhor modelo no atributo classifiers_info
                        self.classifiers_info[model_key]['estimator'] = rnd_search.best_estimator_
                    else:
                        # Treinando modelo sem busca e salvando no atirbuto
                        self.classifiers_info[model_key]['estimator'] = model.fit(X_train, y_train)
                except TypeError as te:
                    logger.error(f'Erro ao aplicar RandomizedSearch. Exception lançada: {te}')
                    return

                # Validando salvamento de objetos pkl dos modelos
                if 'save' in kwargs and bool(kwargs['save']):
                    logger.debug(f'Salvando arquivo pkl do modelo {model_name} treinado')
                    model = self.classifiers_info[model_key]['estimator']
                    output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/models')
                    model_ext = kwargs['model_ext'] if 'model_ext' in kwargs else 'pkl'

                    anomesdia = datetime.now().strftime('%Y%m%d')
                    save_model(model, output_path=output_path, 
                                    filename=model_name.lower() + '_' +  anomesdia + '.' + model_ext)

        except AttributeError as ae:
            logger.error(f'Erro ao treinar modelos. Exception lançada: {ae}')
            logger.warning(f'Treinamento do(s) modelo(s) não realizado')

    def compute_train_performance(self, model_name, estimator, X, y, cv=5, target_names=None):
        """
        Método responsável por aplicar validação cruzada para retornar a amédia das principais métricas de avaliação
        de um modelo de classificação. Na prática, esse método é chamado por um outro método em uma camada
        superior da classe para medição de performance em treino e em teste

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo contida no atributo self.classifiers_info [type: string]
        :param estimator: estimator do modelo a ser avaliado [type: object]
        :param X: conjunto de features do modelo contido nos dados de treino [type: np.array]
        :param y: array contendo a variável resposta dos dados de treino do modelo [type: np.array]
        :param cv: K-folds utiliados na validação cruzada [type: int, default: 5]
        :param target_names: lista com referências para as classes [type: list, default=None]

        Retorno
        -------
        :return train_performance: DataFrame contendo as métricas calculadas usando validação cruzada [type: pd.DataFrame]

        Aplicação
        ---------
        # Instanciando e treinando modelo
        trainer = ClassificadorBinario()
        trainer.fit(model, X_train, y_train)
        train_performance = trainer.compute_train_performance(model_name, estimator, X_train, y_train)
        """

        # Computando métricas utilizando validação cruzada
        logger.debug(f'Computando métricas do modelo {model_name} utilizando validação cruzada com {cv} K-folds')
        try:
            # Iniciando cronômetro e computando predições via validação cruzada
            t0 = time.time()
            y_pred = cross_val_predict(estimator, X, y, cv=cv)
            
            # Construindo classification report
            cr = pd.DataFrame(classification_report(y, y_pred, output_dict=True, target_names=target_names)).T
            
            # Definindo fluxo para target codificado ou nao
            if self.encoded_target:
                n_classes = len(cr) - 4
                acc = [accuracy_score(y.T[i], y_pred.T[i]) for i in range(n_classes)]
            else:
                n_classes = len(cr) - 3
                acc = cr.loc['accuracy', :].values
                acc = [acc[0]] * n_classes
            
            # Customizando classification report
            cr_custom = cr.iloc[:n_classes, :-1]
            cr_custom['model'] = model_name
            cr_custom.reset_index(inplace=True)
            cr_custom.columns = ['class'] + list(cr_custom.columns[1:])
            
            # Calculando acurácia para cada classe
            cr_custom['accuracy'] = acc
            cr_custom['approach'] = f'Treino {cv} K-folds'
            cr_cols = ['model', 'approach', 'class', 'accuracy', 'precision', 'recall', 'f1-score']
            train_performance = cr_custom.loc[:, cr_cols]

            # Adicionando medição de tempo no DataFrame
            t1 = time.time()
            delta_time = t1 - t0
            train_performance['total_time'] = round(delta_time, 3)
            logger.info(f'Métricas computadas com sucesso nos dados de treino em {round(delta_time, 3)} segundos')

            return train_performance

        except Exception as e:
            logger.error(f'Erro ao computar as métricas. Exception lançada: {e}')    

    def compute_val_performance(self, model_name, estimator, X, y, target_names=None):
        """
        Método responsável por aplicar retornar as principais métricas do model utilizando dados de validação.
        Na prática, esse método é chamado por um outro método em uma camada superior da classe para medição 
        de performance em treino e em validação

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo contida no atributo self.classifiers_info [type: string]
        :param estimator: estimator do modelo a ser avaliado [type: object]
        :param X: conjunto de features do modelo contido nos dados de teste [type: np.array]
        :param y: array contendo a variável resposta dos dados de teste do modelo [type: np.array]
        :param target_names: lista com referências para as classes [type: list, default=None]

        Retorno
        -------
        :return val_performance: DataFrame contendo as métricas calculadas nos dados de validação [type: pd.DataFrame]

        Aplicação
        ---------
        # Instanciando e treinando modelo
        trainer = ClassificadorBinario()
        trainer.fit(model, X_train, y_train)
        val_performance = trainer.compute_val_performance(model_name, estimator, X_val, y_val)
        """

        # Predicting data using the trained model and computing probabilities
        logger.debug(f'Computando métricas do modelo {model_name} utilizando dados de teste')
        try:
            # Iniciando cronômetro e computando predições puras
            t0 = time.time()
            y_pred = estimator.predict(X)

            # Construindo classification report
            cr = pd.DataFrame(classification_report(y, y_pred, output_dict=True, target_names=target_names)).T
            
            # Definindo fluxo para target codificado ou nao
            if self.encoded_target:
                n_classes = len(cr) - 4
                acc = [accuracy_score(y.T[i], y_pred.T[i]) for i in range(n_classes)]
            else:
                n_classes = len(cr) - 3
                acc = cr.loc['accuracy', :].values
                acc = [acc[0]] * n_classes
                
            # Customizando classification report
            cr_custom = cr.iloc[:n_classes, :-1]
            cr_custom['model'] = model_name
            cr_custom.reset_index(inplace=True)
            cr_custom.columns = ['class'] + list(cr_custom.columns[1:])

            # Calculando acurácia para cada classe
            cr_custom['accuracy'] = acc
            cr_custom['approach'] = f'Validation set'
            cr_cols = ['model', 'approach', 'class', 'accuracy', 'precision', 'recall', 'f1-score']
            val_performance = cr_custom.loc[:, cr_cols]

            # Adicionando medição de tempo no DataFrame
            t1 = time.time()
            delta_time = t1 - t0
            val_performance['total_time'] = round(delta_time, 3)
            logger.info(f'Métricas computadas com sucesso nos dados de validação em {round(delta_time, 3)} segundos')

            return val_performance

        except Exception as e:
            logger.error(f'Erro ao computar as métricas. Exception lançada: {e}')    

    def evaluate_performance(self, X_train, y_train, X_val, y_val, cv=5, target_names=None, **kwargs):
        """
        Método responsável por executar e retornar métricas dos classificadores em treino (média do resultado
        da validação cruzada com cv K-fols) e teste
        
        Parâmetros
        ----------
        :param X_train: conjunto de features do modelo contido nos dados de treino [type: np.array]
        :param y_train: array contendo a variável resposta dos dados de treino do modelo [type: np.array]
        :param X_val: conjunto de features do modelo contido nos dados de teste [type: np.array]
        :param y_test: array contendo a variável resposta dos dados de teste do modelo [type: np.array]
        :param cv: K-folds utiliados na validação cruzada [type: int, default: 5]
        :param target_names: lista com referências para as classes [type: list, default=None]
        :param encoded_target: define a aplicação do encoding no array target [type: bool, default=True]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/metrics']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='metrics.csv']
        
        Retorno
        -------
        :return df_performances: DataFrame contendo as métricas calculadas em treino e teste [type: pd.DataFrame]

        Aplicação
        -----------
        # Treinando modelo e avaliando performance em treino e teste
        trainer = ClassificadorBinario()
        trainer.fit(estimator, X_train, X_test)

        # Definindo dicionário de controle do resultado
        df_performance = trainer.evaluate_performance(X_train, y_train, X_val, y_val, save=True, output_path=caminho)
        """

        # DataFrame vazio para armazenamento das métrics
        df_performances = pd.DataFrame({})

        # Iterando sobre todos os classificadores da classe
        for model_name, model_info in self.classifiers_info.items():

            # Validando se o modelo já foi treinado (dicionário model_info já terá a chave 'train_performance')
            if 'train_performance' in model_info.keys():
                df_performances = df_performances.append(model_info['train_performance'])
                df_performances = df_performances.append(model_info['val_performance'])
                continue

            # Retornando modelo a ser avaliado
            try:
                estimator = model_info['estimator']
            except KeyError as e:
                logger.error(f'Erro ao retornar a chave "estimator" do dicionário model_info. Modelo {model_name} não treinado')
                continue

            # Computando performance em treino e em teste
            train_performance = self.compute_train_performance(model_name, estimator, X_train, y_train, 
                                                               cv=cv, target_names=target_names)
            val_performance = self.compute_val_performance(model_name, estimator, X_val, y_val, 
                                                           target_names=target_names)
    
            # Adicionando os resultados ao atributo classifiers_info
            self.classifiers_info[model_name]['train_performance'] = train_performance
            self.classifiers_info[model_name]['val_performance'] = val_performance

            # Construindo DataFrame com as métricas retornadas
            model_performance = train_performance.append(val_performance)
            df_performances = df_performances.append(model_performance)
            df_performances['anomesdia_datetime'] = datetime.now()

            # Salvando alguns atributos no dicionário classifiers_info para acessos futuros
            model_data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val
            }
            model_info['model_data'] = model_data

        # Validando salvamento dos resultados
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/metrics')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'metrics.csv'
            save_data(df_performances, output_path=output_path, filename=output_filename)

        return df_performances

    def feature_importance(self, features, top_n=-1, **kwargs):
        """
        Método responsável por retornar a importância das features de um modelo treinado
        
        Parâmetros
        ----------
        :param features: lista contendo as features de um modelo [type: list]
        :param top_n: parâmetro para filtragem das top n features [type: int, default=-1]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/metrics']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='top_features.csv']

        Retorno
        -------
        :return: all_feat_imp: pandas DataFrame com a análise de feature importance dos modelos [type: pd.DataFrame]
        """

        # Inicializando DataFrame vazio para armazenamento das feature importance
        feat_imp = pd.DataFrame({})
        all_feat_imp = pd.DataFrame({})

        # Iterando sobre os modelos presentes na classe
        for model_name, model_info in self.classifiers_info.items():
            # Validando possibilidade de extrair a importância das features do modelo
            logger.debug(f'Extraindo importância das features para o modelo {model_name}')

            try:
                importances = model_info['estimator'].feature_importances_
            except KeyError as ke:
                logger.warning(f'Modelo {model_name} não treinado, sendo impossível extrair o método feature_importances_')
                continue
            except AttributeError as ae:
                logger.warning(f'Modelo {model_name} não possui o método feature_importances_')
                continue

            # Preparando o dataset para armazenamento das informações
            feat_imp['feature'] = features
            feat_imp['importance'] = importances
            feat_imp['model'] = model_name
            feat_imp['anomesdia_datetime'] = datetime.now()
            feat_imp.sort_values(by='importance', ascending=False, inplace=True)
            feat_imp = feat_imp.loc[:, ['model', 'feature', 'importance', 'anomesdia_datetime']]

            # Salvando essa informação no dicionário classifiers_info
            self.classifiers_info[model_name]['feature_importances'] = feat_imp
            all_feat_imp = all_feat_imp.append(feat_imp)
            logger.info(f'Extração da importância das features concluída com sucesso para o modelo {model_name}')

        # Validando salvamento dos resultados
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/metrics')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'top_features.csv'
            save_data(all_feat_imp, output_path=output_path, filename=output_filename)
        
        return all_feat_imp

    def training_flow(self, set_classifiers, X_train, y_train, X_test, y_test, features, **kwargs):
        """
        Método responsável por consolidar um fluxo completo de treinamento dos classificadores, bem como
        o levantamento de métricas e execução de métodos adicionais para escolha do melhor modelo

        Parameters
        ----------
        :param set_classifiers: dicionário contendo informações dos modelos a serem treinados [type: dict]
            set_classifiers = {
                'model_name': {
                    'model': __estimator__,
                    'params': __estimator_params__
                }
            }
        :param X_train: conjunto de features do modelo contido nos dados de treino [type: np.array]
        :param y_train: array contendo a variável resposta dos dados de treino do modelo [type: np.array]
        :param X_test: conjunto de features do modelo contido nos dados de teste [type: np.array]
        :param y_test: array contendo a variável resposta dos dados de teste do modelo [type: np.array]
        :param features: lista contendo as features de um modelo [type: list]
        :param output_path: caminho onde o arquivo de resultados será salvo: [type: string, default=os.path.join(os.path.getcwd(), 'output/')]
        :param **kwargs: argumentos adicionais do método
            :arg approach: indicativo de sufixo para armazenamento no atributo classifiers_info [type: string, default: '']
            :arg random_search: flag para aplicação do RandomizedSearchCV [type: bool, default: False]
            :arg scoring: métrica a ser otimizada pelo RandomizedSearchCV [type: string, default: 'accuracy']
            :arg cv: K-folds utiliados na validação cruzada [type: int, default: 5]
            :arg verbose: nível de verbosity da busca aleatória [type: int, default: 5]
            :arg n_jobs: quantidade de jobs aplicados durante a busca dos hiperparâmetros [type: int, default: -1]
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg models_output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/models']
            :arg metrics_output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/metrics']
            :arg metrics_output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='metrics.csv']
            :arg featimp_output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='top_features.csv']
            :arg top_n_featimp: top features a serem analisadas na importância das features [type: int, default=-1]

        Retorno
        -------
        None

        Aplicação
        ---------
        # Instanciando objeto
        trainer = ClassificadorBinario()
        trainer.training_flow(set_classifiers, X_train, y_train, X_test, y_test, features)
        """

        # Extraindo parâmetros kwargs
        approach = kwargs['approach'] if 'approach' in kwargs else ''
        random_search = kwargs['random_search'] if 'random_search' in kwargs else False
        scoring = kwargs['scoring'] if 'scoring' in kwargs else 'accuracy'
        cv = kwargs['cv'] if 'cv' in kwargs else 5
        verbose = kwargs['verbose'] if 'verbose' in kwargs else -1
        n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else -1
        save = bool(kwargs['save']) if 'save' in kwargs else True
        models_output_path = kwargs['models_output_path'] if 'models_output_path' in kwargs else os.path.join(os.getcwd(), 'output/models')
        metrics_output_path = kwargs['metrics_output_path'] if 'metrics_output_path' in kwargs else os.path.join(os.getcwd(), 'output/metrics')
        metrics_output_filename = kwargs['metrics_output_filename'] if 'metrics_output_filename' in kwargs else 'metrics.csv'
        featimp_output_filename = kwargs['featimp_output_filename'] if 'featimp_output_filename' in kwargs else 'top_features.csv'
        top_n_featimp = kwargs['top_n_featimp'] if 'top_n_featimp' in kwargs else -1
        target_names = kwargs['target_names'] if 'target_names' in kwargs else list(range(len(np.unique(y_train))))

        # Treinando classificadores
        self.fit(set_classifiers, X_train, y_train, approach=approach, random_search=random_search, scoring=scoring,
                 cv=cv, verbose=verbose, n_jobs=n_jobs, save=save, output_path=models_output_path)

        # Avaliando modelos
        self.evaluate_performance(X_train, y_train, X_test, y_test, save=save, output_path=metrics_output_path, 
                                  target_names=target_names, output_filename=metrics_output_filename)

        # Analisando features mais importantes
        self.feature_importance(features, top_n=top_n_featimp, save=save, output_path=metrics_output_path, 
                                output_filename=featimp_output_filename)

    def plot_metrics(self, df_metrics=None, idx_cols=['model', 'class', 'approach'], 
                 value_cols=['accuracy', 'precision', 'recall', 'f1-score'], font_scale=1.2,
                 row='approach', col='model', margin_titles=True, x='class', y='value', hue='variable', 
                 palette='rainbow', legend_loc='center right', x_rotation=30, annot_ndec=2, 
                 annot_fontsize=8, **kwargs):
        """
        Método responsável por plotar os resultados das métricas dos classificadores selecionados

        Parâmetros
        ----------
        :param df_metrics: DataFrame específico contendo as métricas já calculadas pelo objeto [type: pd.DataFrame]
        :param idx_cols: colunas da base de métricas a serem pivotadas 
            [type: list, default=['model', 'class', 'approach']]
        :param value_cols: colunas da base de métricas com os valores das métricas
            [type: list, default=['accuracy', 'precision', 'recall', 'f1-score']
        :param font_scale: escala de fonte do FacetGrid [type: float, default=1.2]
        :param row: referência separadora de linhas do grid [type: string, default='approach']
        :param col: referência separadora de colunas do grid [type: string, default='model']
        :param margin_titles: flag para plotagem dos títulos do grid [type: bool, default=True]
        :param x: eixo x da plotagem [type: string, default='class']
        :param y: eixo y da plotagem [type: string, default='value']
        :param hue: separação das barras em cada plotagem [type: string, default='variable']
        :param palette: paleta de cores utilizada [type: string, default='rainbow']
        :param legend_loc: posicionamento da legenda [type: string, default='center right']
        :param x_rotation: rotação dos labels no eixo x [type: int, default=30]
        :param annot_ndec: casas decimais dos labels nas barras [type: int, default=2]
        :param annot_fontsize: tamanho da fonte dos labels nas barras [type: int, default=8]

        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='metrics_comparison.png']

        Retorno
        -------
        Este método não retorna nenhum parâmetro além da plotagem devidamente salva no diretório destino
        """

        

        # Verificando a necessidade de calcular as métricas
        if df_metrics is None:
            # Retornando elementos necessários para o cálculo da performance
            try:
                X_train = kwargs['X_train']
                y_train = kwargs['y_train']
                X_val = kwargs['X_val']
                y_val = kwargs['y_val']
                target_names = kwargs['target_names'] if 'target_names' in kwargs else list(range(len(np.unique(y_train))))
                df_metrics = self.evaluate_performance(X_train, y_train, X_val, y_val, 
                                                       target_names=target_names)
            except Exception as e:
                logger.error(f'Não foi inserido o argumento df_metrics e nem X e y para calculo. Exception lançada: {e}')
                return

        # Pivotando DataFrame de métricas de modo a facilitar a plotagem
        metrics_melt = pd.melt(df_metrics, id_vars=idx_cols, value_vars=value_cols)

        # Criando FatecGrid e plotando gráfico
        with sns.plotting_context('notebook', font_scale=1.2):
            g = sns.FacetGrid(metrics_melt, row=row, col=col, margin_titles=margin_titles)
            g.map_dataframe(sns.barplot, x=x, y=y, hue=hue, palette=palette)

        # Customizando figura
        figsize = (17, 5 * len(np.unique(metrics_melt[row])))
        g.fig.set_size_inches(figsize)
        plt.legend(loc=legend_loc)
        g.set_xticklabels(rotation=x_rotation)

        for axs in g.axes:
            for ax in axs:
                AnnotateBars(n_dec=annot_ndec, color='black', font_size=annot_fontsize).vertical(ax)

        plt.tight_layout()

        # Salvando figura
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'metrics_comparison.png'
            save_fig(g.fig, output_path=output_path, img_name=output_filename)   
    
    def custom_confusion_matrix(self, model_name, y_true, y_pred, classes, cmap, normalize=False):
        """
        Método utilizada para plotar uma matriz de confusão customizada para um único modelo da classe. Em geral,
        esse método pode ser chamado por um método de camada superior para plotagem de matrizes para todos os
        modelos presentes na classe

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo contida no atributo self.classifiers_info [type: string]
        :param y_true: array contendo a variável target do dataset [type: np.array]
        :param y_pred: array com as predições retornadas pelo respectivo modelo [type: np.array]
        :param classes: nomenclatura das classes da matriz [type: list]
        :param cmap: colormap para a matriz gerada [type: matplotlib.colormap]
        :param normalize: flag para normalizar as entradas da matriz [type: bool, default=False]

        Retorno
        -------
        Este método não retorna nenhuma variável, além da plotagem da matriz especificada

        Aplicação
        -----------
        Visualizar o método self.plot_confusion_matrix()
        """

        # Retornando a matriz de confusão usando função do sklearn
        if not self.encoded_target:
            y_true = pd.get_dummies(y_true).values
            y_pred = pd.get_dummies(y_pred).values
            
        conf_mx = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
        
        
        # Plotando matriz
        plt.imshow(conf_mx, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        tick_marks = np.arange(len(classes))

        # Customizando eixos
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Customizando entradas
        fmt = '.2f' if normalize else 'd'
        thresh = conf_mx.max() / 2.
        for i, j in itertools.product(range(conf_mx.shape[0]), range(conf_mx.shape[1])):
            plt.text(j, i, format(conf_mx[i, j]),
                     horizontalalignment='center',
                     color='white' if conf_mx[i, j] > thresh else 'black')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'{model_name}\nConfusion Matrix', size=12)

    def plot_confusion_matrix(self, classes=None, cmap=plt.cm.Blues, normalize=False, **kwargs):
        """
        Método responsável por plotar gráficos de matriz de confusão usando dados de treino e teste
        para todos os modelos presentes no dicionárion de classificadores self.classifiers_info

        Parâmetros
        ----------
        :param cmap: colormap para a matriz gerada [type: matplotlib.colormap]
        :param normalize: flag para normalizar as entradas da matriz [type: bool, default=False]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='confusion_matrix.png']

        Retorno
        -------
        Este método não retorna nenhuma variável, além da plotagem da matriz especificada

        Aplicação
        ---------
        trainer = ClassificadorBinario()
        trainer.training_flow(set_classifiers, X_train, y_train, X_test, y_test, features)
        trainer.plot_confusion_matrix(output_path=OUTPUT_PATH)
        """

        # Definindo parâmetros de plotagem
        logger.debug('Inicializando plotagem da matriz de confusão para os modelos')
        k = 1
        nrows = len(self.classifiers_info.keys())
        fig = plt.figure(figsize=(14, nrows * 5.5))
        sns.set(style='white', palette='muted', color_codes=True)

        # Iterando sobre cada classificador da classe
        for model_name, model_info in self.classifiers_info.items():
            logger.debug(f'Retornando dados de treino e validação para o modelo {model_name}')
            try:
                # Retornando dados para cada modelo
                X_train = model_info['model_data']['X_train']
                y_train = model_info['model_data']['y_train']
                X_val = model_info['model_data']['X_val']
                y_val = model_info['model_data']['y_val']
                
                # Definindo classes para a plotagem
                if classes is None:
                    classes = list(range(len(np.unique(y_train))))
                
            except Exception as e:
                logger.error(f'Erro ao retornar dados para o modelo {model_name}. Exception lançada: {e}')
                continue

            # Realizando predições em treino (cross validation) e teste
            logger.debug(f'Realizando predições para os dados de treino e validação ({model_name})')
            try:
                train_pred = cross_val_predict(model_info['estimator'], X_train, y_train, cv=5)
                val_pred = model_info['estimator'].predict(X_val)
            except Exception as e:
                logger.error(f'Erro ao realizar predições para o modelo {model_name}. Exception lançada: {e}')
                continue

            logger.debug(f'Gerando matriz de confusão para o modelo {model_name}')
            try:
                # Plotando matriz utilizando dados de treino
                plt.subplot(nrows, 2, k)
                self.custom_confusion_matrix(model_name + ' Train', y_train, train_pred, classes=classes, 
                                             cmap=cmap, normalize=normalize)
                k += 1

                # Plotando matriz utilizando dados de validação
                plt.subplot(nrows, 2, k)
                self.custom_confusion_matrix(model_name + ' Validation', y_val, val_pred, classes=classes, 
                                             cmap=plt.cm.Greens, normalize=normalize)
                k += 1
                logger.info(f'Matriz de confusão gerada para o modelo {model_name}')
            except Exception as e:
                logger.error(f'Erro ao gerar a matriz para o modelo {model_name}. Exception lançada: {e}')
                continue

        # Alinhando figura
        plt.tight_layout()

        # Salvando imagem
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'confusion_matrix.png'
            save_fig(fig, output_path=output_path, img_name=output_filename)   
   
    def plot_feature_importance(self, features, top_n=20, palette='viridis', **kwargs):
        """
        Método responsável por realizar uma plotagem gráfica das variáveis mais importantes pro modelo

        Parâmetros
        ----------
        :param features: lista de features do conjunto de dados [type: list]
        :param top_n: quantidade de features a ser considerada na plotagem [type: int, default=20]
        :param palette: paleta de cores utilizazda na plotagem [type: string, default='viridis']
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='feature_importances.png']

        Retorno
        -------
        Este método não retorna nada além da imagem devidamente salva no diretório destino
        """

        # Definindo parâmetros de plotagem
        logger.debug('Inicializando plotagem das features mais importantes para os modelos')
        feat_imp = pd.DataFrame({})
        i = 0
        ax_del = 0
        nrows = len(self.classifiers_info.keys())
        fig, axs = plt.subplots(nrows=nrows, figsize=(16, nrows * 6))
        sns.set(style='white', palette='muted', color_codes=True)
        
        # Iterando sobre os modelos presentes na classe
        for model_name, model_info in self.classifiers_info.items():
            # Validando possibilidade de extrair a importância das features do modelo
            logger.debug(f'Extraindo importância das features para o modelo {model_name}')
            try:
                importances = model_info['estimator'].feature_importances_
            except:
                logger.warning(f'Modelo {model_name} não possui o método feature_importances_')
                ax_del += 1
                continue
            
            # Preparando o dataset para armazenamento das informações
            feat_imp['feature'] = features
            feat_imp['importance'] = importances
            feat_imp.sort_values(by='importance', ascending=False, inplace=True)

            logger.debug(f'Plotando gráfico de importância das features para o modelo {model_name}')
            try:
                # Plotando feature importance
                sns.barplot(x='importance', y='feature', data=feat_imp.iloc[:top_n, :], ax=axs[i], 
                            palette=palette)

                # Customizando gráfico
                axs[i].set_title(f'Features Mais Importantes: {model_name}', size=14)
                format_spines(axs[i], right_border=False)
                i += 1
  
                logger.info(f'Gráfico de importância das features plotado com sucesso para o modelo {model_name}')
            except Exception as e:
                logger.error(f'Erro ao gerar gráfico de importância das features para o modelo {model_name}. Exception lançada: {e}')
                continue

        # Deletando eixos sobressalentes (se aplicável)
        if ax_del > 0:
            logger.debug('Deletando eixos referentes a análises não realizadas')
            try:
                for i in range(-1, -(ax_del+1), -1):
                    fig.delaxes(axs[i])
            except Exception as e:
                logger.error(f'Erro ao deletar eixo. Exception lançada: {e}')
        
        # Alinhando figura
        plt.tight_layout()

        # Salvando imagem
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'feature_importances.png'
            save_fig(fig, output_path=output_path, img_name=output_filename)
    
    def plot_shap_analysis(self, model_name, features, n_classes, figsize=(16, 10), target_names=None,
                           **kwargs):
        """
        Método responsável por plotar a análise shap pras features em um determinado modelo
        
        Parâmetros
        ----------
        :param model_name: chave de um classificador específico já treinado na classe [type: string]
        :param features: lista de features do dataset [type: list]
        :param n_classes: número de classes presentes no modelo [type: int]
        :param figsize: tamanho da figure de plotagem [type: tuple, default=(16, 10)]
        :param target_names: nomes das classes para título da plotagem [type: list, default=None]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='confusion_matrix.png']

        Retorno
        -------
        Este método não retorna nenhum parâmetro além da análise shap especificada
        """

        logger.debug(f'Explicando o modelo {model_name} através da análise shap')
        try:
            model_info = self.classifiers_info[model_name]
            model = model_info['estimator']
        except Exception as e:
            logger.error(f'Classificador {model_name} não existente ou não treinado. Opções possíveis: {list(self.classifiers_info.keys())}')
            return

        logger.debug(f'Retornando parâmetros da classe para o modelo {model_name}')
        try:
            # Retornando parâmetros do modelo
            X_train = model_info['model_data']['X_train']
            X_val = model_info['model_data']['X_val']
            df_train = pd.DataFrame(X_train, columns=features)
            df_val = pd.DataFrame(X_val, columns=features)
        except Exception as e:
            logger.error(f'Erro ao retornar parâmetros para o modelo {model_name}. Exception lançada: {e}')

        logger.debug(f'Criando explainer e gerando valores shap para o modelo {model_name}')
        try:
            explainer = shap.TreeExplainer(model, df_train)
            shap_values = explainer.shap_values(df_val)
        except Exception as e:
            try:
                logger.warning(f'TreeExplainer não se encaixa no modelo {model_name}. Tentando LinearExplainer')
                explainer = shap.LinearExplainer(model, df_train)
                shap_values = explainer.shap_values(df_val, check_additivity=False)
            except Exception as e:
                logger.error(f'Não foi possível retornar os parâmetros para o modelo {model_name}. Exception lançada: {e}')
                return
        
        # Configurando plotagem
        nrows = ceil(n_classes / 2)
        fig = plt.figure()
        if not target_names:
            target_names = [f'Classe {i}' for i in range(1, n_classes + 1)]
        
        # Iterando sobre as classes
        logger.debug(f'Plotando análise shap para o modelo {model_name}')
        for c in range(1, n_classes + 1):
            plt.subplot(nrows, 2, c)
            try:
                shap.summary_plot(shap_values[c - 1], X_val, plot_type='violin', show=False, 
                                  plot_size=(15, nrows * 9))
            except Exception as e:
                logger.error(f'Erro ao plotar análise shap para o modelo {model_name}. Exception lançada: {e}')
                return
            
            # Configurações finais e salvamento da imagem
            plt.title(f'Shap summary plot: {target_names[c - 1]}', size=15)
            plt.tight_layout()
            if 'save' in kwargs and bool(kwargs['save']):
                output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
                output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else f'shap_analysis_{model_name}.png'
                save_fig(fig, output_path, img_name=output_filename)
    
    def visual_analysis(self, features, metrics=True, df_metrics=None, feat_imp=True, cfmx=True, 
                        model_shap=None, show=False, save=True, output_path=os.path.join(os.getcwd(), 'output/imgs'), 
                        **kwargs):
        """
        Método responsável por consolidar análises gráficas no processo de modelagem

        Parâmetros
        ----------
        :param features: lista de features do conjunto de dados [type: list]
        :param metrics: flag inficativo da execução do método plot_metrics() [type: bool, default=True]
        :param feat_imp: flag indicativo da execução da método plot_feature_importance() [type: bool, default=True]
        :param cfmx: flag indicativo da execução da método plot_confusion_matrix() [type: bool, default=True]
        :param roc: flag indicativo da execução da método plot_roc_curve() [type: bool, default=True]
        :param score_dist: flag indicativo da execução da método plot_score_distribution() [type: bool, default=True]
        :param score_bins: flag indicativo da execução da método plot_score_bins() [type: bool, default=True]
        :param learn_curve: flag indicativo da execução da método plot_learning_curve() [type: bool, default=True]
        :param model_shap: chave do modelo a ser utilizado na análise shap [type: string, default=None]
        :param show: flag indicativo para mostragem das figuras em jupyter notebook [type: bool, default=False]
        :param save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
        :param output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']        

        Retorno
        -------
        Este método não retorna nada além das imagens devidamente salvas no diretório destino

        Aplicação
        ---------
        trainer = ClassificadorBinario()
        trainer.fit(set_classifiers, X_train, y_train, X_test, y_test)        
        """

        # Verificando parâmetro para mostrar
        backend_ = mpl.get_backend()
        if not show:
            mpl.use('Agg')

        logger.debug(f'Inicializando análise gráfica nos modelos treinados')
        target_names = kwargs['target_names'] if 'target_names' in kwargs else list(range(len(np.unique(y_train))))
    
        # Verificando plotagem das métricas
        if metrics and df_metrics is None:
            try:
                X_train = kwargs['X_train']
                y_train = kwargs['y_train']
                X_val = kwargs['X_val']
                y_val = kwargs['y_val']
                y_val = kwargs['y_val']

                # Calculando métricas
                metrics = self.evaluate_performance(X_train, y_train, X_val, y_val, target_names=target_names)
              
            except Exception as e:
                logger.error(f'Para a plotagem das métricas, é necessário fornecer ou a base de métricas ou X e y')
            
            self.plot_metrics(df_metrics=metrics, save=save, output_path=output_path)
            
        elif metrics and df_metrics is not None:
            self.plot_metrics(df_metrics=metrics, save=save, output_path=output_path)
             
        # Verificando plotagem de feature importance
        if feat_imp:
            try:
                self.plot_feature_importance(features=features, save=save, output_path=output_path)
            except Exception as e:
                logger.error(f'Erro ao plotar feature importances. Exception: {e}')

        # Verificando plotagem de matriz de confusão
        if cfmx:
            try:
                self.plot_confusion_matrix(classes=target_names, save=save, output_path=output_path)
            except Exception as e:
                logger.error(f'Erro ao plotar matriz de confusão. Exception: {e}')

        if model_shap is not None:
            try:
                self.plot_shap_analysis(save=save, model_name=model_shap, features=features, output_path=output_path)
            except Exception as e:
                logger.error(f'Erro ao plotar análise shap. Exception: {e}')

        # Resetando configurações
        mpl.use(backend_)

    def get_estimator(self, model_name):
        """
        Método responsável por retornar o estimator de um modelo selecionado

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo no dicionário classifiers_info da classe [type: string]

        Retorno
        -------
        :return model: estimator do modelo já treinado
        """

        logger.debug(f'Retornando estimator do modelo {model_name} já treinado')
        try:
            model_info = self.classifiers_info[model_name]
            return model_info['estimator']
        except Exception as e:
            logger.error(f'Classificador {model_name} não existente ou não treinado. Opções possíveis: {list(self.classifiers_info.keys())}')
            return

    def get_metrics(self, model_name):
        """
        Método responsável por retornar as métricas obtidas no treinamento

        Parâmetros
        ----------
        None

        Retorno
        -------
        :return model_performance: DataFrame contendo as métricas dos modelos treinados [type: pd.DataFrame]
        """

        logger.debug(f'Retornando as métricas do modelo {model_name}')
        try:
            # Retornando dicionário do modelo e métricas já salvas
            model_info = self.classifiers_info[model_name]
            train_performance = model_info['train_performance']
            test_performance = model_info['test_performance']
            model_performance = train_performance.append(test_performance)
            model_performance.reset_index(drop=True, inplace=True)

            return model_performance
        except Exception as e:
            logger.error(f'Erro ao retornar as métricas para o modelo {model_name}. Exception lançada: {e}')

    def get_model_info(self, model_name):
        """
        Método responsável por coletar as informações registradas de um determinado modelo da classe

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo no dicionário classifiers_info da classe [type: string]

        Retorno
        -------
        :return model_info: dicionário com informações registradas do modelo [type: dict]
            model_info = {
                'estimator': model,
                'train_scores': np.array,
                'test_scores': np.array,
                'train_performance': pd.DataFrame,
                'test_performance': pd.DataFrame,
                'model_data': {
                    'X_train': np.array,
                    'y_train': np.array,
                    'X_test': np.array,
                    'y_test': np.array,
                'feature_importances': pd.DataFrame
                }
            }
        """

        logger.debug(f'Retornando informações registradas do modelo {model_name}')
        try:
            # Retornando dicionário do modelo
            return self.classifiers_info[model_name]
        except Exception as e:
            logger.error(f'Erro ao retornar informações do modelo {model_name}. Exception lançada: {e}')

    def get_classifiers_info(self):
        """
        Método responsável por retornar o dicionário classifiers_info contendo todas as informações de todos os modelos

        Parâmetros
        ----------
        None

        Retorno
        -------
        :return classifiers_info: dicionário completo dos modelos presentes na classe
            classifiers_info ={
                'model_name': model_info = {
                                'estimator': model,
                                'train_scores': np.array,
                                'test_scores': np.array,
                                'train_performance': pd.DataFrame,
                                'test_performance': pd.DataFrame,
                                'model_data': {
                                    'X_train': np.array,
                                    'y_train': np.array,
                                    'X_test': np.array,
                                    'y_test': np.array,
                                'feature_importances': pd.DataFrame
                                }
                            }
        """

        return self.classifiers_info



"""
---------------------------------------------------
------------------ 3. REGRESSION ----------------
              1.1 Linear Regression
---------------------------------------------------
"""

class RegressorLinear:
    """
    Classe responsável por consolidar métodos úteis para o treinamento
    e avaliação de modelos de regressão linear em um contexto de
    aprendizado supervisionado
    """
    
    def __init__(self):
        """
        Método construtor inicializa dicionário de informações dos modelos treinados
        """
        self.regressors_info = {}
        
    def fit(self, set_regressors, X_train, y_train, **kwargs):
        """
        Método responsável por treinar cada um dos regressores contidos no dicionário
        set_regressors através da aplicação das regras estabelecidas pelos argumentos do método

        Parâmetros
        ----------
        :param set_regressors: dicionário contendo informações dos modelos a serem treinados [type: dict]
            set_regressors = {
                'model_name': {
                    'model': __estimator__,
                    'params': __estimator_params__
                }
            }
        :param X_train: features do modelo a ser treinado [type: np.array]
        :param y_train: array contendo variável target do modelo [type: np.array]
        :param **kwargs: argumentos adicionais do método
            :arg approach: indicativo de sufixo para armazenamento no atributo classifiers_info [type: string, default: '']
            :arg random_search: flag para aplicação do RandomizedSearchCV [type: bool, default: False]
            :arg scoring: métrica a ser otimizada pelo RandomizedSearchCV [type: string, default: 'accuracy']
            :arg cv: K-folds utiliados na validação cruzada [type: int, default: 5]
            :arg verbose: nível de verbosity da busca aleatória [type: int, default: -1]
            :arg n_jobs: quantidade de jobs aplicados durante a busca dos hiperparâmetros [type: int, default: -1]
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento de objetos do modelo [type: string, default=cwd() + 'output/models']
            :arg model_ext: extensão do objeto gerado (pkl ou joblib) - sem o ponto [type: string, default='pkl']

        Retorno
        -------
        Este método não retorna nada além do preenchimento de informações do treinamento no atributo self.classifiers_info

        Aplicação
        ---------
        # Instanciando objeto
        trainer = RegressorLinear()
        trainer.fit(set_regressors, X_train_prep, y_train)
        """

        # Referenciando argumentos adicionais
        approach = kwargs['approach'] if 'approach' in kwargs else ''

        # Iterando sobre os modelos presentes no dicionário de classificadores
        try:
            for model_name, model_info in set_regressors.items():
                # Definindo chave do classificador para o dicionário classifiers_info
                model_key = model_name + approach
                logger.debug(f'Treinando modelo {model_key}')
                model = model_info['model']

                # Criando dicionário vazio para armazenar dados do modelo
                self.regressors_info[model_key] = {}

                # Validando aplicação da busca aleatória pelos melhores hiperparâmetros
                try:
                    if 'random_search' in kwargs and bool(kwargs['random_search']):
                        params = model_info['params']
                        
                        # Retornando parâmetros em kwargs
                        scoring = kwargs['scoring'] if 'scoring' in kwargs else 'neg_mean_squared_error'
                        cv = kwargs['cv'] if 'cv' in kwargs else 5
                        verbose = kwargs['verbose'] if 'verbose' in kwargs else -1
                        n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else -1
                        
                        # Preparando e aplicando busca
                        rnd_search = RandomizedSearchCV(model, params, scoring=scoring, cv=cv,
                                                        verbose=verbose, random_state=42, n_jobs=n_jobs)
                        logger.debug('Aplicando RandomizedSearchCV')
                        rnd_search.fit(X_train, y_train)

                        # Salvando melhor modelo no atributo classifiers_info
                        self.regressors_info[model_key]['estimator'] = rnd_search.best_estimator_
                    else:
                        # Treinando modelo sem busca e salvando no atirbuto
                        self.regressors_info[model_key]['estimator'] = model.fit(X_train, y_train)
                except TypeError as te:
                    logger.error(f'Erro ao aplicar RandomizedSearch. Exception lançada: {te}')
                    return

                # Validando salvamento de objetos pkl dos modelos
                if 'save' in kwargs and bool(kwargs['save']):
                    logger.debug(f'Salvando arquivo pkl do modelo {model_name} treinado')
                    model = self.regressors_info[model_key]['estimator']
                    output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/models')
                    model_ext = kwargs['model_ext'] if 'model_ext' in kwargs else 'pkl'

                    save_model(model, output_path=output_path, filename=model_name.lower() + '.' + model_ext)

        except AttributeError as ae:
            logger.error('Erro ao treinar modelos. Exception lançada: {ae}')
            logger.warning(f'Treinamento do(s) modelo(s) não realizado')
            
    def compute_train_performance(self, model_name, estimator, X, y, cv=5):
        """
        Método responsável por aplicar validação cruzada para retornar a amédia das principais métricas de avaliação
        de um modelo de regressão. Na prática, esse método é chamado por um outro método em uma camada
        superior da classe para medição de performance em treino e em teste

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo contida no atributo self.classifiers_info [type: string]
        :param estimator: estimator do modelo a ser avaliado [type: object]
        :param X: conjunto de features do modelo contido nos dados de treino [type: np.array]
        :param y: array contendo a variável resposta dos dados de treino do modelo [type: np.array]
        :param cv: K-folds utiliados na validação cruzada [type: int, default: 5]

        Retorno
        -------
        :return train_performance: DataFrame contendo as métricas calculadas usando validação cruzada [type: pd.DataFrame]

        Aplicação
        ---------
        # Instanciando e treinando modelo
        trainer = ClassificadorBinario()
        trainer.fit(model, X_train, y_train)
        train_performance = trainer.compute_train_performance(model_name, estimator, X_train, y_train)
        """

        # Computando métricas utilizando validação cruzada
        logger.debug(f'Computando métricas do modelo {model_name} utilizando validação cruzada com {cv} K-folds')
        try:
            t0 = time.time()
            mae = -(cross_val_score(estimator, X, y, cv=5, scoring='neg_mean_absolute_error')).mean()
            mse_scores = cross_val_score(estimator, X, y, cv=5, scoring='neg_mean_squared_error')
            mse = (-mse_scores).mean()
            rmse = np.sqrt(-mse_scores).mean()
            r2 = cross_val_score(estimator, X, y, cv=5, scoring='r2').mean()

            # Criando DataFrame com o resultado obtido
            t1 = time.time()
            delta_time = t1 - t0
            train_performance = {}
            train_performance['model'] = model_name
            train_performance['approach'] = f'Treino {cv} K-folds'
            train_performance['mae'] = round(mae, 3)
            train_performance['mse'] = round(mse, 3)
            train_performance['rmse'] = round(rmse, 3)
            train_performance['r2'] = round(r2, 3)
            train_performance['total_time'] = round(delta_time, 3)
            logger.info(f'Métricas computadas com sucesso nos dados de treino em {round(delta_time, 3)} segundos')

            return pd.DataFrame(train_performance, index=train_performance.keys()).reset_index(drop=True).loc[:0, :]

        except Exception as e:
            logger.error(f'Erro ao computar as métricas. Exception lançada: {e}')    

    def compute_val_performance(self, model_name, estimator, X, y):
        """
        Método responsável por aplicar retornar as principais métricas do modelo utilizando dados de validação.
        Na prática, esse método é chamado por um outro método em uma camada superior da classe para medição 
        de performance em treino e em teste

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo contida no atributo self.classifiers_info [type: string]
        :param estimator: estimator do modelo a ser avaliado [type: object]
        :param X: conjunto de features do modelo contido nos dados de validação [type: np.array]
        :param y: array contendo a variável resposta dos dados de validação do modelo [type: np.array]

        Retorno
        -------
        :return test_performance: DataFrame contendo as métricas calculadas nos dados de teste [type: pd.DataFrame]

        Aplicação
        ---------
        # Instanciando e treinando modelo
        trainer = ClassificadorBinario()
        trainer.fit(model, X_train, y_train)
        test_performance = trainer.compute_val_performance(model_name, estimator, X_val, y_val)
        """

        # Predicting data using the trained model and computing probabilities
        logger.debug(f'Computando métricas do modelo {model_name} utilizando dados de validação')
        try:
            t0 = time.time()
            y_pred = estimator.predict(X)

            # Retrieving metrics using validation data
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred)

            # Creating a DataFrame with metrics
            t1 = time.time()
            delta_time = t1 - t0
            test_performance = {}
            test_performance['model'] = model_name
            test_performance['approach'] = 'Validation set'
            test_performance['mae'] = round(mae, 3)
            test_performance['mse'] = round(mse, 3)
            test_performance['rmse'] = round(rmse, 3)
            test_performance['r2'] = round(r2, 3)
            test_performance['total_time'] = round(delta_time, 3)
            logger.info(f'Métricas computadas com sucesso nos dados de validação em {round(delta_time, 3)} segundos')

            return pd.DataFrame(test_performance, index=test_performance.keys()).reset_index(drop=True).loc[:0, :]

        except Exception as e:
            logger.error(f'Erro ao computar as métricas. Exception lançada: {e}')

    def evaluate_performance(self, X_train, y_train, X_val, y_val, cv=5, **kwargs):
        """
        Método responsável por executar e retornar métricas dos regressores em treino (média do resultado
        da validação cruzada com cv K-fols) e teste

        Parâmetros
        ----------
        :param X_train: conjunto de features do modelo contido nos dados de treino [type: np.array]
        :param y_train: array contendo a variável resposta dos dados de treino do modelo [type: np.array]
        :param X_val: conjunto de features do modelo contido nos dados de validação [type: np.array]
        :param y_val: array contendo a variável resposta dos dados de validação do modelo [type: np.array]
        :param cv: K-folds utiliados na validação cruzada [type: int, default: 5]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/metrics']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='metrics.csv']

        Retorno
        -------
        :return df_performances: DataFrame contendo as métricas calculadas em treino e teste [type: pd.DataFrame]

        Aplicação
        -----------
        # Treinando modelo e avaliando performance em treino e teste
        trainer = ClassificadorBinario()
        trainer.fit(estimator, X_train, X_test)

        # Definindo dicionário de controle do resultado
        df_performance = trainer.evaluate_performance(X_train, y_train, X_val, y_val, save=True, output_path=caminho)
        """

        # DataFrame vazio para armazenamento das métrics
        df_performances = pd.DataFrame({})

        # Iterando sobre todos os classificadores da classe
        for model_name, model_info in self.regressors_info.items():

            # Validando se o modelo já foi treinado (dicionário model_info já terá a chave 'train_performance')
            if 'train_performance' in model_info.keys():
                df_performances = df_performances.append(model_info['train_performance'])
                df_performances = df_performances.append(model_info['test_performance'])
                continue

            # Retornando modelo a ser avaliado
            try:
                estimator = model_info['estimator']
            except KeyError as e:
                logger.error(f'Erro ao retornar a chave "estimator" do dicionário model_info. Modelo {model_name} não treinado')
                continue

            # Computando performance em treino e em teste
            train_performance = self.compute_train_performance(model_name, estimator, X_train, y_train, cv=cv)
            val_performance = self.compute_val_performance(model_name, estimator, X_val, y_val)

            # Adicionando os resultados ao atributo classifiers_info
            self.regressors_info[model_name]['train_performance'] = train_performance
            self.regressors_info[model_name]['val_performance'] = val_performance

            # Construindo DataFrame com as métricas retornadas
            model_performance = train_performance.append(val_performance)
            df_performances = df_performances.append(model_performance)
            df_performances['anomesdia_datetime'] = datetime.now()

            # Salvando alguns atributos no dicionário classifiers_info para acessos futuros
            model_data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val
            }
            model_info['model_data'] = model_data

        # Validando salvamento dos resultados
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/metrics')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'metrics.csv'
            save_data(df_performances, output_path=output_path, filename=output_filename)

        return df_performances
    
    def feature_importance(self, features, top_n=-1, **kwargs):
        """
        Método responsável por retornar a importância das features de um modelo treinado
        
        Parâmetros
        ----------
        :param features: lista contendo as features de um modelo [type: list]
        :param top_n: parâmetro para filtragem das top n features [type: int, default=-1]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/metrics']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='top_features.csv']

        Retorno
        -------
        :return: all_feat_imp: pandas DataFrame com a análise de feature importance dos modelos [type: pd.DataFrame]
        """

        # Inicializando DataFrame vazio para armazenamento das feature importance
        feat_imp = pd.DataFrame({})
        all_feat_imp = pd.DataFrame({})

        # Iterando sobre os modelos presentes na classe
        for model_name, model_info in self.regressors_info.items():
            # Validando possibilidade de extrair a importância das features do modelo
            logger.debug(f'Extraindo importância das features para o modelo {model_name}')

            try:
                importances = model_info['estimator'].feature_importances_
            except KeyError as ke:
                logger.warning(f'Modelo {model_name} não treinado, sendo impossível extrair o método feature_importances_')
                continue
            except AttributeError as ae:
                logger.warning(f'Modelo {model_name} não possui o método feature_importances_')
                continue

            # Preparando o dataset para armazenamento das informações
            feat_imp['feature'] = features
            feat_imp['importance'] = importances
            feat_imp['model'] = model_name
            feat_imp['anomesdia_datetime'] = datetime.now()
            feat_imp.sort_values(by='importance', ascending=False, inplace=True)
            feat_imp = feat_imp.loc[:, ['model', 'feature', 'importance', 'anomesdia_datetime']]

            # Salvando essa informação no dicionário classifiers_info
            self.regressors_info[model_name]['feature_importances'] = feat_imp
            all_feat_imp = all_feat_imp.append(feat_imp)
            logger.info(f'Extração da importância das features concluída com sucesso para o modelo {model_name}')

        # Validando salvamento dos resultados
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/metrics')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'top_features.csv'
            save_data(all_feat_imp, output_path=output_path, filename=output_filename)
        
        return all_feat_imp
    
    def training_flow(self, set_regressors, X_train, y_train, X_val, y_val, features, **kwargs):
        """
        Método responsável por consolidar um fluxo completo de treinamento dos regressores, bem como
        o levantamento de métricas e execução de métodos adicionais para escolha do melhor modelo

        Parâmetros
        ----------
        :param set_regressors: dicionário contendo informações dos modelos a serem treinados [type: dict]
            set_regressors = {
                'model_name': {
                    'model': __estimator__,
                    'params': __estimator_params__
                }
            }
        :param X_train: conjunto de features do modelo contido nos dados de treino [type: np.array]
        :param y_train: array contendo a variável resposta dos dados de treino do modelo [type: np.array]
        :param X_test: conjunto de features do modelo contido nos dados de teste [type: np.array]
        :param y_test: array contendo a variável resposta dos dados de teste do modelo [type: np.array]
        :param features: lista contendo as features de um modelo [type: list]
        :param output_path: caminho onde o arquivo de resultados será salvo: [type: string, default=os.path.join(os.path.getcwd(), 'output/')]
        :param **kwargs: argumentos adicionais do método
            :arg approach: indicativo de sufixo para armazenamento no atributo classifiers_info [type: string, default: '']
            :arg random_search: flag para aplicação do RandomizedSearchCV [type: bool, default: False]
            :arg scoring: métrica a ser otimizada pelo RandomizedSearchCV [type: string, default: 'accuracy']
            :arg cv: K-folds utiliados na validação cruzada [type: int, default: 5]
            :arg verbose: nível de verbosity da busca aleatória [type: int, default: 5]
            :arg n_jobs: quantidade de jobs aplicados durante a busca dos hiperparâmetros [type: int, default: -1]
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg models_output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/models']
            :arg metrics_output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/metrics']
            :arg metrics_output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='metrics.csv']
            :arg featimp_output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='top_features.csv']
            :arg top_n_featimp: top features a serem analisadas na importância das features [type: int, default=-1]

        Retorno
        -------
        None

        Aplicação
        ---------
        # Instanciando objeto
        trainer = ClassificadorBinario()
        trainer.training_flow(set_classifiers, X_train, y_train, X_test, y_test, features)
        """

        # Definindo variáveis padrão para retorno dos resultados
        """if not os.path.isdir(output_path):
            os.makedirs(output_path)"""

        # Extraindo parâmetros kwargs
        approach = kwargs['approach'] if 'approach' in kwargs else ''
        random_search = kwargs['random_search'] if 'random_search' in kwargs else False
        scoring = kwargs['scoring'] if 'scoring' in kwargs else 'neg_mean_squared_error'
        cv = kwargs['cv'] if 'cv' in kwargs else 5
        verbose = kwargs['verbose'] if 'verbose' in kwargs else -1
        n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else -1
        save = bool(kwargs['save']) if 'save' in kwargs else True
        models_output_path = kwargs['models_output_path'] if 'models_output_path' in kwargs else os.path.join(os.getcwd(), 'output/models')
        metrics_output_path = kwargs['metrics_output_path'] if 'metrics_output_path' in kwargs else os.path.join(os.getcwd(), 'output/metrics')
        metrics_output_filename = kwargs['metrics_output_filename'] if 'metrics_output_filename' in kwargs else 'metrics.csv'
        featimp_output_filename = kwargs['featimp_output_filename'] if 'featimp_output_filename' in kwargs else 'top_features.csv'
        top_n_featimp = kwargs['top_n_featimp'] if 'top_n_featimp' in kwargs else -1

        # Treinando classificadores
        self.fit(set_regressors, X_train, y_train, approach=approach, random_search=random_search, scoring=scoring,
                 cv=cv, verbose=verbose, n_jobs=n_jobs, output_path=models_output_path)

        # Avaliando modelos
        self.evaluate_performance(X_train, y_train, X_val, y_val, save=save, output_path=metrics_output_path, 
                                  output_filename=metrics_output_filename)

        # Analisando Features mais importantes
        self.feature_importance(features, top_n=top_n_featimp, save=save, output_path=metrics_output_path, 
                                output_filename=featimp_output_filename)
        
    def plot_feature_importance(self, features, top_n=20, palette='viridis', **kwargs):
        """
        Método responsável por realizar uma plotagem gráfica das variáveis mais importantes pro modelo

        Parâmetros
        ----------
        :param features: lista de features do conjunto de dados [type: list]
        :param top_n: quantidade de features a ser considerada na plotagem [type: int, default=20]
        :param palette: paleta de cores utilizazda na plotagem [type: string, default='viridis']
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='feature_importances.png']

        Retorno
        -------
        Este método não retorna nada além da imagem devidamente salva no diretório destino
        """

        # Definindo parâmetros de plotagem
        logger.debug('Inicializando plotagem das features mais importantes para os modelos')
        feat_imp = pd.DataFrame({})
        i = 0
        ax_del = 0
        nrows = len(self.regressors_info.keys())
        fig, axs = plt.subplots(nrows=nrows, figsize=(16, nrows * 6))
        sns.set(style='white', palette='muted', color_codes=True)
        
        # Iterando sobre os modelos presentes na classe
        for model_name, model_info in self.regressors_info.items():
            # Validando possibilidade de extrair a importância das features do modelo
            logger.debug(f'Extraindo importância das features para o modelo {model_name}')
            try:
                importances = model_info['estimator'].feature_importances_
            except:
                logger.warning(f'Modelo {model_name} não possui o método feature_importances_')
                ax_del += 1
                continue
            
            # Preparando o dataset para armazenamento das informações
            feat_imp['feature'] = features
            feat_imp['importance'] = importances
            feat_imp.sort_values(by='importance', ascending=False, inplace=True)

            logger.debug(f'Plotando gráfico de importância das features para o modelo {model_name}')
            try:
                # Plotando feature importance
                sns.barplot(x='importance', y='feature', data=feat_imp.iloc[:top_n, :], ax=axs[i], 
                            palette=palette)

                # Customizando gráfico
                axs[i].set_title(f'Features Mais Importantes: {model_name}', size=14)
                format_spines(axs[i], right_border=False)
                i += 1
  
                logger.info(f'Gráfico de importância das features plotado com sucesso para o modelo {model_name}')
            except Exception as e:
                logger.error(f'Erro ao gerar gráfico de importância das features para o modelo {model_name}. Exception lançada: {e}')
                continue

        # Deletando eixos sobressalentes (se aplicável)
        if ax_del > 0:
            logger.debug('Deletando eixos referentes a análises não realizadas')
            try:
                for i in range(-1, -(ax_del+1), -1):
                    fig.delaxes(axs[i])
            except Exception as e:
                logger.error(f'Erro ao deletar eixo. Exception lançada: {e}')
        
        # Alinhando figura
        plt.tight_layout()

        # Salvando imagem
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'feature_importances.png'
            save_fig(fig, output_path=output_path, img_name=output_filename)   
    
    def plot_metrics(self, figsize=(16, 10), palette='rainbow', cv=5, **kwargs):
        """
        Método responsável por plotar os resultados das métricas dos regressores selecionados

        Parâmetros
        ----------
        :param figsize: dimensões da figura gerada para a plotagem [type: tuple, default=(16, 10)]
        :param palette: paleta de cores do matplotlib [type: string, default='rainbow']
        :param cv: K-folds utiliados na validação cruzada [type: int, default: 5]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='metrics_comparison.png']

        Retorno
        -------
        Este método não retorna nenhum parâmetro além da plotagem devidamente salva no diretório destino
        """

        logger.debug(f'Iniciando plotagem gráfica das métricas dos classificadores')
        metrics = pd.DataFrame()
        for model_name, model_info in self.regressors_info.items():

            logger.debug(f'Retornando métricas via validação cruzada para o modelo {model_name}')
            try:
                # Retornando variáveis do classificador
                metrics_tmp = pd.DataFrame()
                estimator = model_info['estimator']
                X = model_info['model_data']['X_train']
                y = model_info['model_data']['y_train']

                # Métricas não computadas
                mae = -(cross_val_score(estimator, X, y, cv=cv, scoring='neg_mean_absolute_error'))
                mse = -(cross_val_score(estimator, X, y, cv=cv, scoring='neg_mean_squared_error'))
                rmse = np.sqrt(mse)
                r2 = cross_val_score(estimator, X, y, cv=cv, scoring='r2')

                # Adicionando ao DataFrame recém criado
                metrics_tmp['mae'] = mae
                metrics_tmp['mse'] = mse
                metrics_tmp['rmse'] = rmse
                metrics_tmp['r2'] = r2
                metrics_tmp['model'] = model_name

                # Empilhando métricas
                metrics = metrics.append(metrics_tmp)
            except Exception as e:
                logger.error(f'Erro ao retornar as métricas para o modelo {model_name}. Exception lançada: {e}')
                continue

        logger.debug(f'Modificando DataFrame de métricas para plotagem gráfica')
        try:
            # Pivotando métricas (boxplot)
            index_cols = ['model']
            metrics_cols = ['mae', 'mse', 'rmse', 'r2']
            df_metrics = pd.melt(metrics, id_vars=index_cols, value_vars=metrics_cols)

            # Agrupando métricas (barras)
            metrics_group = df_metrics.groupby(by=['model', 'variable'], as_index=False).mean()
        except Exception as e:
            logger.error(f'Erro ao pivotar DataFrame. Exception lançada: {e}')
            return

        logger.debug(f'Plotando análise gráfica das métricas para os modelos treinados')
        x_rot = kwargs['x_rot'] if 'x_rot' in kwargs else 90
        bar_label = bool(kwargs['bar_label']) if 'bar_label' in kwargs else False
        try:
            # Criando figura de plotagem
            fig, axs = plt.subplots(nrows=2, ncols=4, figsize=figsize)
            
            i = 0
            for metric in metrics_cols:                
                # Definindo eixos
                ax0 = axs[0, i]
                ax1 = axs[1, i]
                
                # Plotando gráficos
                sns.boxplot(x='variable', y='value', data=df_metrics.query('variable == @metric'), 
                            hue='model', ax=ax0, palette=palette)
                sns.barplot(x='model', y='value', data=metrics_group.query('variable == @metric'), 
                            ax=ax1, palette=palette, order=list(df_metrics['model'].unique()))
            
                # Personalizando plotagem
                ax0.set_title(f'Boxplot {cv}K-folds: {metric}')
                ax1.set_title(f'Média {cv} K-folds: {metric}')
                
                format_spines(ax0, right_border=False)
                format_spines(ax1, right_border=False)
                
                if bar_label:
                    AnnotateBars(n_dec=3, color='black', font_size=12).vertical(ax1)
                
                if i < 3:
                    ax0.get_legend().set_visible(False)
                    
                for tick in ax1.get_xticklabels():
                    tick.set_rotation(x_rot)
            
                i += 1
        except Exception as e:
            logger.error(f'Erro ao plotar gráfico das métricas. Exception lançada: {e}')
            return

        # Alinhando figura
        plt.tight_layout()

        # Salvando figura
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'metrics_comparison.png'
            save_fig(fig, output_path=output_path, img_name=output_filename)      
            
    def plot_learning_curve(self, ylim=None, cv=5, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10), **kwargs):
        """
        Método responsável por calcular a curva de aprendizado para um modelo treinado
        
        Parâmetros
        ----------
        :param model_name: chave de referência para análise de um modelo já treinado[type: string]
        :param figsize: dimensões da figura de plotagem [type: tuple, default=(16, 6)]
        :param ylim: climite do eixo vertical [type: int, default=None]
        :param cv: k-folds utilizados na validação cruzada para levantamento de informações [type: int, default=5]
        :param n_jobs: número de processadores utilizado no levantamento das informações [type: int, default=1]
        :param train_sizes: array de passos utilizados na curva [type: np.array, default=np.linspace(.1, 1.0, 10)]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='confusion_matrix.png']

        Retorno
        -------
        Este método não retorna nenhum parâmetro além do salvamento do gráfico de distribuição especificado

        Aplicação
        -----------
        trainer.plot_learning_curve(model_name='LightGBM')
        """

        logger.debug(f'Inicializando plotagem da curvas de aprendizado dos modelos')
        i = 0
        nrows = len(self.regressors_info.keys())
        fig, axs = plt.subplots(nrows=nrows, figsize=(16, nrows * 6))

        # Iterando sobre os classificadores presentes na classe
        for model_name, model_info in self.regressors_info.items():
            ax = axs[i]
            logger.debug(f'Retornando parâmetros pro modelo {model_name} e aplicando método learning_curve')
            try:
                model = model_info['estimator']
                X_train = model_info['model_data']['X_train']
                y_train = model_info['model_data']['y_train']

                # Chamando função learning_curve para retornar os scores de treino e validação
                train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

                # Computando médias e desvio padrão (treino e validação)
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                val_scores_mean = np.mean(val_scores, axis=1)
                val_scores_std = np.std(val_scores, axis=1)
            except Exception as e:
                logger.error(f'Erro ao retornar parâmetros e scores pro modelo {model_name}. Exception lançada: {e}')
                continue

            logger.debug(f'Plotando curvas de aprendizado de treino e validação para o modelo {model_name}')
            try:
                # Resultados utilizando dados de treino
                ax.plot(train_sizes, train_scores_mean, 'o-', color='navy', label='Training Score')
                ax.fill_between(train_sizes, (train_scores_mean - train_scores_std), (train_scores_mean + train_scores_std),
                                alpha=0.1, color='blue')

                # Resultados utilizando dados de validação (cross validation)
                ax.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Cross Val Score')
                ax.fill_between(train_sizes, (val_scores_mean - val_scores_std), (val_scores_mean + val_scores_std),
                                alpha=0.1, color='crimson')

                # Customizando plotagem
                ax.set_title(f'Model {model_name} - Learning Curve', size=14)
                ax.set_xlabel('Training size (m)')
                ax.set_ylabel('Score')
                ax.grid(True)
                ax.legend(loc='best')
            except Exception as e:
                logger.error(f'Erro ao plotar curva de aprendizado para o modelo {model_name}. Exception lançada: {e}')
                continue
            i += 1
        
        # Alinhando figura
        plt.tight_layout()

        # Salvando imagem
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'learning_curve.png'
            save_fig(fig, output_path=output_path, img_name=output_filename) 

    def visual_analysis(self, features, metrics=True, feat_imp=True, show=False, save=True, 
                        learn_curve=True, output_path=os.path.join(os.getcwd(), 'output/imgs'), **kwargs):
        """
        Método responsável por consolidar análises gráficas no processo de modelagem

        Parâmetros
        ----------
        :param features: lista de features do conjunto de dados [type: list]
        :param metrics: flag inficativo da execução do método plot_metrics() [type: bool, default=True]
        :param feat_imp: flag indicativo da execução da método plot_feature_importance() [type: bool, default=True]
        :param cfmx: flag indicativo da execução da método plot_confusion_matrix() [type: bool, default=True]
        :param roc: flag indicativo da execução da método plot_roc_curve() [type: bool, default=True]
        :param score_dist: flag indicativo da execução da método plot_score_distribution() [type: bool, default=True]
        :param score_bins: flag indicativo da execução da método plot_score_bins() [type: bool, default=True]
        :param learn_curve: flag indicativo da execução da método plot_learning_curve() [type: bool, default=True]
        :param model_shap: chave do modelo a ser utilizado na análise shap [type: string, default=None]
        :param show: flag indicativo para mostragem das figuras em jupyter notebook [type: bool, default=False]
        :param save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
        :param output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']        

        Retorno
        -------
        Este método não retorna nada além das imagens devidamente salvas no diretório destino

        Aplicação
        ---------
        trainer = ClassificadorBinario()
        trainer.fit(set_classifiers, X_train, y_train, X_test, y_test)        
        """

        # Verificando parâmetro para mostrar
        backend_ = mpl.get_backend()
        if not show:
            mpl.use('Agg')

        # Verificando plotagem das métricas
        if metrics:
            try:        
                self.plot_metrics(save=save, output_path=output_path)
            except Exception as e:
                logger.error(f'Erro ao plotar métrics. Exception: {e}')
             
        # Verificando plotagem de feature importance
        if feat_imp:
            try:
                self.plot_feature_importance(features=features, save=save, output_path=output_path)
            except Exception as e:
                logger.error(f'Erro ao plotar feature importances. Exception: {e}')

        # Verificando plotagem de curva de aprendizado
        if learn_curve:
            try:
                self.plot_learning_curve(save=save, output_path=output_path)
            except Exception as e:
                logger.error(f'Erro ao plotar curva de aprendizado. Exception: {e}')

        # Resetando configurações
        mpl.use(backend_)

    def get_estimator(self, model_name):
        """
        Método responsável por retornar o estimator de um modelo selecionado

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo no dicionário regressors_info da classe [type: string]

        Retorno
        -------
        :return model: estimator do modelo já treinado
        """

        logger.debug(f'Retornando estimator do modelo {model_name} já treinado')
        try:
            model_info = self.regressors_info[model_name]
            return model_info['estimator']
        except Exception as e:
            logger.error(f'Classificador {model_name} não existente ou não treinado. Opções possíveis: {list(self.classifiers_info.keys())}')
            return

    def get_metrics(self, model_name):
        """
        Método responsável por retornar as métricas obtidas no treinamento

        Parâmetros
        ----------
        None

        Retorno
        -------
        :return model_performance: DataFrame contendo as métricas dos modelos treinados [type: pd.DataFrame]
        """

        logger.debug(f'Retornando as métricas do modelo {model_name}')
        try:
            # Retornando dicionário do modelo e métricas já salvas
            model_info = self.regressors_info[model_name]
            train_performance = model_info['train_performance']
            test_performance = model_info['test_performance']
            model_performance = train_performance.append(test_performance)
            model_performance.reset_index(drop=True, inplace=True)

            return model_performance
        except Exception as e:
            logger.error(f'Erro ao retornar as métricas para o modelo {model_name}. Exception lançada: {e}')

    def get_model_info(self, model_name):
        """
        Método responsável por coletar as informações registradas de um determinado modelo da classe

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo no dicionário regressors_info da classe [type: string]

        Retorno
        -------
        :return model_info: dicionário com informações registradas do modelo [type: dict]
            model_info = {
                'estimator': model,
                'train_performance': pd.DataFrame,
                'test_performance': pd.DataFrame,
                'model_data': {
                    'X_train': np.array,
                    'y_train': np.array,
                    'X_test': np.array,
                    'y_test': np.array,
                'feature_importances': pd.DataFrame
                }
            }
        """

        logger.debug(f'Retornando informações registradas do modelo {model_name}')
        try:
            # Retornando dicionário do modelo
            return self.regressors_info[model_name]
        except Exception as e:
            logger.error(f'Erro ao retornar informações do modelo {model_name}. Exception lançada: {e}')

    def get_regressors_info(self):
        """
        Método responsável por retornar o dicionário regressors_info contendo todas as informações de todos os modelos

        Parâmetros
        ----------
        None

        Retorno
        -------
        :return regressors_info: dicionário completo dos modelos presentes na classe
            regressors_info ={
                'model_name': model_info = {
                                'estimator': model,
                                'train_performance': pd.DataFrame,
                                'test_performance': pd.DataFrame,
                                'model_data': {
                                    'X_train': np.array,
                                    'y_train': np.array,
                                    'X_test': np.array,
                                    'y_test': np.array,
                                'feature_importances': pd.DataFrame
                                }
                            }
        """

        return self.regressors_info
