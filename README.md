<h1 align="center">
  <a href="https://pypi.org/project/mlcomposer/"><img src="https://i.imgur.com/MIcPH8g.png" alt="mlcomposer logo"></a>
</h1>

<div align="center">
  <strong>:robot: Applying Machine Learning in an easy and efficient way like you never did before! :robot:</strong>
</div>
<br/>

<div align="center">  
  
  ![Release](https://img.shields.io/badge/release-ok-brightgreen)
  [![PyPI](https://img.shields.io/pypi/v/mlcomposer?color=blueviolet)](https://pypi.org/project/mlcomposer/)
  ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mlcomposer?color=green)
  ![PyPI - Status](https://img.shields.io/pypi/status/mlcomposer)

</div>
<br/>


## Table of content

- [About mlcomposer](#about-xplotter)
- [Package Structure](#package-structure)
  - [Transformers Module](#transformers-module)
  - [Trainer Module](#trainer-module) 


___

## About mlcomposer

Have ever tried to apply Machine Learning for solving a problem and found yourself lost on how much code you needed in order to reach your goal? Have you ever though it was too hard to do it? Don't be afraid because this package certainly can help you improving your skills and your modeling flow. Meet **mlcomposer** as a useful python package for helping users to use some built in classes and functions for applying Machine Learning as easy as possible. 

With **mlcomposer**, you can:
* Build data prep solutions with custom pipelines by using python classes that handle transformation steps
* Train and extract all information you need from Machine Learning basic tasks like classification and regression
* Build and visualize custom evaluation charts like performance reports, confusion matrix, ROC curves and others
* Save models in pkl or joblib formats after applying hyperparameter optimization in only one method execution
* Compare performances for multiple models at once
* Much more...

___

## Package Structure

The package is built around two main modules called `transformers` and `trainer`. The first one contains custom python classes written strategically for improving constructions of pipelines using native sklearn's class `Pipeline`. The second one is a powerful tool for training and evaluating Machine Learning models with classes for each different task (binary classification, multiclass classification and regression at this time). We will dive deep into those pieces on this documentation and I'm sure you will like it!

### Transformers Module

As said on the top of this section, the `transformers` module allocates custom classes with useful transformations to be applied on data prep pipelines. In order to provide the opportunity to integrate these classes into a preparation pipeline using sklearn's `Pipeline`, every class on transformers module inherits `BaseEstimator` and `TransformerMixin` from sklearn. This way, the code for data transformation itself is written inside a `transform()` method in each class, giving the chance for execute a sequence of steps outside the module in a more complex Pipeline and also the possibility to use user custom classes on this preparation flow.

If things are still a little complicated, the table below contains all classes built inside `transformers` module. After that, it will be placed an example of a data prep pipeline written using some of those classes.

| Class                       | Short Description                                                                                     |
| :-------------------------: | :---------------------------------------------------------------------------------------------------: |         
| `ColumnFormatter`           | Applies a custom column formatting in a pandas DataFrame to standardize column names                  |
| `ColumnSelection`           | Filters columns in a DataFrame based on a list passed as a class attribute                            |
| `BinaryTargetMapping`       | Transforms a raw target column in a binary one (1 or 0) based on a positive class argument            |
| `DropDuplicates`            | Drops duplicated rows in a dataset                                                                    |
| `DataSplitter`              | Applies a separation on data and creates new sets using `train_test_split()` function                 |
| `CategoricalLimitter`       | Limits entries in categorical columns and restrict them based on a "n_cat" argument                   |
| `CategoricalMapper`         | Receives a dictionary for mapping entries on categorical columns                                      |
| `DtypeModifier`             | Changes dtypes for columns based on a "mod_dict" argument                                             |
| `DummiesEncoding`           | Applies dummies encoding process (OrdinalEncoder using `pd.get_dummies()` method)                     |
| `FillNullData`              | Fills null data (at just only columns, if needed)                                                     |
| `FeatureSelection`          | Uses feature importance analysis for selecting top k features for the model                           |
| `LogTransformation`         | Applies `log1p()` from numpy in order to log transform all numerical data                             |
| `DynamicLogTransformation`  | Contains a boolean flag for applying or not log transformation (can be tunned afterall)               |
| `DynamicScaler`             | Applies normalization on data (`StandardScaler` or `MinMaxScaler` based on an application flag        |
| `ModelResults`              | Receives a set of features and an estimator for building a DataFrame with source data and predictions |

With table above, we can imagine a dot of custom transformations that can be applied one by one on data preparation pipelines. The snippet below simulates a pipeline construction with some of those classes. The idea is to create a block of code that automatically fills null data with a dummy number, applies log transformation and scaler on numerical features and finally applies dummies encoding on categorical features. Let's see how it can be built.

```python
from mlcomposer.transformers import FillNulldata, DynamicLogTransformation, DynamicScaler
from sklearn.pipeline import Pipeline

# Building a numerical pipeline
num_pipeline = Pipeline([
    ('imputer', FillNullData(value_fill=-999)),
    ('log_transformer', DynamicLogTransformation(application=True, num_features=NUM_FEATURES, 
                                                 cols_to_log=COLS_TO_LOG)),
    ('scaler', DynamicScaler(scaler_type='Standard'))
])

# Building a categorical pipeline
cat_pipeline = Pipeline([
    ('encoder', DummiesEncoding(dummy_na=True, cat_features_final=CAT_FEATURES_FINAL))
])

# Building a complete pipeline
prep_pipeline = ColumnTransformer([
    ('num', num_pipeline, NUM_FEATURES),
    ('cat', cat_pipeline, CAT_FEATURES)
])
```

Done! With few steps and some python tricks, it was possible to build a resilient pipeline for handling all transformation steps specified. The best from it is that it was not necessary to build up those custom classes as long as the `transformers` module did it for us.
___

### Trainer Module
