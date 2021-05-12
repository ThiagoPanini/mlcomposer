<h1 align="center">
  <a href="https://pypi.org/project/mlcomposer/"><img src="https://i.imgur.com/MIcPH8g.png" alt="mlcomposer logo"></a>
</h1>

<div align="center">
  <strong>:robot: Applying Machine Learning in an easy and efficient way like you never did before! :robot:</strong>
</div>
<br/>

<div align="center">  
 
  [![PyPI](https://img.shields.io/pypi/v/mlcomposer?color=blueviolet)](https://pypi.org/project/mlcomposer/)
  ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mlcomposer?color=green)
  ![PyPI - Status](https://img.shields.io/pypi/status/mlcomposer)

</div>


<div align="center">  
  
  ![Downloads](https://img.shields.io/pypi/dm/mlcomposer?color=darkblue)
  ![Downloads](https://img.shields.io/pypi/dw/mlcomposer?color=blue)
  ![Downloads](https://img.shields.io/pypi/dd/mlcomposer?color=lightblue)

</div>
<br/>


## Table of contents

- [About mlcomposer](#about-mlcomposer)
- [Package Structure](#package-structure)
  - [Transformers Module](#transformers-module)
  - [Trainer Module](#trainer-module) 
- [Examples](#examples)
- [Usage around the world](#usage-around-the-world)
- [Contribution](#contribution)
- [Social Media](#social-media)

___

## About mlcomposer

Have you ever tried to apply Machine Learning for solving a problem and found yourself lost on how much code you needed in order to reach your goal? Have you ever though it was too hard to do it? Don't be afraid because this package certainly can help you improving your skills and your modeling flow. Meet **mlcomposer** as a useful python package for helping users to use some built in classes and functions for applying Machine Learning as easy as possible. 

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

Well, by now what we can do after designing a complete pipeline for data preparation step? The answear could be straightforward: train and evaluate Machine Learning models! And that's the point the `trainer` module from `mlcomposer` gets in. With `trainer`, we can use classes already built for handling common steps for training and evaluating models considering basic ML approaches like binary and multiclass classification or linear regression.

So, it's possible to define the `trainer` module as a extremely poweful tool that gives the user the hability to solve complex problems and generates useful analysis with few lines of codes wherever there's a basic ML problem. It provides classes that encapsulates most of all steps needed for completing the solution. For make it cleaner, the table below shows up what kind of problems `trainer` module can solves or, in other words, what classes are built into the module.

| Class                       | Short Description                                                                                                          |
| :-------------------------: | :------------------------------------------------------------------------------------------------------------------------: |         
| `BinaryClassifier`          | Solves binary classification problems by delivering useful methods for training and evaluating multiple models at once     |
| `MulticlassClassifier`      | Solves multiclass classification problems by delivering useful methods for training and evaluating multiple models at once |
| `LinearRegressor`           | Solves linear regression problems by delivering useful methods for training and evaluating multiple models at once         |

Below it will be place an example of `trainer` module usage by passing through a complete flow of a binary classification problem. Two powerful methods of `BinaryClassifier` class will be applied in order to train, evaluate, extract feature importance, plotting confusion matrix, ROC Curve, score distribution and other visual analysis: the `training_flow()` and `visual_analysis()` methods. Take a look:

```python
#[...] other imports, variable definition and data pre process
from mlcomposer.trainer import BinaryClassifier

# Creating a object
trainer = BinaryClassifier()

# Training and evaluating models
trainer.training_flow(set_classifiers, X_train_prep, y_train, X_val_prep, y_val, 
                      features=MODEL_FEATURES, metrics_output_path=METRICS_OUTPUT_PATH,
                      models_output_path=MODELS_OUTPUT_PATH)

# Generating and saving figures for visual analysis
trainer.visual_analysis(features=MODEL_FEATURES, model_shap=MODEL_SHAP, 
                        output_path=IMGS_OUTPUT_PATH)
```

The simple execution of these two methods on binary classification problems can generate the following output:

```bash
└── output
    ├── imgs
    │   ├── confusion_matrix.png
    │   ├── feature_importances.png
    │   ├── learning_curve.png
    │   ├── metrics_comparison.png
    │   ├── roc_curve.png
    │   ├── score_bins_percent.png
    │   ├── score_bins.png
    │   ├── score_distribution.png
    │   └── shap_analysis_modelname.png
    ├── metrics
    │   ├── metrics.csv
    │   └── top_features.csv
    └── models
        ├── modelname_date.pkl
```

___

## Installing the Package

The last version of `mlcomposer` package are published and available on [PyPI repository](https://pypi.org/project/mlcomposer/)

> :pushpin: **Note:** as a good practice for every Python project, the creation of a <a href="https://realpython.com/python-virtual-environments-a-primer/">virtual environment</a> is needed to get a full control of dependencies and third part packages on your code. By this way, the code below can be used for creating a new venv on your OS.
> 

```bash
# Creating and activating venv on Linux
$ python -m venv <path_venv>/<name_venv>
$ source <path_venv>/<nome_venv>/bin/activate

# Creating and activating venv on Windows
$ python -m venv <path_venv>/<name_venv>
$ <path_venv>/<nome_venv>/Scripts/activate
```

With the new venv active, all you need is execute the code below using pip for installing xplotter package (upgrading pip is optional):

```bash
$ pip install --upgrade pip
$ pip install mlcomposer
```

The mlcomposer package is built in a layer above some other python packages like pandas, numpy, sklearn and shap. Because of that, when installing mlcomposer, the pip utility will also install all dependencies linked to the package.

## Examples

For making the package usage easy as possible for new users, it's placed on this Github repository a direcotry identified as `examples/` with python scripts with complete examples using `transformers` and `trainer` modules for solving differente Machine Learning problems with different approachs. 

## Usage Around the World

* For being easy and simple, mlcomposer can have a lot of applications. One of the most famous one is the notebook [Titanic Dataset Exploration](https://www.kaggle.com/thiagopanini/exploring-and-predicting-survival-on-titanic/comments) posted on Kaggle by [Thiago Panini](https://www.kaggle.com/thiagopanini). This well written notebook uses the insight module for plotting beautiful charts and building a really complete Exploratory Data Analysis proccess and, by now, it achieved a <b>bronze medal</b> with 34 upvotes by Kaggle's community and a incredible mark of more than 1,600 views!

<div align="center">
   <img src="https://i.imgur.com/J7ZcTD9.png">
</div>

* Another example is the Kaggle's notebook [Exploring and Modeling Housing Prices](https://www.kaggle.com/thiagopanini/exploring-and-modeling-housing-prices) built by [Thiago Panini](https://www.kaggle.com/thiagopanini) with `xplotter` and `mlcomposer` for extracting useful insights from housing prices data and also building a complete workflow for training and evaluating linear regression models. The notebook achieved a **silver medal** with 32 upvotes by Kaggle's community and almost 2,000 views!

<div align="center">
   <img src="https://i.imgur.com/COqermc.png">
</div>

* Still on Kaggle, there is the excellent [Presenting xplotter and mlcomposer on TPS May21]() that is part of the Tabular Playground Series competition that allows users to user data to explore techniques and knowlege. On the linked notebook, the user [Thiago Panini](https://www.kaggle.com/thiagopanini) explored some tools of `xplotter` and `mlcomposer` for building up an analysis of a Multiclass Classification model. People seems to like those packages: this recent notebook achieved a **bronze medal** with 16 upvotes with just 143 views! A good upvote rate!

<div align="center">
   <img src="https://i.imgur.com/zRQy5qI.png">
</div>

_Obs: the views, upvotes and maybe the medals information may be different depending on when you are looking at this README_

___

## Contribution

The mlcomposer python package is an open source implementation and the more people use it, the more happy the developers will be. So if you want to contribute with mlcomposer, please feel free to follow the best practices for implementing coding on this github repository through creating new branches, making merge requests and pointig out whenever you think there is a new topic to explore or a bug to be fixed.

Thank you very much for reaching this and it will be a pleasure to have you as mlcomposer user or developer.

___

## Social Media

* Follow me on LinkedIn: https://www.linkedin.com/in/thiago-panini/
* See my other Python packages: https://github.com/ThiagoPanini
