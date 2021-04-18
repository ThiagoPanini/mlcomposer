"""
---------------------------------------------------
------- Exemplos de uso - Regressão Linear --------
---------------------------------------------------
Script responsável por consolidar exemplos de
aplicação relacionadas as funcionalidades presentes
no módulo ml da biblioteca mlcomposer

Sumário
---------------------------------------------------
1. Configuração inicial
    1.1 Importando bibliotecas
    1.2 Definição de variáveis do projeto
2. Preparação da base de dados
    2.1 Leitura das bases de treino e teste
    2.2 Construindo pipelines de preparação
    2.3 Aplicação dos pipelines de preparação
3. Treinamento e avaliação de modelos
    3.1 Estruturando objetos de modelagem
    3.2 Fluxos de treinamento e avaliação
---------------------------------------------------
"""

# Autor: Thiago Panini
# Data: 13/04/2021


"""
---------------------------------------------------
------------ 1. CONFIGURAÇÃO INICIAL --------------
           1.1 Importando bibliotecas
---------------------------------------------------
"""

# Bibliotecas padrão
import pandas as pd
import numpy as np
import os
from warnings import filterwarnings
filterwarnings('ignore')

# Variáveis de ambiente
from dotenv import load_dotenv, find_dotenv

# Preparação dos dados
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from ml.transformers import FiltraColunas, ModificaTipoPrimitivo, \
    EliminaDuplicatas, DynamicLogTransformation, DynamicScaler, \
    DummiesEncoding, AgrupamentoCategoricoFinal, LogTransformation

# Modelagem dos dados
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from ml.trainer import RegressorLinear


"""
------------------------------------------------------
-------------- 1. CONFIGURAÇÃO INICIAL ---------------
        1.2 Definição de variáveis do projeto
------------------------------------------------------ 
"""

# Lendo variáveis de ambiente
load_dotenv(find_dotenv())

# Definindo variáveis de diretório
DATA_PATH = os.getenv('LIN_REG_DATA_PATH')
TRAIN_PATH = os.path.join(DATA_PATH, 'train.csv')
TEST_PATH = os.path.join(DATA_PATH, 'test.csv')
OUTPUT_PATH = os.path.join(os.getcwd(), os.path.dirname(__file__), 'output')
MODELS_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'models')
METRICS_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'metrics')
IMGS_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'imgs')

# Leitura das bases para definições de variáveis do projeto
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Definindo variáveis do projeto
TARGET = 'SalePrice'
TO_DROP = ['Condition2', 'RoofMatl', 'Id']
INITIAL_FEATURES = [col for col in train.columns if col not in TO_DROP]
CAT_GROUP_DICT = {'Functional': ['Typ', 'Min2', 'Min1'],
                  'SaleType': ['WD', 'New', 'COD'],
                  'HouseStyle': ['1Story', '2Story', '1.5Fin', 'SLvl', 'SFoyer'],
                  'Condition1': ['Norm', 'Feedr', 'Artery', 'RRAn', 'PosN'],
                  'Neighborhood': ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt', 'Sawyer'],
                  'Exterior2nd': ['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'Plywood'],
                  'Exterior1st': ['VinylSd', 'HdBoard', 'MetalSd', 'Wd Sdng', 'Plywood']}
OTHER_TAG = 'Other'
COLS_TO_LOG = ['MSSubClass', 'LotArea', 'OverallCond', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
               'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
               'BsmtHalfBath', 'HalfBath', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF',
               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

# Separando features por tipo primitivo
NUM_FEATURES = [col for col, dtype in train[INITIAL_FEATURES].dtypes.items() if dtype != 'object' and col != TARGET]
CAT_FEATURES = [col for col, dtype in train[INITIAL_FEATURES].dtypes.items() if dtype == 'object' and col != TARGET]
    

"""
------------------------------------------------------
----------- 2. PREPARAÇÃO DA BASE DE DADOS -----------
        2.2 Construindo pipelines de preparação
------------------------------------------------------ 
"""

# Pipeline inicial de preparação de dados
initial_prep_pipeline = Pipeline([
    ('col_filter', FiltraColunas(features=INITIAL_FEATURES)),
    ('cat_agrup', AgrupamentoCategoricoFinal(cat_dict=CAT_GROUP_DICT, other_tag=OTHER_TAG)),
    ('log_target', LogTransformation(cols_to_log=TARGET))
])

# Pipeline numérico
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('log', DynamicLogTransformation(num_features=NUM_FEATURES, cols_to_log=COLS_TO_LOG)),
    ('scaler', DynamicScaler(scaler_type=None))
])

# Pipeline categórico
cat_pipeline = Pipeline([
    ('encoder', DummiesEncoding(dummy_na=True))
])

# Pipeline completo
prep_pipeline = ColumnTransformer([
    ('num', num_pipeline, NUM_FEATURES),
    ('cat', cat_pipeline, CAT_FEATURES)
])


"""
------------------------------------------------------
----------- 2. PREPARAÇÃO DA BASE DE DADOS -----------
      2.3 Aplicação dos pipelines de preparação
------------------------------------------------------ 
"""

# Pipeline inicial
train_prep = initial_prep_pipeline.fit_transform(train)

# Criando base de treino e de validação
X_train, X_val, y_train, y_val = train_test_split(train_prep.drop(TARGET, axis=1), 
                                                  train_prep[TARGET].values, test_size=.20, random_state=42)

# Aplicando pipeline de preparação (treino)
X_train_prep = prep_pipeline.fit_transform(X_train)
train_features = prep_pipeline.named_transformers_['cat'].named_steps['encoder'].features_after_encoding

# Aplicando pipeline de preparação (validação)
X_val_prep = prep_pipeline.fit_transform(X_val)
val_features = prep_pipeline.named_transformers_['cat'].named_steps['encoder'].features_after_encoding

# Resultados
print(f'Shape of X_train_prep: {X_train_prep.shape}')
print(f'Shape of X_val_prep: {X_val_prep.shape}')


"""
------------------------------------------------------
----------- 2. PREPARAÇÃO DA BASE DE DADOS -----------
         2.4 Alinhando features após encoding
------------------------------------------------------ 
"""

# Relação de entradas não inclusas em teste
not_included_val = [col for col in train_features if col not in val_features]

# Retornando features do modelo
X_train_prep_df = pd.DataFrame(X_train_prep, columns=NUM_FEATURES+train_features)
MODEL_FEATURES = list(X_train_prep_df.columns)
    
# Alinhando conjunto de teste (preenchendo com 0)
X_val_prep_df = pd.DataFrame(X_val_prep, columns=NUM_FEATURES+val_features)
for col in not_included_val:
    X_val_prep_df[col] = 0
X_val_prep = np.array(X_val_prep_df.loc[:, MODEL_FEATURES])

# Verificando
print(f'\nShape of new X_train_prep: {X_train_prep.shape}')
print(f'Shape of new X_val_prep: {X_val_prep.shape}')

# Validando bases
print(f'Total of model features: {len(MODEL_FEATURES)}')

# Ordering X_val
X_val_prep_df = X_val_prep_df.loc[:, MODEL_FEATURES]

# Looking at the last features for each set
print(f'\nLast 5 features of X_train_prep: \n{X_train_prep_df.iloc[:, -5:].columns}')
print(f'\nLast 5 features of X_val_prep: \n{X_val_prep_df.iloc[:, -5:].columns}')

# Transforming into an array
X_train_prep = np.array(X_train_prep_df)
X_val_prep = np.array(X_val_prep_df)

# Results
print(f'\nShape of final X_train_prep: {X_train_prep.shape}')
print(f'Shape of final X_val_prep: {X_val_prep.shape}')


"""
------------------------------------------------------
------- 3. TREINAMENTO E AVALIAÇÃO DE MODELOS --------
        3.1 Estruturando objetos de modelagem
------------------------------------------------------ 
"""

# Instanciando objetos
linreg = LinearRegression()
dtree = DecisionTreeRegressor(random_state=42)
forest = RandomForestRegressor(random_state=42)
lasso = Lasso()
ridge = Ridge()
enet = ElasticNet()


# Definindo hiperparâmetros de busca
lin_reg_params = {
    'fit_intercept': [True, False],
    'normalize': [True, False]
}

tree_reg_params = {
    'max_depth': [100, 200, 300, 350, 400, 500],
    'max_features': np.arange(1, len(MODEL_FEATURES)),
    'random_state': [42]
}

forest_reg_params = {
    'n_estimators': [75, 90, 100, 200, 300, 400, 450, 500], 
    'max_features': np.arange(1, len(MODEL_FEATURES)),
    'random_state': [42]
}

ridge_reg_params = {
    'alpha': np.linspace(1e-5, 20, 400),
    'fit_intercept': [True, False],
    'normalize': [True, False]
}

lasso_reg_params = {
    'alpha': np.linspace(1e-5, 20, 400),
    'fit_intercept': [True, False],
    'normalize': [True, False]
}

elastic_reg_params = {
    'alpha': np.linspace(1e-5, 20, 400),
    'l1_ratio': np.linspace(0, 1, 400),
    'fit_intercept': [True, False],
    'normalize': [True, False]
}


# Criando dicionário set_classifiers
model_obj = [linreg, dtree, lasso, ridge, enet]
model_names = [type(model).__name__ for model in model_obj]
model_params = [lin_reg_params, tree_reg_params, lasso_reg_params,
                ridge_reg_params, elastic_reg_params]
set_regressors = {name: {'model': obj, 'params': param} for (name, obj, param) in zip(model_names, model_obj, model_params)}


"""
------------------------------------------------------
------- 3. TREINAMENTO E AVALIAÇÃO DE MODELOS --------
        3.2 Fluxos de treinamento e avaliação
------------------------------------------------------ 
"""

# Instanciando trainer e aplicando treinamento
trainer = RegressorLinear()

# Realizando treinamento e avaliação dos modelos
trainer.training_flow(set_regressors, X_train_prep, y_train, X_val_prep, y_val, 
                      features=MODEL_FEATURES, metrics_output_path=METRICS_OUTPUT_PATH,
                      models_output_path=MODELS_OUTPUT_PATH)

# Gerando e salvando gráficos de análises visuais
trainer.visual_analysis(features=MODEL_FEATURES, output_path=IMGS_OUTPUT_PATH)