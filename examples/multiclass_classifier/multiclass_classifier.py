"""
---------------------------------------------------
--- Exemplos de uso - Classificação Multiclasse ---
---------------------------------------------------
Script responsável por consolidar exemplos de
aplicação relacionadas as funcionalidades presentes
no módulo ml da biblioteca mlcomposer

Sumário
---------------------------------------------------
1. Configuração inicial
    1.1 Importando bibliotecas
    1.2 Definição de variáveis do projeto
2. Treinamento e avaliação de modelos
    2.1 Estruturando objetos de modelagem
    2.2 Fluxos de treinamento e avaliação
---------------------------------------------------
"""

# Autor: Thiago Panini
# Data: 17/04/2021


"""
---------------------------------------------------
------------ 1. CONFIGURAÇÃO INICIAL --------------
           1.1 Importando bibliotecas
---------------------------------------------------
"""

# Bibliotecas padrão
import pandas as pd
import os
from warnings import filterwarnings
filterwarnings('ignore')

# Variáveis de ambiente
from dotenv import load_dotenv, find_dotenv

# Preparação dos dados
from sklearn.datasets import load_iris

# Modelagem dos dados
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from mlcomposer.trainer import ClassificadorMulticlasse


"""
------------------------------------------------------
-------------- 1. CONFIGURAÇÃO INICIAL ---------------
        1.2 Definição de variáveis do projeto
------------------------------------------------------ 
"""

# Definindo variáveis de diretório
OUTPUT_PATH = os.path.join(os.getcwd(), os.path.dirname(__file__), 'output')
MODELS_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'models')
METRICS_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'metrics')
IMGS_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'imgs')

# Definição de modelo para análise shap
MODEL_SHAP = 'RandomForestClassifier'


"""
------------------------------------------------------
----------- 2. PREPARAÇÃO DA BASE DE DADOS -----------
        2.1 Leitura das bases de treino e teste
------------------------------------------------------ 
"""

# Leitura das bases
iris = load_iris()
X = iris.data
y = iris.target

# Separando dados em treino e teste
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.20, random_state=42)

# Separando variáveis a partir do dicionário lido
TARGET_NAMES = iris['target_names']
MODEL_FEATURES = iris['feature_names']


"""
------------------------------------------------------
------- 2. TREINAMENTO E AVALIAÇÃO DE MODELOS --------
        2.1 Estruturando objetos de modelagem
------------------------------------------------------ 
"""

# Instanciando objetos
dtree = DecisionTreeClassifier(random_state=42)
forest = RandomForestClassifier(random_state=42)
svm = SVC(kernel='rbf', probability=True)

# Criando dicionário set_classifiers
model_obj = [svm, dtree, forest]
model_names = [type(model).__name__ for model in model_obj]
set_classifiers = {name: {'model': obj, 'params': {}} for (name, obj) in zip(model_names, model_obj)}


"""
------------------------------------------------------
------- 2. TREINAMENTO E AVALIAÇÃO DE MODELOS --------
        2.2 Fluxos de treinamento e avaliação
------------------------------------------------------ 
"""

# Instanciando novo objeto
trainer = ClassificadorMulticlasse()

# Realizando treinamento e avaliação dos modelos
trainer.training_flow(set_classifiers, X_train, y_train, X_val, y_val, target_names=TARGET_NAMES,
                      features=MODEL_FEATURES, metrics_output_path=METRICS_OUTPUT_PATH,
                      models_output_path=MODELS_OUTPUT_PATH)

# Gerando e salvando gráficos de análises visuais
trainer.visual_analysis(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, features=MODEL_FEATURES, 
                        target_names=TARGET_NAMES, output_path=IMGS_OUTPUT_PATH)
