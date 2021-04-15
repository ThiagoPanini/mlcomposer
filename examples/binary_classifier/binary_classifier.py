"""
---------------------------------------------------
------- Exemplos de utilização - mlcomposer -------
---------------------------------------------------
Script responsável por consolidar exemplos de
aplicação relacionadas as funcionalidades presentes
no módulo ml da biblioteca mlcomposer

Sumário
-----------------------------------

-----------------------------------
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
        DummiesEncoding

# Modelagem dos dados
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from ml.trainer import ClassificadorBinario


"""
------------------------------------------------------
-------------- 1. CONFIGURAÇÃO INICIAL ---------------
        1.2 Definição de variáveis do projeto
------------------------------------------------------ 
"""

# Lendo variáveis de ambiente
load_dotenv(find_dotenv())

# Definindo variáveis de diretório
DATA_PATH = os.getenv('DATA_PATH')
TRAIN_PATH = os.path.join(DATA_PATH, 'train.csv')
TEST_PATH = os.path.join(DATA_PATH, 'test.csv')
OUTPUT_PATH = os.path.join(os.getcwd(), os.path.dirname(__file__), 'output')
MODELS_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'models')
METRICS_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'metrics')
IMGS_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'imgs')

# Definindo variávies relacionadas as features iniciais do modelo
TARGET = 'Survived'
INITIAL_FEATURES = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
INITIAL_PRED_FEATURES = [col for col in INITIAL_FEATURES if col not in TARGET]

# Dicionário para modificação de tipos primitivos
DTYPE_MODIFICATION_DICT = {'Pclass': str} # Transforma os elementos de 'Pclass' em <str>

# Separação de features numéricas e categóricas
NUM_FEATURES = ['Age', 'SibSp', 'Parch', 'Fare']
CAT_FEATURES = ['Pclass', 'Sex', 'Embarked']

# Features do modelo após o processo de encoding
CAT_FEATURES_FINAL = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Pclass_nan', 'Sex_female', 'Sex_male', 'Sex_nan',
                      'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Embarked_nan']

# Definindo set final de features do modelo
MODEL_FEATURES = NUM_FEATURES + CAT_FEATURES_FINAL

# Variáveis de transformação no pipeline numérico
NUM_STRATEGY_IMPUTER = 'median'
SCALER_TYPE = None
LOG_APPLICATION = False
COLS_TO_LOG = ['Fare', 'Age']

# Variáveis de pipeline categórico
ENCODER_DUMMY_NA = True

# Definição de modelo para análise shap
MODEL_SHAP = 'RandomForestClassifier'


"""
------------------------------------------------------
----------- 2. PREPARAÇÃO DA BASE DE DADOS -----------
        2.1 Leitura das bases de treino e teste
------------------------------------------------------ 
"""

# Leitura das bases
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
    

"""
------------------------------------------------------
----------- 2. PREPARAÇÃO DA BASE DE DADOS -----------
        2.3 Construindo pipelines de preparação
------------------------------------------------------ 
"""

# Construindo pipeline inicial utilizado no treinamento
initial_train_pipeline = Pipeline([
    ('col_filter', FiltraColunas(features=INITIAL_FEATURES)),
    ('dtype_modifier', ModificaTipoPrimitivo(mod_dict=DTYPE_MODIFICATION_DICT)),
    ('dup_dropper', EliminaDuplicatas())
])

# Construindo pipeline inicial utilizado na predição (sem eliminação de duplicatas)
initial_pred_pipeline = Pipeline([
    ('col_filter', FiltraColunas(features=INITIAL_PRED_FEATURES)),
    ('dtype_modifier', ModificaTipoPrimitivo(mod_dict=DTYPE_MODIFICATION_DICT))
])

# Construindo pipeline numérico
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy=NUM_STRATEGY_IMPUTER)),
    ('log_transformer', DynamicLogTransformation(application=LOG_APPLICATION, num_features=NUM_FEATURES, 
                                                 cols_to_log=COLS_TO_LOG)),
    ('scaler', DynamicScaler(scaler_type=SCALER_TYPE))
])

# Construindo pipeline categórico
cat_pipeline = Pipeline([
    ('encoder', DummiesEncoding(dummy_na=ENCODER_DUMMY_NA, cat_features_final=CAT_FEATURES_FINAL))
])

# Criando pipeline completo
prep_pipeline = ColumnTransformer([
    ('num', num_pipeline, NUM_FEATURES),
    ('cat', cat_pipeline, CAT_FEATURES)
])


"""
------------------------------------------------------
----------- 2. PREPARAÇÃO DA BASE DE DADOS -----------
      2.4 Aplicação dos pipelines de preparação
------------------------------------------------------ 
"""

# Executando pipeline inicial nos dados de treino e teste
train_prep = initial_train_pipeline.fit_transform(train)
teste_prep = initial_pred_pipeline.fit_transform(test)

# Separando dados de treino e de validação
X_train, X_val, y_train, y_val = train_test_split(train_prep.drop(TARGET, axis=1), train_prep[TARGET].values,
                                                  test_size=.20, random_state=42)

# Executando pipelines de preparação
X_train_prep = prep_pipeline.fit_transform(X_train)
X_val_prep = prep_pipeline.fit_transform(X_val)


"""
------------------------------------------------------
------- 3. TREINAMENTO E AVALIAÇÃO DE MODELOS --------
        3.1 Estruturando objetos de modelagem
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
#print(f'Classificadores a serem treinados: \n\n{model_names}')


"""
------------------------------------------------------
------- 3. TREINAMENTO E AVALIAÇÃO DE MODELOS --------
  3.2 Encapsulando etapa de treinamento e avaliação
------------------------------------------------------ 
"""

"""# Criando objeto e realizando treinamento
trainer = ClassificadorBinario()
trainer.fit(set_classifiers, X_train_prep, y_train, random_search=False)

# Resultados do treinamento (analítico)
metrics = trainer.evaluate_performance(X_train_prep, y_train, X_val_prep, y_val, 
                                       save=True, output_path=OUTPUT_PATH)

# Resultados do treinamento (visual)
trainer.plot_metrics(save=True, output_path=OUTPUT_PATH)

# Análise das features mais importantes pra cada modelo
trainer.plot_feature_importance(features=MODEL_FEATURES, save=True, output_path=OUTPUT_PATH)

# Plotando matriz de confusão
trainer.plot_confusion_matrix(save=True, output_path=OUTPUT_PATH)

# Visualizando curva ROC
trainer.plot_roc_curve(save=True, output_path=OUTPUT_PATH)

# Visualizando distribuição de scores
trainer.plot_score_distribution(save=True, output_path=OUTPUT_PATH)

# Analisando score em faixas
trainer.plot_score_bins(save=True, output_path=OUTPUT_PATH)

# Analisando curva de aprendizado
trainer.plot_learning_curve(save=True, output_path=OUTPUT_PATH)

# Realizando análise shap
trainer.plot_shap_analysis(model_name=MODEL_SHAP, features=MODEL_FEATURES, 
                           save=True, output_path=OUTPUT_PATH)"""


# Instanciando novo objeto
trainer = ClassificadorBinario()

# Realizando treinamento e avaliação dos modelos
trainer.training_flow(set_classifiers, X_train_prep, y_train, X_val_prep, y_val, 
                      features=MODEL_FEATURES, metrics_output_path=METRICS_OUTPUT_PATH,
                      models_output_path=MODELS_OUTPUT_PATH)

# Gerando e salvando gráficos de análises visuais
trainer.visual_analysis(features=MODEL_FEATURES, model_shap=MODEL_SHAP, 
                        output_path=IMGS_OUTPUT_PATH)