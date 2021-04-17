<h1 align="center">
  <a href="https://pypi.org/project/mlcomposer/"><img src="https://i.imgur.com/MIcPH8g.png" alt="mlcomposer logo"></a>
</h1>

<div align="center">
  <strong>:robot: Aplicando Machine Learning de forma rápida, fácil e eficiente! :robot:</strong>
</div>
<br/>

<div align="center">  
  
  ![Release](https://img.shields.io/badge/release-ok-brightgreen)
  [![PyPI](https://img.shields.io/pypi/v/mlcomposer?color=blueviolet)](https://pypi.org/project/mlcomposer/)
  ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mlcomposer?color=green)
  ![PyPI - Status](https://img.shields.io/pypi/status/mlcomposer)

</div>
<br/>

O objetivo deste pacote é proporcionar uma forma totalmente encapsulada e eficiente de treinar e avaliar modelos de Machine Learning sem a necessidade de programar uma centena de linhas de código. Através de classes e funções previamente definidas, a biblioteca `mlcomposer` entrega motores para realização das mais diversas transformações de dados dentro de um *pipeline* automático de Data Prep, além de blocos de código capazes de treinar, avaliar, salvar e visualizar diferentes modelos de aprendizado de máquina em um piscar de olhos.

Sua construção é baseada em uma camada superior a bibliotecas Python fundamentais para a modelagem, como o scikit-learn, pandas e numpy. Adicionalmente, funções de visualização de métricas utilizam bibliotecas gráficas, tais como matplotlib e seaborn. Utilizar a biblioteca `mlcomposer` significa introduzir um nível de organização dificilmente visto em projetos práticos de ML, além de proporcionar um alto ganho de eficiência para a entrega do melhor modelo para determinada tarefa.


## Table of content

- [Features](#features)
    - [Módulo transformers](#módulo-transformers)
    - [Módulo trainer](#módulo-trainer)
- [Instalação](#instalação)
- [Utilização](#utilização)

___

## Features

Em sua versão atual, o pacote `mlcomposer` conta com o módulo `ml` responsável por alocar submódulos específicos dentro de dois temas fundamentais em um projeto de aprendizado de máquina: preparação e modelagem. Dessa forma, pode-se utilizar a tabela abaixo como uma referência mais detalhada sobre o conteúdo de cada submódulo citado:

| Submódulo         | Descrição                                                    | Funções/Métodos   | Classes         | Componentes Totais  |Linhas de Código |
| :---------------: | :---------------:                                            | :---------------: | :-------------: | :-----------------: | :-------------: |
| `ml.transformers` | Classes transformadoras para Data Prep                       |         0         |        16       |        16           |     ~670        |
| `ml.trainer`      | Classes de modelagem para treinamento e avaliação de modelos |        58         |        3        |        61           |    ~3600        |

Observando a tabela explicativa acima, percebe-se a quantidade de elementos disponibilizados previamente pelo pacote `mlcomposer` de modo a facilitar o trabalho do Cientista de Dados na preparação e modelagem dos dados. Na prática, os componentes entregues permitem que o cientista possa direcionar esforços para a real avaliação do melhor cenário dentro do contexto do projeto, evitando possíveis gargalos no desenvolvimento devido a necessidade de uma densa codificação para o retorno de insumos suficientes para que se possa partir para a etapa de produtizaçoã dos modelos.

### Módulo transformers

Visando detalhar as funcionalidades presentes em cada um dos submódulos do pacote, a lista abaixo contém todas as classes transformadoras presentes no submódulo `ml.transformers`. Em resumo, os blocos disponibilizados sob esse título representam, além de construções de classes voltadas especificamente para a transformação de dados, também uma forma fácil de encadear um pipeline completo de preparação. Para tal, as classes aqui disponibilizadas foram desenvolvidas herdando os elementos das classes `BaseTransformer e TransformerMixin` do scikit-learn, proporcionando assim uma integração dos métodos `fit()` e `transform()` de modo a gerar automaticamente o método `fit_transform()`. Com isso, os transformadores podem ser sequencialmente alocados na classe `Pipeline`, também oriunda do scikit-learn, permitindo a criação de fluxos completos de preparação sem a necessidade de codificações externas dentro do projeto.

Por fim, as classes transformadoras presentes neste módulo são:
* **FormataColunas()**: aplica padronização nas colunas de um DataFrame a partir da aplicação dos métodos `lower()`, `strip()` e `replace()`;
* **FiltraColunas()**: filtra colunas de um DataFrame;
* **DefineTarget()**: aplica transformação binária na coluna target em uma base de dados (classe positiva=1 e classe negativa=0);
* **EliminaDuplicatas()**: elimina duplicatas de um DataFrame através da aplicação do método `drop_duplicates()`
* **SplitDados()**: realiza a separação de uma base de dados em treino e teste através da função `train_test_split()`
* **AgrupamentoCategoricoInicial()**: agrupa categorias em colunas categóricas com muitas entradas;
* **AgrupamentoCategoricoFinal()**: recebe um dicionário de entradas categóricas para agrupamento customizado nas colunas;
* **ModificaTipoPrimitivo()**: transforma tipos primitivos de uma base através de um dicionário de referência;
* **DummiesEncoding()**: aplica codificação em colunas categóricas a partir da aplicação do método `pd.get_dummies()`;
* **PreencheDadosNulos()**: preenche dados nulos de um DataFrame a partir da aplicação do método `fillna()`;
* **EliminaDadosNulos()**: elimina dados nulos de um DataFrame a partir da aplicação do método `dropna()`;
* **SeletorTopFeatures()**: recebe o resultado do método `feature_importances_()` e filtra os top k índices;
* **LogTransformation()**: aplica transformação logaritma em um DataFrame;
* **DynamicLogTransformation()** aplica transformação logaritma em um DataFrame baseado em um flag booleano de aplicação;
* **DynamicScaler()**: aplica normalização em um DataFrame baseado em um flag booleano de aplicação;
* **ConsumoModelo()**: gera um DataFrame final contendo a predição e o score de um determinado modelo já treinado;

___

### Módulo trainer

Considerando as evoluções alcançadas na etapa de preparação dos dados com a utilização das funcionalidades do módulo `ml.transformers`, o módulo `ml.trainer` atua diretamente na utilização de uma base de dados devidamente preparada para treinar e avaliar diferentes modelos de Machine Learning a partir das mais variadas abordagens. Neste módulo, são propostas classes para cada tipo diferente de aprendizado, sendo elas:

* **ClassificadorBinario()**: classe contendo diversos métodos responsáveis pelo treinamento e avaliação de modelos de classificação binária;
* **ClassificadorMulticlass()**: classe contendo diversos métodos responsáveis pelo treinamento e avaliação de modelos de classificação multiclasse;
* **RegressorLinear()**: classe contendo diversos métodos responsáveis pelo treinamento e avaliação de modelos de regressão linear;

Dentro de cada uma dessas três classes, uma série de métodos são disponibilizados de modo a encapsular todo o treinamento, tunagem de hiperparâmetros, avaliação de métrics utilizando validação cruzada, avaliação visual de métricas e até persistências de modelos treinados em formato pkl. Os resultados proporcionados por esses métodos serão detalhados mais a frente na sessão de utilização prática do pacote.

___

## Instalação

A última versão do pacote `mlcomposer` encontra-se publicada no repositório <a href="https://pypi.org/project/mlcomposer/">PyPI</a>.

> :pushpin: **Nota:** como boa prática de utilização em qualquer projeto Python, a construção de um <a href="https://realpython.com/python-virtual-environments-a-primer/">ambiente virtual</a> se faz necessária para um maior controle das funcionalidades e das dependências atreladas ao código. Para tal, o bloco abaixo considera os códigos necessários a serem executados no cmd para a criação de ambientes virtuais Python nos sistemas Linux e Windows.
> 

```bash
# Criação e ativação de venv no linux
$ python -m venv <path_venv>/<nome_venv>
$ source <path_venv>/<nome_venv>/bin/activate

# Criação e ativação de venv no windows
$ python -m venv <path_venv>/<nome_venv>
$ <path_venv>/<nome_venv>/Scripts/activate
```

Com o ambiente virtual ativo, basta realizar a atualização do `pip` seguida da instalação do pacote:

```bash
$ pip install --upgrade pip
$ pip install mlcomposer
```

Como mencionado anteriormente, a construção do pacote `mlcomposer` é feita utilizando, como pilar, bibliotecas de modelagem fundamental em Python. Dessa forma, ao realizar a instação no ambiente virtual, é esperado que outras bibliteocas dependentes também sejam instaladas. O output esperado no prompt de comando após a instalação deve ser semelhante ao ilustrado abaixo:

```
Collecting mlcomposer
  Downloading mlcomposer-0.0.3-py3-none-any.whl (2.4 kB)
Collecting numpy==1.18.5
  Downloading numpy-1.18.5-cp38-cp38-manylinux1_x86_64.whl (20.6 MB)
     |████████████████████████████████| 20.6 MB 2.6 MB/s 
Collecting shap==0.37.0
  Downloading shap-0.37.0.tar.gz (326 kB)
     |████████████████████████████████| 326 kB 3.0 MB/s 
Collecting scikit-learn==0.23.2
  Downloading scikit_learn-0.23.2-cp38-cp38-manylinux1_x86_64.whl (6.8 MB)
     |████████████████████████████████| 6.8 MB 3.1 MB/s 
Collecting seaborn==0.10.0
  Downloading seaborn-0.10.0-py3-none-any.whl (215 kB)
[...]
Installing collected packages: six, numpy, threadpoolctl, scipy, pytz, python-dateutil, pyparsing, pillow, llvmlite, kiwisolver, joblib, cycler, tqdm, slicer, scikit-learn, pandas, numba, matplotlib, shap, seaborn, python-dotenv, mlcomposer
    Running setup.py install for shap ... done
Successfully installed cycler-0.10.0 joblib-0.14.1 kiwisolver-1.3.1 llvmlite-0.36.0 matplotlib-3.4.1 mlcomposer-0.0.3 numba-0.53.1 numpy-1.18.5 pandas-1.1.5 pillow-8.2.0 pyparsing-2.4.7 python-dateutil-2.8.1 python-dotenv-0.17.0 pytz-2021.1 scikit-learn-0.23.2 scipy-1.6.2 seaborn-0.10.0 shap-0.37.0 six-1.15.0 slicer-0.0.3 threadpoolctl-2.1.0 tqdm-4.60.0
```

## Utilização

A partir desse ponto, a biblioteca `mlcomposer` pode ser importada em scripts Phyton que alocam processos de preparação e treinamento de modelos. Exemplos práticos podem ser encontrados neste repositório no diretório `examples/`. Ilustrando uma aplicação relacionada ao treinamento de um modelo de classificação binária com as funcionalidades do módulo `ml.trainer`, o código abaixo mostra como a aplicação de apenas duas funções podem gerar uma série de insights valiosos para o Cientista de Dados:

```python
# Instanciando objetos
dtree = DecisionTreeClassifier(random_state=42)
forest = RandomForestClassifier(random_state=42)
svm = SVC(kernel='rbf', probability=True)

# Criando dicionário set_classifiers
model_obj = [svm, dtree, forest]
model_names = [type(model).__name__ for model in model_obj]
set_classifiers = {name: {'model': obj, 'params': {}} for (name, obj) in zip(model_names, model_obj)}

# Instanciando novo objeto
trainer = ClassificadorBinario()

# Realizando treinamento e avaliação dos modelos
trainer.training_flow(set_classifiers, X_train_prep, y_train, X_val_prep, y_val, 
                      features=MODEL_FEATURES, metrics_output_path=METRICS_OUTPUT_PATH,
                      models_output_path=MODELS_OUTPUT_PATH)

# Gerando e salvando gráficos de análises visuais
trainer.visual_analysis(features=MODEL_FEATURES, model_shap=MODEL_SHAP, 
                        output_path=IMGS_OUTPUT_PATH)
```

As configurações acima aplicadas na execução dos métodos geram três principais diretórios, cada qual alocando ricos outputs relacionados ao processo de modelagem.

```
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
    │   └── shap_analysis_RandomForestClassifier.png
    ├── metrics
    │   ├── metrics.csv
    │   └── top_features.csv
    └── models
        ├── decisiontreeclassifier_20210414.pkl
        ├── randomforestclassifier_20210414.pkl
        └── svc_20210414.pkl
```

* No diretório `imgs`, são gerados gráficos de análises dos modelos treinados, como uma plotagem de eficiência, matriz de confusão, curva ROC, entre outros;
* No diretório `metrics`, são disponibilizados arquivos analíticos de performance dos modelos e uma lista com as variáveis mais importantes de cada um;
* No diretório `models`, são armazenados os arquivos pkl dos modelos treinados com uma referência de data no formato yyyyMMdd.
