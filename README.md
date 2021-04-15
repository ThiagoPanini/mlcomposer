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

## Features

Em sua versão atual, o pacote `mlcomposer` conta com o módulo `ml` responsável por alocar submódulos específicos dentro de dois temas fundamentais em um projeto de aprendizado de máquina: preparação e modelagem. Dessa forma, pode-se utilizar a tabela abaixo como uma referência mais detalhada sobre o conteúdo de cada submódulo citado:

| Submódulo         | Descrição                                                    | Funções/Métodos   | Classes         | Componentes Totais  |Linhas de Código |
| :---------------: | :---------------:                                            | :---------------: | :-------------: | :-----------------: | :-------------: |
| `ml.transformers` | Classes transformadoras para Data Prep                       |         0         |        16       |        16           |     ~670        |
| `ml.trainer`      | Classes de modelagem para treinamento e avaliação de modelos |        58         |        3        |        61           |    ~3600        |

Observando a tabela explicativa acima, percebe-se a quantidade de elementos disponibilizados previamente pelo pacote `mlcomposer` de modo a facilitar o trabalho do Cientista de Dados na preparação e modelagem dos dados. Na prática, os componentes entregues permitem que o cientista possa direcionar esforços para a real avaliação do melhor cenário dentro do contexto do projeto, evitando possíveis gargalos no desenvolvimento devido a necessidade de uma densa codificação para o retorno de insumos suficientes para que se possa partir para a etapa de produtizaçoã dos modelos.

### Módulo ml.transformers

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

### Módulo ml.trainer

Considerando as evoluções alcançadas na etapa de preparação dos dados com a utilização das funcionalidades do módulo `ml.transformers`, o módulo `ml.trainer` atua diretamente na utilização de uma base de dados devidamente preparada para treinar e avaliar diferentes modelos de Machine Learning a partir das mais variadas abordagens. Neste módulo, são propostas classes para cada tipo diferente de aprendizado, sendo elas:

* **ClassificadorBinario()**: classe contendo diversos métodos responsáveis pelo treinamento e avaliação de modelos de classificação binária;
* **ClassificadorMulticlass()**: classe contendo diversos métodos responsáveis pelo treinamento e avaliação de modelos de classificação multiclasse;
* **RegressorLinear()**: classe contendo diversos métodos responsáveis pelo treinamento e avaliação de modelos de regressão linear;

Dentro de cada uma dessas três classes, uma série de métodos são disponibilizados de modo a encapsular todo o treinamento, tunagem de hiperparâmetros, avaliação de métrics utilizando validação cruzada, avaliação visual de métricas e até persistências de modelos treinados em formato pkl. Os resultados proporcionados por esses métodos serão detalhados mais a frente na sessão de utilização prática do pacote.

___

## Instalação

A última versão do pacote `mlcomposer` encontra-se publicada no repositório <a href="https://pypi.org/project/mlcomposer/">PyPI</a>.

> :pushpin: **Nota:** como boa prática de utilização em qualquer projeto Python, a construção de um <a href="">ambiente virtual</a> se faz necessária para um maior controle das funcionalidades e das dependências atreladas ao código. Para tal, o bloco abaixo considera os códigos necessários a serem executados no cmd para a criação de ambientes virtuais Python nos sistemas Linux e Windows.
> 

```bash
# Criação e ativação de venv no linux
$ python -m venv <path_venv>/<nome_venv>
$ source <path_venv>/<nome_venv>/bin/activate

# Criação e ativação de venv no windows
$ python -m venv <path_venv>/<nome_venv>
$ <path_venv>/<nome_venv>/Scripts/activate
```

Com o ambiente virtual ativo, basta realizar a instalação do pacote através do comando `pip`:

```bash
$ pip install mlcomposer
```
