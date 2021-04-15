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

## transformers

Visando detalhar as funcionalidades presentes em cada um dos submódulos do pacote, a lista abaixo contém todas as classes transformadoras presentes no submódulo `ml.transformers`. Em resumo, os blocos disponibilizados sob esse título representam, além de construções de classes voltadas especificamente para a transformação de dados, também uma forma fácil de encadear um pipeline completo de preparação. Para tal, as classes aqui disponibilizadas foram desenvolvidas herdando os elementos das classes `BaseTransformer e TransformerMixin` do scikit-learn, proporcionando assim uma integração dos métodos `fit()` e `transform()` de modo a gerar automaticamente o método `fit_transform()`. Com isso, os transformadores podem ser sequencialmente alocados na classe `Pipeline`, também oriunda do scikit-learn, 
