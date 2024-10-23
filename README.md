# Previsão de Preço de Casas usando Regressão Linear

Este projeto utiliza **Regressão Linear** para prever o preço de casas com base na área (em metros quadrados). A biblioteca **scikit-learn** foi utilizada para construir o modelo, e a previsão é feita com base em dados históricos de área e preço.

## Descrição

A **Regressão Linear** é uma técnica de aprendizado supervisionado que busca modelar a relação entre uma variável dependente contínua (neste caso, o preço da casa) e uma variável independente (a área da casa). O objetivo é encontrar uma linha reta que melhor se ajusta aos dados para realizar previsões futuras.

### Exemplo Prático

Com base nos dados de casas de diferentes áreas e seus respectivos preços, podemos prever o preço de uma nova casa fornecendo sua área como entrada para o modelo.

Por exemplo:
- Para uma casa com **80 m²**, o modelo prevê um preço de **R$ 240.000,00**.

### Conjunto de Dados

Os dados usados no projeto são os seguintes:

| Área (m²) | Preço (R$) |
|-----------|------------|
| 50        | 150000     |
| 70        | 210000     |
| 100       | 300000     |
| 120       | 360000     |
| 150       | 450000     |

### Implementação

Aqui está o código usado para criar o modelo de regressão linear:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Criando o dataset
dados = {
    'Área (m²)': [50, 70, 100, 120, 150],
    'Preço (R$)': [150000, 210000, 300000, 360000, 450000]
}
dataset = pd.DataFrame(dados)

# Preparar os dados
area = dataset['Área (m²)'].values.reshape(-1, 1)
preco = dataset['Preço (R$)'].values

# Criar o modelo de Regressão Linear
modelo = LinearRegression()
modelo.fit(area, preco)

# Fazer previsões
nova_area = np.array([[80]])
preco_previsto = modelo.predict(nova_area)

print(f"O preço previsto para uma casa de 80 m² é: R$ {preco_previsto[0]:.2f}")
Saída Esperada
Após rodar o código acima, o resultado será:


O preço previsto para uma casa de 80 m² é: R$ 240000.00
Requisitos
Para executar este projeto, você precisará das seguintes bibliotecas Python:

pandas: Para manipulação e análise de dados.
NumPy: Para operações matemáticas.
scikit-learn: Para criar e treinar o modelo de regressão linear.
Instale as dependências usando o seguinte comando:


pip install pandas numpy scikit-learn
Diferenças entre Regressão Linear e Logística
A Regressão Linear é utilizada para prever valores contínuos, enquanto a Regressão Logística é usada para prever probabilidades de uma variável categórica.

Aspecto	Regressão Linear	Regressão Logística
Saída	Valor contínuo	Probabilidade de uma classe
Modelo	Linear	Não-linear (logística)
Exemplo	Preço de uma casa	Probabilidade de compra








