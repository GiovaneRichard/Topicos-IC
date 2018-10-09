# **************************************************
# Trabalho 01 Introdução IC
# Equipe: Anne Almeida, Giovane Richard e Robert
# Professora: Luciana Balieiro
# Algoritmo: REGRESSÃO LINEAR
# **************************************************

# separação dos dados de treinamento do  modelo linear e dados para validação
# inclui o módulo de regressão linear
from sklearn import linear_model
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
from pprint import pprint


boston = load_boston()
#boston = load_Heart_Disease()
#itens da base
boston.keys()

#print(boston.DESCR)

tabela = pandas.DataFrame(boston.data)
tabela.columns = boston.feature_names
tabela.head()
# mostrar uma quantidade maior de linhas, no caso, 10
tabela.head(10)

# adicionando a coluna de Preço
tabela['Preco'] = boston.target
tabela.head(10)

# Mostrando graficamente a relação entre Crimes e o Preço
plt.scatter(tabela.CRIM, tabela.Preco)
plt.xlabel('Taxa de Crimes')
plt.ylabel('Preço')
plt.show()

# Mostrando graficamente a relação entre AGE e o Preço
plt.scatter(tabela.AGE, tabela.Preco)
plt.xlabel('Proporção de casa construídas antes de 1940')
plt.ylabel('Preço')
plt.show()

# Mostrando graficamente a relação entre AGE e o Preço
plt.scatter(tabela.LSTAT, tabela.Preco)
plt.xlabel('lower status')
plt.ylabel('Preço')
plt.show()

# Seleção de 2 colunas
X = tabela[["RM", "LSTAT"]]
#print(X)

# separação em dois conjuntos, um para o treinamento e outro para validação (20 últimos)
X_t = X[:-20] 
X_v = X[-20:]
# print(X_t["RM"])
y_t = tabela["Preco"][:-20]
y_v = tabela["Preco"][-20:]

regressao = linear_model.LinearRegression()

# treinamento do modelo
regressao.fit(X_t, y_t)

# fazendo a predição
y_pred = regressao.predict(X_v)

# coeficiente a
print('\n Coeficiente:  ', regressao.coef_)
# intercepto b
print('\n Coeficiente:  ', regressao.intercept_)

# y = 5.10 * RM + -0.65 * LSTAT + -1.24

# Aprendizado manual e os valores com base nos coeficientes encontrados na regressao
y_teste = 5.10 * X_v["RM"] - 0.65 * X_v ["LSTAT"] - 1.24

# exibindo o valor predito manualmentes começando em 486
# exibindo o valor real y_t
# exibindo o valor predito pela regressão linear

print('\n', y_teste[486], y_t[0], y_pred[0])

# plota todos os valores de validação
plt.scatter(X_v["LSTAT"], y_v, color = 'green')
plt.scatter(X_v["LSTAT"], y_pred, color = 'blue')
plt.legend(["Real", "Predito"])