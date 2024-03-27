import pandas as pd
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

dados = pd.read_csv('https://raw.githubusercontent.com/ViniciusCBezerra/ValidacaoMachineLearning/main/Exercicios/desafio002/arquivos/diabetes.csv')

x = dados.drop('diabete',axis=1)
y = dados['diabete']

x_treino, x_teste , y_treino, y_teste = train_test_split(
    x, y,
    test_size=0.3,
    random_state=5
)

modelo1 = DecisionTreeClassifier(random_state=5)
cv_results = cross_validate(modelo1,x_treino,y_treino)
modelo1.fit(x_treino,y_treino)
y_pred1 = modelo1.predict(x_teste)


modelo2 = LogisticRegression(random_state=5)
cv_results = cross_validate(modelo2,x_treino,y_treino)

pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('modelo2',modelo2)
])
validacao = cross_validate(pipeline,x_treino,y_treino)

modelo3 = GaussianNB()

nome_modelo = ['Árvores','Logística','Naive Bayes']

pipelines = []
for modelo, nome in zip([modelo1, modelo2, modelo3], nome_modelo):
    pipeline = Pipeline([
        ('scaler',StandardScaler()),
        ('model',modelo)
    ])
    pipelines.append(pipeline)

    validacao = cross_validate(pipeline,x_treino,y_treino)
    print(validacao['test_score'].mean())

votacao = VotingClassifier(estimators=[
    (nome_modelo[0],pipelines[0]),
    (nome_modelo[1],pipelines[1]),
    (nome_modelo[2],pipelines[2]),
],voting='hard')

validacao = cross_validate(votacao,x_treino,y_treino,cv=5)
print(validacao['test_score'].mean())