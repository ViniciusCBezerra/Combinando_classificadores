import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier


df = pd.read_csv('https://raw.githubusercontent.com/ViniciusCBezerra/ValidacaoMachineLearning/main/dados_inadimplencia.csv')

x = df.drop('inadimplente',axis=1)
y = df['inadimplente']

x_treino,x_teste,y_treino,y_teste = train_test_split(
    x,y,
    random_state=42,
    stratify=y
)

modelo1 = DecisionTreeClassifier(random_state=42)
validacao = cross_validate(modelo1,x_treino,y_treino,cv=5)
modelo1.fit(x_treino,y_treino)

modelo2 = LogisticRegression(random_state=42)
pipeline = Pipeline([
    ('Scaler',StandardScaler()),
    ('model',modelo2)
])
validacao = cross_validate(pipeline,x_treino,y_treino,cv=5)
modelo2.fit(x_treino,y_treino)

modelo3 = GaussianNB()

nome_modelos = ['Árvore','Logística','Naives Bayes']
pipelines = []

for modelo, nome in zip([modelo1, modelo2, modelo3], nome_modelos):
    pipeline = Pipeline([
        ('Scaler',StandardScaler()),
        ('model',modelo)
    ])
    pipelines.append(pipeline)

    validacao = cross_validate(pipeline,x_treino,y_treino,cv=5)
    #print(validacao['test_score'].mean())

votacao = VotingClassifier(estimators=[
    (nome_modelos[0],pipelines[0]),
    (nome_modelos[1],pipelines[1]),
    (nome_modelos[2],pipelines[2])
],voting='hard')

validacao = cross_validate(votacao,x_treino,y_treino,cv=5)

parametros = {
    'voting': ['hard','soft'],
    'weights': [(1,1,1),(2,1,1),(1,2,1),(1,1,2)]
}

grid_search = GridSearchCV(votacao,parametros,n_jobs=-1)

bagging_classifier = BaggingClassifier(
    n_estimators=10,
    random_state=42
)
bagging_classifier.fit(x_treino,y_treino)

modelo_base = pipelines[0]
param_bagging = {
    'n_estimators': [10,20,30],
    'max_samples': [0.5,0.7,0.9],
    'max_features': [0.5,0.7,0.9]
}

bagging_grid = GridSearchCV(
    BaggingClassifier(),
    param_bagging,
    cv=5,
    n_jobs=-1
)
bagging_grid.fit(x_treino,y_treino)
melhores_param = bagging_grid.best_params_

bagging_classifier = BaggingClassifier(estimator=modelo_base,**melhores_param)
bagging_classifier.fit(x_treino,y_treino)
y_pred = bagging_classifier.predict(x_teste)
print(accuracy_score(y_teste,y_pred))

param_extra = {
    'n_estimators': [10,20,30],
    'max_features': [0.5,0.7,0.9]
}

extratrees_grid = GridSearchCV(
    ExtraTreesClassifier(),
    param_extra,
    cv=5,
    n_jobs=-1
)
extratrees_grid.fit(x_treino,y_treino)
melhores_param = extratrees_grid.best_params_

extratrees_classifier = ExtraTreesClassifier(**melhores_param)
extratrees_classifier.fit(x_treino,y_treino)
y_pred = extratrees_classifier.predict(x_teste)
print(accuracy_score(y_teste,y_pred))

adaboost_classifier = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1
)
adaboost_classifier.fit(x_treino,y_treino)

y_pred = adaboost_classifier.predict(x_teste)
print(accuracy_score(y_teste,y_pred))