import pandas as pd
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier


def prepara(df):
    df.drop(['Unnamed: 0', 'id'],axis=1,inplace=True)
    df.dropna(inplace=True)
    df = pd.get_dummies(df,columns=colunas_categoricas)
    x = df.drop('satisfaction',axis=1)
    y = df['satisfaction']
    return x,y


url1 = 'https://raw.githubusercontent.com/alura-cursos/combina-classificadores/main/dados/train.csv'
url2 = 'https://raw.githubusercontent.com/alura-cursos/combina-classificadores/main/dados/test.csv'

colunas_categoricas = ['Gender','Customer Type','Type of Travel', 'Class']

treino = pd.read_csv(url1)
teste = pd.read_csv(url2)

x_treino,y_treino = prepara(treino)
x_teste,y_teste = prepara(teste)

modelo1 = DecisionTreeClassifier(random_state=42)
validacao = cross_validate(modelo1, x_treino, y_treino, cv=5)
#print(validacao['test_score'].mean())

modelo2 = LogisticRegression()
pipeline = Pipeline([
    ('Scaler', StandardScaler()),
    ('model',modelo2)
])
validacao = cross_validate(pipeline,x_treino,y_treino,cv=5)
#print(validacao['test_score'].mean())

modelo3 = GaussianNB()

nome_modelos = ['Árvore','Logística','Naives Bayes']
pipelines = []

for modelo, nome in zip([modelo1,modelo2,modelo3],nome_modelos):
    pipeline = Pipeline([
        ('Scaler', StandardScaler()),
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
#print(validacao['test_score'].mean())

bagging_classifier = BaggingClassifier(
    n_estimators=10,
    random_state=42
)
bagging_classifier.fit(x_treino,y_treino)

y_pred = bagging_classifier.predict(x_teste)

modelo_base = pipelines[0]
param_bagging = {
    'n_estimators': [10,20,30],
    'max_samples': [0.5,0.7,0.9],
    'max_features': [0.5,0.7,0.9]
}
bagging_grid = GridSearchCV(
    BaggingClassifier(),
    param_bagging,
    n_jobs=-1,
    cv=5
)

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
