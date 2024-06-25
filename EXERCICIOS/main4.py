import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier


def prepara(df):
    df.drop(['Unnamed: 0','id'],axis=1,inplace=True)
    df.dropna(inplace=True)
    df = pd.get_dummies(df,columns=colunas_categoricas)
    x = df.drop('satisfaction',axis=1)
    y = df['satisfaction']

    return x,y

url1 = 'https://raw.githubusercontent.com/alura-cursos/combina-classificadores/main/dados/train.csv'
url2 = 'https://raw.githubusercontent.com/alura-cursos/combina-classificadores/main/dados/test.csv'

treino = pd.read_csv(url1)
teste = pd.read_csv(url2)
colunas_categoricas=['Gender','Customer Type','Type of Travel','Class']

x_treino,y_treino = prepara(treino)
x_teste,y_teste = prepara(teste)

modelo1 = DecisionTreeClassifier(random_state=42)
modelo1.fit(x_treino,y_treino)
validacao = cross_validate(modelo1,x_treino,y_treino,cv=5)

modelo2 = LogisticRegression(random_state=42)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('modelo', modelo2)
])
validacao = cross_validate(pipeline,x_treino,y_treino,cv=5)
print(validacao['test_score'].mean())

modelo3 = GaussianNB()

nome_modelos = ['Árvore', 'Logística', 'Naives Bayes']
pipelines = []

for modelo, nome  in zip([modelo1, modelo2, modelo3],nome_modelos):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('modelo', modelo)
    ])

    pipelines.append(pipeline)

parametros = {
    'voting': ['hard','soft'],
    'weights': [(1,1,1),(2,1,1),(1,2,1),(1,1,2)]
}

bagging_classifier = BaggingClassifier(
    n_estimators=10,
    random_state=42
)
bagging_classifier.fit(x_treino,y_treino)

y_pred = bagging_classifier.predict(x_teste)
print(accuracy_score(y_teste,y_pred))

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

parametros_extratrees = {
    'n_estimators': [10,20,30],
    'max_features': [0.5,0.7,0.9]
}

extratrees_grid = GridSearchCV(
    ExtraTreesClassifier(),
    parametros_extratrees,
    cv=5,
    n_jobs=-1
)
extratrees_grid.fit(x_treino,y_treino)
melhores_param_trees = extratrees_grid.best_params_

extratrees_classifier = ExtraTreesClassifier(**melhores_param_trees)
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

param_cat = {
    'iterations': [100,200,300],
    'depth': [4,6,8],
    'learning_rate': [0.1,0.01,0.001]
}

grid_cat = GridSearchCV(
    estimator=CatBoostClassifier(verbose=0),
    param_grid=param_cat,
    cv=5,
    n_jobs=-1,
    scoring='accuracy'
)
grid_cat.fit(x_treino,y_treino)
y_pred_cat = grid_cat.predict(x_teste)
print(accuracy_score(y_teste,y_pred_cat))
