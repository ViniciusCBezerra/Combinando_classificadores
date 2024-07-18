import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier


def prepara(df):
    df.drop(['Unnamed: 0','id'],axis=1, inplace=True)
    df.dropna(inplace=True)
    df = pd.get_dummies(df,columns=colunas_categoricas)
    x = df.drop('satisfaction',axis=1)
    y = df['satisfaction'] 

    return x,y


url1 = 'https://raw.githubusercontent.com/alura-cursos/combina-classificadores/main/dados/train.csv'
url2 = 'https://raw.githubusercontent.com/alura-cursos/combina-classificadores/main/dados/test.csv'


colunas_categoricas = ['Gender', 'Customer Type', 'Class', 'Type of Travel']

treino = pd.read_csv(url1)
teste = pd.read_csv(url2)

x_treino,y_treino = prepara(treino)
x_teste,y_teste = prepara(teste)

modelo1 = DecisionTreeClassifier(random_state=42)
validacao = cross_validate(modelo1,x_treino,y_treino,cv=5)
modelo1.fit(x_treino,y_treino)
y_pred = modelo1.predict(x_teste)

modelo2 = LogisticRegression(random_state=42)
pipeline = Pipeline(
    [
        ('Scaler',StandardScaler()),
        ('model',modelo2)
    ]
)
validacao = cross_validate(pipeline,x_treino,y_treino,cv=5)


modelo3 = GaussianNB()

nome_modelos = ['Árvores','Logística','Naive Bayes']
pipelines = []

for modelo, nome in zip([modelo1,modelo2,modelo3],nome_modelos):
    pipeline = Pipeline(
        [
            ('Scaler',StandardScaler()),
            ('model',modelo)
        ]
    )
    pipelines.append(pipeline)

    validacao  = cross_validate(pipeline,x_treino,y_treino,cv=5)


votacao = VotingClassifier(estimators=[
    (nome_modelos[0],pipelines[0]),
    (nome_modelos[1],pipelines[1]),
    (nome_modelos[2],pipelines[2])
],voting='hard')

validacao = cross_validate(votacao,x_treino,y_treino,cv=5)
modelo_base = pipelines[0]

param_extra = {
    'n_estimators': [10,20,30],
    'max_features': [0.5,0.7,0.9]
}

extratrees_grid = GridSearchCV(
    ExtraTreesClassifier(),
    param_extra,
    n_jobs=-1,
    cv=5
)
extratrees_grid.fit(x_treino,y_treino)
melhores_param_trees = extratrees_grid.best_params_

extratrees_classifier = ExtraTreesClassifier(**melhores_param_trees)
extratrees_classifier.fit(x_treino,y_treino)
y_pred = extratrees_classifier.predict(x_teste)
print(accuracy_score(y_teste,y_pred))

modelo = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1
)
modelo.fit(x_treino,y_treino)

y_pred = modelo.predict(x_teste)
print(accuracy_score(y_teste,y_pred))

param_adaboost = {
    'n_estimators': [50,100,200],
    'learning_rate': [0.1,0.01,0.001]
}
adaboost_classifier = AdaBoostClassifier()

adaboost_grid = GridSearchCV(
    adaboost_classifier,
    param_adaboost,
    scoring='accuracy',
    n_jobs=-1,
    cv=5
)
adaboost_grid.fit(x_treino,y_treino)
print(adaboost_grid.best_params_)

y_pred = adaboost_grid.predict(x_teste)
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
    scoring='accuracy',
    n_jobs=-1    
)
grid_cat.fit(x_treino,y_treino)
y_pred_cat = grid_cat.predict(x_teste)

print(accuracy_score(y_teste,y_pred_cat))


modelos_base = [
    (modelo_base[0],pipelines[0]),
    (modelo_base[1],pipelines[1]),
    (modelo_base[2],pipelines[2])
]

modelo_meta = LogisticRegression(random_state=42)

stacking_classifier = StackingClassifier(estimators=modelos_base,final_estimator=modelo_meta)
stacking_classifier.fit(x_treino,y_treino)
y_pred = stacking_classifier.predict(x_teste)
print(accuracy_score(y_teste,y_pred))