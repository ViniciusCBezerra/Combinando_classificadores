import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

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
grid_search.fit(x_treino,y_treino)

print(grid_search.best_params_)
print(grid_search.best_score_)

