import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV


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

votacao = VotingClassifier(estimators=(
    (nome_modelos[0],pipelines[0]),
    (nome_modelos[1],pipelines[1]),
    (nome_modelos[2],pipelines[2])
),voting='soft')

validacao = cross_validate(votacao,x_treino,y_treino,cv=5)

parametros = {
    'voting': ['hard','soft'],
    'weights': [(1,1,1),(2,1,1),(1,2,1),(1,1,2)]
}

grid_search = GridSearchCV(votacao,parametros,n_jobs=-1)
grid_search.fit(x_treino,y_treino)

print(grid_search.best_params_)
print(grid_search.best_score_)
