import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler


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
print(validacao['test_score'].mean())