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

parametros = {
    'voting': ['hard','soft'],
    'weights': [(1,1,1),(2,1,1),(1,2,1),(1,1,2)]
}

grid_search = GridSearchCV(votacao,parametros,n_jobs=-1)

bagging_classifier = BaggingClassifier(n_estimators=10,random_state=42)
bagging_classifier.fit(x_treino,y_treino)
y_pred = bagging_classifier.predict(x_teste)

print(accuracy_score(y_teste,y_pred))
