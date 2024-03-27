import pandas as pd

def prepara_dados(df):
    df.drop(['Unnamed: 0','id'],axis=1,inplace=True)
    df.dropna(inplace=True)
    df = pd.get_dummies(df,columns=colunas_categoricas)

    x = df.drop('')


url1 = 'https://raw.githubusercontent.com/alura-cursos/combina-classificadores/main/dados/train.csv'
url2 = 'https://raw.githubusercontent.com/alura-cursos/combina-classificadores/main/dados/test.csv'


colunas_categoricas = ['Gender','Customer Type','Type of Travel','Class']
treino = pd.read_csv(url1)
teste = pd.read_csv(url2)
print(treino.info())
