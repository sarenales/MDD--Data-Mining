
import csv  # importar fichero csv
import math
import statistics as st

import matplotlib.pyplot as plt  # graficos, como el plot de R
from matplotlib.ticker import MaxNLocator

import numpy as np  # posibilita hacer arrays
import pandas as pd  # dataframes, los datos de una matrix iguales.


import random
# from sklearn.model_selection import KFold  # kflod validation
 # y para cada columna q sea d un tipo, hay q importarlo. (R, mathlab..)

#pruebas
def main():
    path_dataset = "mtcars.csv" # Escoged bien la ruta!!
    mtcars = pd.read_csv(path_dataset) # Leemos el csv
    # Discretizamos la variable clase para convertirlo en un problema de clasificacion
    ix_consumo_alto = mtcars.mpg >= 21 # crea vector; si cada elemento es mayor de 21 dela columa mtcars.mpg se guarda
    mtcars.mpg[ix_consumo_alto] = 1 # cuando es true
    mtcars.mpg[~ix_consumo_alto] = 0 # cuando no cumple
    print("Este es el dataset sin normalizar")
    print(mtcars)
    print("\n\n")
    # Ahora normalizamos los datos; loc ->selecciona (todas las filas mpg); apply->por cada columna aplica normalize(en c++ maps)
    mtcars_normalizado = mtcars.loc[:, mtcars.columns != 'mpg'].apply(normalize, axis=1) # se le pasa el puntero ala funcion
    # AÃ±adimos la clase a nuestro dataset normalizado
    mtcars_normalizado['mpg'] = mtcars['mpg']
    print("Este es el dataset normalizado")
    print(mtcars_normalizado)
    print("\n\n")
    # Hacemos un split en train y test con un porcentaje del 0.75 Train
    parte_train, parte_test = splitTrainTest(mtcars_normalizado, 0.75)
    # Separamos las labels del Test. Es como si no nos las dieran!!
    # tengo q predecir y luego comprobar. 
    # distancia euclidea solo con las x 
    

 
    true_labels = parte_test.pop('mpg').tolist() # saco la columna mpg y la paso a lista

    acc = [] 

    # Predecimos el conjunto de test

    for K in range (1,21):
        pred_labels = []
        for i in range(len(parte_test)):
            pred_labels.append(knn(parte_test.iloc[i], parte_train, K))


        parte_test['mpg'] = true_labels
        parte_test['pred_mpg'] = pred_labels
        print("Este es el dataset de test con las predicciones")
        print(parte_test)
        parte_test.pop('mpg')
        parte_test.pop('pred_mpg')
        
        # Mostramos por pantalla el Accuracy por ejemplo
        print("Accuracy: ", accuracy(true_labels, pred_labels))
        acc.append(accuracy(true_labels, pred_labels))
    

    # Algun grafico? Libreria matplotlib.pyplot
    plt.plot(range(1,21), acc)
    return(0)

# FUNCIONES de preprocesado
def normalize(x): # normalizar 
    return((x-min(x)) / (max(x) - min(x)))

def standardize(x): # tipificar
    return((x-st.mean(x))/st.variance(x))
    
# FUNCIONES de evaluacion
def splitTrainTest(data, percentajeTrain):
    """
    Takes a pandas dataframe and a percentaje (0-1)
    Returns both train and test sets

    - crear vector T,F
    - luego lo indexo

    numero aleatorio entre 0y1
    np.random.rand(len(mtcars)) -> mascara >0.75
    """
    mk = np.random.rand(len(data))
    mascara = mk > 0.75

    test_path = data.loc[mascara]
    train_path = data.loc[~mascara]
    print("TRAIN")
    print(train_path)
    print("TEST")
    print(test_path)

    return (train_path, test_path)

# FUNCIONES de visualizacion
# def kFoldCV(data, K):
#     """
#     Takes a pandas dataframe and the number of folds of the CV
#     YOU CAN USE THE sklearn KFold function here
#     How to: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
#     """
#     kf = KFold(n_splits = K, shuffle = True, random_state = 2)
#     result = list(kf.split(data))
#     print(result)

# FUNCION modelo prediccion
def knn(newx, data, K):
    """
    Receives two pandas dataframes. Newx consists on a single row df.
    Returns the prediction for newx
    Data is our frame
    """

    data_list = data.values.tolist() # convertir a lista
    newx_list = newx.values.tolist() # convertir a lista

    distances = []


    for i in data_list:
        distances.append(euclideanDistance2points(i, newx_list))

    distances_order = sorted(distances)
    neighbors = []

    for i in range(0,K):
        neighbors.append(distances_order.index(i))

    labels = {}
    for j in neighbors:
        if str(data.loc[data.index[i]]['mpg']) in labels:
            labels[str(data.loc[data.index[i]]['mpg'])] += 1
        else:
            labels[str(data.loc[data.index[i]]['mpg'])] = 1

    newlabel = float(max(labels, key=labels.get)) 

    return(newlabel)

def euclideanDistance2points(x,y):
    """
    Takes 2 matrix - Not pandas dataframe!
    """
    # directamente entre matrices. todo vector se puede operar uno con otro.
    # ((math.sqrt((x-y)**2)))
    s = 0
    for(i,j) in zip(x,y):
        s += (i-j)**2 
    return (math.sqrt(s))

# FUNCION accuracy
def accuracy(true, pred):
    cont = 0 
    for (t,p) in zip(true, pred):
        if t == p:
            cont += 1

    return (cont/len(true))

if __name__ == '__main__':
    np.random.seed(25) #pone una semilla (como en weka)
    main()
