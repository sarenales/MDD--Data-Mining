
import csv  # importar fichero csv
import math
import statistics as st

import matplotlib.pyplot as plt  # graficos, como el plot de R
from matplotlib.ticker import MaxNLocator

import numpy as np  # posibilita hacer arrays
import pandas as pd  # dataframes, los datos de una matrix iguales.

from pylab import rcParams
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
    mtcars_normalizado = mtcars.loc[:, mtcars.columns != 'mpg'].apply(normalize, axis=1) # se le pasa el puntero a la funcion
   
    # Anadimos la clase a nuestro dataset normalizado
    mtcars_normalizado['mpg'] = mtcars['mpg']
    print("Este es el dataset normalizado")
    print(mtcars_normalizado)
    print("\n\n")
    
    # Hacemos un split en train y test con un porcentaje del 0.75 Train
    parte_test, parte_train = splitTrainTest(mtcars_normalizado, 0.75)
    
    # Separamos las labels del Test. Es como si no nos las dieran!!
    testT = parte_test.loc[:, parte_test.columns != 'mpg'] # saco la columna mpg y la paso a lista
    testR = parte_test.loc[:, parte_test.columns == 'mpg']

    acc = [] 

    # Predecimos el conjunto de test

    true_labels = 0
    predicted_labes = 0
    
    print("CASOS DE TEST: \n")
    
    for i in range(len(testT)):
        estimation = knn(testT.iloc[i,:], parte_train, 3)
        predicted_labes +=1
        print("Numero de Prueba: ", i , "Clase ESTIMADA -->", estimation, "Clase REAL --> ", testR.iloc[i,0], "\n")
        if estimation == testR.iloc[i,0]:
            true_labels +=1
            
    # hacemos un mini resumen de lo conseguido

    print("Accuracy conseguido:", accuracy(true_labels, predicted_labes))
            


    # Algun grafico? Libreria matplotlib.pyplot
   # plt.plot(range(1,21), acc)
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
    trainSetT = data.loc[:, data.columns != 'mpg']
    trainSetR = data.loc[:, data.columns == 'mpg']
    
    distances = []
    shortests_distances = []

    # calculamos todas las distancias
    for i in range(len(trainSetT)):
        d = euclideanDistance2points(newx, trainSetT.iloc[i,:])
        distances.append(d)
        shortests_distances.append(d)
        
    shortests_distances.sort()
    
    # obtenemos los indices de los vecinos mas cercanos
    indices = []
    for v in range(K):
        i = distances.index(shortests_distances[v])
        indices.append(i)
        # marcamos el elemento ya utilizado, para no repetirlo
        distances[i] = -1
        
    # clases estimada
    posibles_clases = []
    for j in indices:
        posibles_clases.append(trainSetR.iloc[i,0])
        
    #obtenemos las diferentes clases posibles
    clases = [posibles_clases[0]]
    for c in posibles_clases:
        if c not in clases:
            clases.append(c)
            
    # obtenemos la clase mayoritaria
    clasem = clases[0]
    mayoritaria = posibles_clases.count(clasem)
    for c in clases:
        n = posibles_clases.count(c)
        # si esta es mayor que la mayoritaria calculada
        if n > mayoritaria:
            mayoritaria = n
            clasem = c

    return(clasem)

def euclideanDistance2points(x,y):
    """
    Takes 2 matrix - Not pandas dataframe!
    """
    # directamente entre matrices. todo vector se puede operar uno con otro.
    # ((math.sqrt((x-y)**2)))
    s = 0
    for(i,j) in zip(x,y):
        s += (i-j)**2 
    return (np.sqrt(s))

# FUNCION accuracy
def accuracy(true, pred):
    return (true/pred)

if __name__ == '__main__':
    np.random.seed(25) #pone una semilla (como en weka)
    main()
