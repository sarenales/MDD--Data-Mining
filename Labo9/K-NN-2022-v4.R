# la siguiente linea debes "adaptarla" a donde tengas tu directorio de trabajo, a donde estÃ© el fichero csv de trabajo
setwd("D:/silvi/Documents/uni3D/cuatri1/MDD/Labo9")

# encontrarÃ¡s el fichero iris.csv en egela, "directorio de datatasets de la asignatura"
# descÃ¡rgatelo en local y cÃ¡rgalo en R, en tu directorio de trabajo
# tiene varios nombres separados por coma

iris<-read.csv("iris.csv", header=TRUE, sep=",")
summary(iris) # se muestra un resumen de los datos del fichero de iris.
names(iris)   # se muestran los nombres de las variables predictoras.
head(iris)  # se muestran los valores de cada  columna para las primeras filas.

# se reescribe el atributo, variety,la clase, el cual es de tipo caracter y queremos que sea de tipo nominal, factor
# cada cadena de caracteres es una varaible nominal, etiqueta
# diriamos que es un tipo de casting
# Como hemos podido ver la ultima columna del dataset es la clase del individuo, el nombre de la variedad de flor, y es por supuesto una clase nominal. 
# Para poder trabajar mejor con el dataset convertiremos la variable en numérica.

iris$variety <- as.factor(iris$variety) 
table(iris$variety) # se muestran la longitud se cada clase

# caret --> paquete de R con funciones para "classification and regression"
# https://topepo.github.io/caret/
# primero, instalaciÃ³n del paquete:
install.packages("caret", dependencies=T)
# luego, cargar las funciones de la libreria:
library(caret)
set.seed("1234567890") # Y¿para que hacemos esto?
# se establece una semilla inicial
# en funcion de ella se crearan las particiones de train y test para el modelo.

help("createDataPartition")
# o visita https://www.rdocumentation.org/packages/caret/versions/6.0-86/topics/createDataPartition
# o busca en Google "createdatapartition R"


# una vez establecida la semilla creamos las particiones con el metodo createDataPartition

# explica cada parametro de cada funcion: en este caso, los parametros "y", "p", "list". Idem para el resto de funciones del script

# es importatne entrener que parametros recoge esta funcion y como cambia la salida en funcion de ellos
# se escogen muestras bootstrap desde el train
# Parametros: 
# y     es el vector de resultados, para eso escogemos la columna de la clase
# p     es el porcentaje de los datos que se iran a train
# list  es una variable booleana que indica si el resultado se guardara en una lista
trainSetIndexes <- createDataPartition(y=iris$variety,p=.66,list=FALSE)

# Y¿que recoge el objeto "trainSetIndexes"?

# usando la variable en la que guardamos la particion y su negada como indice para el dataset obtenemos las dos particiones

# se guardan la variable trainSet las muestras que pertecen a las del trainSetIndexes
trainSet <- iris[trainSetIndexes,]


# todas menos las del trainSetIndexes
testSet <- iris[-trainSetIndexes,]

# nuumero de elementos del subconjunto del trainSet
nrow(trainSet)

# numero de elementos del subconjunto del testSet
nrow(testSet)


# guardamos en la variable ctrl, los parametrso que tendra nuestro clasificador, un KNN.
# controla que la particion del training. basicamente se realiza un 10-cross-validation
# Parametros:
# method  es metodo utilizado, cv es cross-validation
# number  numero de capas/particiones que se realizan en cada iteracion
ctrl <- trainControl("cv", number=10)
# la siguiente funciÃ³n, "train", es clave. ExplicaciÃ³n de cada parÃ¡metro: a tu cuenta. Ya que he visto varios enlaces con la ayuda, mejor, consulta el siguiente enlace: 
# https://www.rdocumentation.org/packages/caret/versions/6.0-86/topics/train

# Â¿CÃ³mo codifica cuÃ¡les son las predictoras, y cuÃ¡l la clase a predecir?
print(ctrl)


# Â¿Ves algÃºn otro parÃ¡metro interesante en la ayuda de la funciÃ³n "train? Â¿CuÃ¡les?

# la funcion train() es la que geneera el clasificador juntando todas las piezas que hemos generado hasta ahora.
# La clase a predecir se marca en el primer parametro y separado por el caracter ~. Estos son los parametros mas importantes que toma:

# method: Especifica que metodo de clasificacion se va a usar para crear el clasificador
# tuneLegth: se usa para especificar para cuantos parámetros diferentes se probará el modelo. En nuestro caso se probarán 5 valores diferentes para K y el modelo se quedará con el que tenga mejor accuracy.
# trControl: trControl especifica que parametros queremos que tenga el clasificador, se usan los parametros ya especificados previamente.
# preProc: expresa el pre procesado previo que se le realiza a nuestros datos antes de entrenar el clasificador.

KNNModel1 <- train(variety ~ ., data=trainSet, method="knn", tuneLength=5, trControl=ctrl, preProc=c("scale"))
KNNModel1
plot(KNNModel1)
# Â¿QuÃ© implica "escalar" una variable numÃ©rica, tal y como hemos pedido en las opciones de preproceso?

KNNPredict1 <- predict(KNNModel1, newdata=testSet)
confusionMatrix(KNNPredict1, testSet$variety)
# Â¿se muestra la matriz de confusiÃ³n en el "orden" que lo hace WEKA? Ojo... 
# Fijate que, como WEKA, muestra scores por cada clase

# Usando el subconjunto de test y el modelo generado podemos ver cual es su fiabilidad para conjunto de individuos nuevo.

# Â¿quÃ© efecto tiene el parÃ¡metro "tuneGrid"?
KNNModel2 <- train(variety ~ ., data=trainSet, method="knn", tuneGrid = expand.grid(k = c(1, 3, 5, 15, 19)), trControl=ctrl, preProc=c("scale"))
KNNModel2
plot(KNNModel2)
 
# Si nos fijamos la matriz de confusión que nos da R está traspuesta en comparación con la que nos devolvía Weka, es decir las columnas son las clases reales y las filas las clases predecidas.

# Para un segundo ejemplo, en lugar de usar tuneLength para especificar el número de valores diferentes que tomará el parámetro del modelo, usaremos tuneGrid; con el que podremos elegir con que valores específicos se probará el clasificador.


# En el proceso de aprendizaje: 
# Â¿quÃ© diferencia principal hay entre el de "KNNModel1" y "KNNModel2"?
# Â¿QuÃ© parÃ¡metro se ha tuneado en el proceso de aprendizaje? Â¿Con quÃ© valor definitivo se ha quedado, cada modelo?

# Hemos probado el clasificador para los valores [1,3,5,15,19] y dado que el accuracy más alto se ha conseguido para K=3, como se puede ver en la gráfica, el modelo se ha decantado por ese valor.


KNNPredict2 <- predict(KNNModel2, newdata=testSet)
confusionMatrix(KNNPredict2, testSet$variety)

# Ahora, aÃ±ade las lineas que aprendan&validen: un modelo naiveBayes, y otro de Ã¡rboles de clasificaciÃ³n.
# En el siguiente enlace tienes los modelos, agrupados por familias, que se pueden aprender con el paquete "caret": 
# Fijate en las familias "Bayesian model" y "Tres-based model": y escoge en ellas dos clasificadores que conozcamos
# https://topepo.github.io/caret/train-models-by-tag.html

# Clasificadores Naïve Bayes y árbol de clasificación

# Además del KNN, mediante el paquete caret y la función train se pueden aprender otros clasificadores. Vamos a hacer la prueba para el modelo Bayesiano Naïve Bayes y el árbol de clasificación.

library(naivebayes)

