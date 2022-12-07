
# la siguiente línea debes "adaptarla" a donde tengas tu directorio de trabajo, a donde esté el fichero csv de trabajo
setwd("/users/1002494/Desktop")
# encontrarás el fichero iris.csv en egela, "directorio de datatasets de la asignatura"
# descárgatelo en local y cárgalo en R, en tu directorio de trabajo
# tiene varios nombres separados por coma
iris<-read.csv("iris.csv", header=TRUE, sep=",")
summary(iris) # se muestra un resumen de los datos del fichero de iris
names(iris)   # se muestran los nombres de las variables predictoras
head(iris)    
# se reescribe el atributo, variety,la clase, el cual es de tipo caracter y queremos que sea de tipo nominal, factor
# cada cadena de caracteres es una varaible nominal, etiqueta
# diriamos que es un tipo de casting
iris$variety <- as.factor(iris$variety) 
table(iris$variety) # se muestran la longitud se cada clase

# caret --> paquete de R con funciones para "classification and regression"
# https://topepo.github.io/caret/
# primero, instalación del paquete:
install.packages("caret", dependencies=T)
# luego, cargar las funciones de la librería:
library(caret)
set.seed("1234567890") # ¿para qué hacemos ésto?
# se establece una semilla

help("createDataPartition")
# o visita https://www.rdocumentation.org/packages/caret/versions/6.0-86/topics/createDataPartition
# o busca en Google "createdatapartition R"


#explica cada parámetro de cada función: en este caso, los parámetros "y", "p", "list". Idem para el resto de funciones del script
# se escogen muestras bootstrap desde el train
# Parametros: 
# y     es el vector de resultados, para eso escogemos la columna de la clase
# p     es el porcentaje de los datos que se iran a train
# list  es una variable booleana que indica si el resultado se guardara en una lista
trainSetIndexes <- createDataPartition(y=iris$variety,p=.66,list=FALSE)
# ¿qué recoge el objeto "trainSetIndexes"?
# se guardan la variable trainSet las muestras que pertecen a las del trainSetIndexes
trainSet <- iris[trainSetIndexes,]
# todas menos las del trainSetIndexes
testSet <- iris[-trainSetIndexes,]
nrow(trainSet)
nrow(testSet)

# controla que la particion del training. basicamente se realiza un 10-cross-validation
# Parametros:
# method  es metodo utilizado, cv es cross-validation
# number  numero de capas/particiones que se realizan en cada iteracion
ctrl <- trainControl("cv", number=10)
# la siguiente función, "train", es clave. Explicación de cada parámetro: a tu cuenta. Ya que he visto varios enlaces con la ayuda, mejor, consulta el siguiente enlace: 
# https://www.rdocumentation.org/packages/caret/versions/6.0-86/topics/train

# ¿Cómo codifica cuáles son las predictoras, y cuál la clase a predecir?
print(ctrl)

# Al menos, describe los siguientes parámetros: "method", "tuneLength", "trControl", "preProc"
# method: tecnica de validacion que se va a utilizar, es este caso cv, cross-validation
# tuneLenth:
# trControl:
# preProc:


# ¿Ves algún otro parámetro interesante en la ayuda de la función "train? ¿Cuáles?
KNNModel1 <- train(variety ~ ., data=trainSet, method="knn", tuneLength=5, trControl=ctrl, preProc=c("scale"))
KNNModel1
plot(KNNModel1)
# ¿Qué implica "escalar" una variable numérica, tal y como hemos pedido en las opciones de preproceso?

KNNPredict1 <- predict(KNNModel1, newdata=testSet)
confusionMatrix(KNNPredict1, testSet$variety)
# ¿se muestra la matriz de confusión en el "orden" que lo hace WEKA? Ojo... 
# Fíjate que, como WEKA, muestra scores por cada clase

# ¿qué efecto tiene el parámetro "tuneGrid"?
KNNModel2 <- train(variety ~ ., data=trainSet, method="knn", tuneGrid = expand.grid(k = c(1, 3, 5, 15, 19)), trControl=ctrl, preProc=c("scale"))
KNNModel2
plot(KNNModel2)

# En el proceso de aprendizaje: 
# ¿qué diferencia principal hay entre el de "KNNModel1" y "KNNModel2"?
# ¿Qué parámetro se ha tuneado en el proceso de aprendizaje? ¿Con qué valor definitivo se ha quedado, cada modelo?

KNNPredict2 <- predict(KNNModel2, newdata=testSet)
confusionMatrix(KNNPredict2, testSet$variety)

# Ahora, añade las líneas que aprendan&validen: un modelo naiveBayes, y otro de árboles de clasificación.
# En el siguiente enlace tienes los modelos, agrupados por familias, que se pueden aprender con el paquete "caret": 
# Fíjate en las familias "Bayesian model" y "Tres-based model": y escoge en ellas dos clasificadores que conozcamos
# https://topepo.github.io/caret/train-models-by-tag.html

