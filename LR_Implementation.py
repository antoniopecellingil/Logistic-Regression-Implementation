#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
# ===================================================================
# Ampliación de Inteligencia Artificial, 2022-23
# PARTE I del trabajo práctico: Implementación de regresión logística
# Dpto. de CC. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autor(a) del trabajo: 
#
# APELLIDOS: Pecellin Gil
# NOMBRE: Antonio
#
# Segundo(a) componente (si se trata de un grupo):
#
# APELLIDOS:
# NOMBRE:
# ----------------------------------------------------------------------------


# ****************************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen. La discusión 
# y el intercambio de información de carácter general con los compañeros se permite, 
# pero NO AL NIVEL DE CÓDIGO. Igualmente el remitir código de terceros, OBTENIDO A TRAVÉS
# DE LA RED o cualquier otro medio, se considerará plagio. En particular no se 
# permiten implementaciones obtenidas con HERRAMIENTAS DE GENERACIÓN AUTOMÁTICA DE CÓDIGO. 
# Si tienen dificultades para realizar el ejercicio, consulten con el profesor. 
# En caso de detectarse plagio (previamente con aplicaciones anti-plagio o durante 
# la defensa, si no se demuestra la autoría mediante explicaciones convincentes), 
# supondrá una CALIFICACIÓN DE CERO en la asignatura, para todos los alumnos involucrados. 
# Sin perjuicio de las medidas disciplinarias que se pudieran tomar. 
# *****************************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, MÉTODOS
# Y ATRIBUTOS QUE SE PIDEN. EN PARTICULAR: NO HACERLO EN UN NOTEBOOK.

# NOTAS: 
# * En este trabajo NO SE PERMITE usar Scikit Learn (excepto las funciones que
#   se usan en carga_datos.py). 

# * SE RECOMIENDA y SE VALORA especialmente usar numpy. Las implementaciones 
#   saldrán mucho más cortas y eficientes, y se puntuarÁn mejor.   

import numpy as np

# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar todos los conjuntos de datos,
# basta con descomprimir el archivo datos-trabajo-aia.tgz y ejecutar el
# archivo carga_datos.py (algunos de estos conjuntos de datos se cargan usando
# utilidades de Scikit Learn, por lo que para que la carga se haga sin
# problemas, deberá estar instalado el módulo sklearn). Todos los datos se
# cargan en arrays de numpy:

# * Datos sobre concesión de prestamos en una entidad bancaria. En el propio
#   archivo datos/credito.py se describe con más detalle. Se carga en las
#   variables X_credito, y_credito.   

# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.  

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresita (republicano o demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos. 

# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer.
  
# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos. Los textos
#   se han vectorizado usando CountVectorizer de Scikit Learn, con la opción
#   binary=True. Como vocabulario, se han usado las 609 palabras que ocurren
#   más frecuentemente en las distintas críticas. La vectorización binaria
#   convierte cada texto en un vector de 0s y 1s en la que cada componente indica
#   si el correspondiente término del vocabulario ocurre (1) o no ocurre (0)
#   en el texto (ver detalles en el archivo carga_datos.py). Los datos se
#   cargan finalmente en las variables X_train_imdb, X_test_imdb, y_train_imdb,
#   y_test_imdb.    

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En digitdata.zip están todos los datos en formato
#   comprimido. Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles). 



# ==================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA 
# ==================================================

# Definir una función 

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test. La división ha de ser ALEATORIA y
# ESTRATIFICADA respecto del valor de clasificación. Por supuesto, en el orden 
# en el que los datos y los valores de clasificación respectivos aparecen en
# cada partición debe ser consistente con el orden original en X e y.   

# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:

# >>> Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
# >>> y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0]
#    (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos de
# datos, y la misma que en el total: 267/168=178/112=89/56

# >>> np.unique(y_votos,return_counts=True)
#  (array([0, 1]), array([168, 267]))
# >>> np.unique(ye_votos,return_counts=True)
#  (array([0, 1]), array([112, 178]))
# >>> np.unique(yp_votos,return_counts=True)
#  (array([0, 1]), array([56, 89]))

# La división en trozos es aleatoria y, por supuesto, en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con los datos del cáncer, en el que se observa que las proporciones
# entre clases se conservan en la partición. 
    
# >>> Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)

# >>> np.unique(y_cancer,return_counts=True)
# (array([0, 1]), array([212, 357]))

# >>> np.unique(yev_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yp_cancer,return_counts=True)
# (array([0, 1]), array([42, 71]))    


# Podemos ahora separar Xev_cancer, yev_cancer, en datos para entrenamiento y en 
# datos para validación.

# >>> Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)

# >>> np.unique(ye_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yv_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))


# Otro ejemplo con más de dos clases:

# >>> Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)

# >>> np.unique(y_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([202, 228, 220]))

# >>> np.unique(ye_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([121, 137, 132]))

# >>> np.unique(yp_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([81, 91, 88]))
# ------------------------------------------------------------------


## ---------- 

import random
def  particion_entr_prueba(X, y, test=0.2):
    
    X = np.array(X)
    y = np.array(y)
    size = len(X)
    
    # Verificamos si los valores de y ya son numéricos
    if not np.issubdtype(y.dtype, np.number):
        # Obtenemos los valores únicos de y
        valores_unicos = np.unique(y)
        
        # Pasamos las clases a numeros
        codificacion = {valor: i for i, valor in enumerate(valores_unicos)}
        y_codificado = np.array([codificacion[valor] for valor in y])
        print(codificacion)
        
    else:
        y_codificado = y
    
    #Calculamos la proporcion de clases
    
    clases , conteo = np.unique(y_codificado,return_counts= True)
    
    num_clases_test = {j:int(i*test) for j,i in zip(clases,conteo)}
    
    #Obtenemos los indices por clase
    indices_clases = {c: np.where(y_codificado == c) for c in clases}
    
    #Seleccionamos aleatoriamente los inidices para test
    
    indices_test = []
    for c in clases:
        indices_clase = indices_clases[c]
        #np.random.shuffle(indices_clase)
        seleccionados = random.sample(list(indices_clase[0]), num_clases_test[c])
        indices_test.extend(seleccionados)
        
    #Extraemos los indices de entrenamiento
    indices_entrenamiento = np.setdiff1d(range(size), indices_test)
    
    #Extraemos los conjuntos de entrenamiento y de test
    
    Xe = X[indices_entrenamiento]
    Xp = X[indices_test]
    ye = y_codificado[indices_entrenamiento]
    yp = y_codificado[indices_test]
    
    return Xe, Xp, ye, yp


# ===========================
# EJERCICIO 2: NORMALIZADORES
# ===========================

# En esta sección vamos a definir dos maneras de normalizar los datos. De manera 
# similar a como está diseñado en scikit-learn, definiremos un normalizador mediante
# una clase con un metodo "ajusta" (fit) y otro método "normaliza" (transform).


# ---------------------------
# 2.1) Normalizador standard
# ---------------------------

# Definir la siguiente clase que implemente la normalización "standard", es 
# decir aquella que traslada y escala cada característica para que tenga
# media 0 y desviación típica 1. 

# En particular, definir la clase: 


# class NormalizadorStandard():

#    def __init__(self):

#         .....
        
#     def ajusta(self,X):

#         .....        

#     def normaliza(self,X):

#         ......

# 


# donde el método ajusta calcula las corresondientes medias y desviaciones típicas
# de las características de X necesarias para la normalización, y el método 
# normaliza devuelve el correspondiente conjunto de datos normalizados. 

# Si se llama al método de normalización antes de ajustar el normalizador, se
# debe devolver (con raise) una excepción:

class NormalizadorNoAjustado(Exception): pass


# Por ejemplo:
    
    
# >>> normst_cancer=NormalizadorStandard()
# >>> normst_cancer.ajusta(Xe_cancer)
# >>> Xe_cancer_n=normst_cancer.normaliza(Xe_cancer)
# >>> Xv_cancer_n=normst_cancer.normaliza(Xv_cancer)
# >>> Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)

# Una vez realizado esto, la media y desviación típica de Xe_cancer_n deben ser 
# 0 y 1, respectivamente. No necesariamente ocurre lo mismo con Xv_cancer_n, 
# ni con Xp_cancer_n. 



# ------ 
class NormalizadorStandard():
    
    def __init__(self):
        self.media = None
        self.desviacion = None

    def ajusta(self,X):
        #Calculamos media y desviacion tipica de X
        self.media = np.mean(X, axis=0)
        self.desviacion = np.std(X, axis=0)
        
    def normaliza(self, X):
        #Comprobamos que el normalizador este ajustado
        if self.media is None or self.desviacion is None:
            raise(NormalizadorNoAjustado(ValueError("Para normalizar, llama antes al método ajusta")))
        
        #Devolvemos X normalizado
        return (X - self.media) / self.desviacion



# ------------------------
# 2.2) Normalizador MinMax
# ------------------------

# Hay otro tipo de normalizador, que consiste en asegurarse de que todas las
# características se desplazan y se escalan de manera que cada valor queda entre 0 y 1. 
# Es lo que se conoce como escalado MinMax

# Se pide definir la clase NormalizadorMinMax, de manera similar al normalizador 
# del apartado anterior, pero ahora implementando el escalado MinMax.

# Ejemplo:

# >>> normminmax_cancer=NormalizadorMinMax()
# >>> normminmax_cancer.ajusta(Xe_cancer)
# >>> Xe_cancer_m=normminmax_cancer.normaliza(Xe_cancer)
# >>> Xv_cancer_m=normminmax_cancer.normaliza(Xv_cancer)
# >>> Xp_cancer_m=normminmax_cancer.normaliza(Xp_cancer)

# Una vez realizado esto, los máximos y mínimos de las columnas de Xe_cancer_m
#  deben ser 1 y 0, respectivamente. No necesariamente ocurre lo mismo con Xv_cancer_m,
# ni con Xp_cancer_m. 


# ------ 

class NormalizadorMinMax():
    
    def __init__(self):
        self.min = None
        self.max = None

    def ajusta(self, X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        
    def normaliza(self, X):
        #Similar al anterior pero con la formula del normalizador MinMax
        if self.min is None or self.max is None:
            raise NormalizadorNoAjustado(ValueError("Para normalizar, llama antes al método ajusta"))
        return (X - self.min) / (self.max - self.min)








# ===========================================
# EJERCICIO 3: REGRESIÓN LOGÍSTICA MINI-BATCH
# ===========================================


# En este ejercicio se propone la implementación de un clasificador lineal 
# binario basado regresión logística (mini-batch), con algoritmo de entrenamiento 
# de descenso por el gradiente mini-batch (para minimizar la entropía cruzada).


# En concreto se pide implementar una clase: 

# class RegresionLogisticaMiniBatch():

#    def __init__(self,rate=0.1,rate_decay=False,n_epochs=100,
#                 batch_tam=64):

#         .....
        
#     def entrena(self,X,y,Xv=None,yv=None,n_epochs=100,salida_epoch=False,
#                     early_stopping=False,paciencia=3):

#         .....        

#     def clasifica_prob(self,ejemplos):

#         ......
    
#     def clasifica(self,ejemplo):
                        
#          ......



# * El constructor tiene los siguientes argumentos de entrada:



#   + rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#     durante todo el aprendizaje. Si rate_decay es True, rate es la
#     tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#   + rate_decay, indica si la tasa de aprendizaje debe disminuir en
#     cada epoch. En concreto, si rate_decay es True, la tasa de
#     aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#     con la siguiente fórmula: 
#        rate_n= (rate_0)*(1/(1+n)) 
#     donde n es el número de epoch, y rate_0 es la cantidad introducida
#     en el parámetro rate anterior. Su valor por defecto es False. 
#  
#   + batch_tam: tamaño de minibatch


# * El método entrena tiene como argumentos de entrada:
#   
#     +  Dos arrays numpy X e y, con los datos del conjunto de entrenamiento 
#        y su clasificación esperada, respectivamente. Las dos clases del problema 
#        son las que aparecen en el array y, y se deben almacenar en un atributo 
#        self.clases en una lista. La clase que se considera positiva es la que 
#        aparece en segundo lugar en esa lista.
#     
#     + Otros dos arrays Xv,yv, con los datos del conjunto de  validación, que se 
#       usarán en el caso de activar el parámetro early_stopping. Si son None (valor 
#       por defecto), se supone que en el caso de que early_stopping se active, se 
#       consideraría que Xv e yv son resp. X e y.

#     + n_epochs es el número máximo de epochs en el entrenamiento. 

#     + salida_epoch (False por defecto). Si es True, al inicio y durante el 
#       entrenamiento, cada epoch se imprime  el valor de la entropía cruzada 
#       del modelo respecto del conjunto de entrenamiento, y su rendimiento 
#       (proporción de aciertos). Igualmente para el conjunto de validación, si lo
#       hubiera. Esta opción puede ser útil para comprobar 
#       si el entrenamiento  efectivamente está haciendo descender la entropía
#       cruzada del modelo (recordemos que el objetivo del entrenamiento es 
#       encontrar los pesos que minimizan la entropía cruzada), y está haciendo 
#       subir el rendimiento.
# 
#     + early_stopping (booleano, False por defecto) y paciencia (entero, 3 por defecto).
#       Si early_stopping es True, dejará de entrenar cuando lleve un número de
#       epochs igual a paciencia sin disminuir la menor entropía conseguida hasta el momento
#       en el conjunto de validación 
#       NOTA: esto se suele hacer con mecanismo de  "callback" para recuperar el mejor modelo, 
#             pero por simplificar implementaremos esta versión más sencilla.  
#        



# * Método clasifica: recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY de clases que el modelo predice para esos ejemplos. 

# * Un método clasifica_prob, que recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY con las probabilidades que el modelo 
#   asigna a cada ejemplo de pertenecer a la clase positiva.       
    

# Si se llama a los métodos de clasificación antes de entrenar el modelo, se
# debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception): pass

        
  

# RECOMENDACIONES: 


# + IMPORTANTE: Siempre que se pueda, tratar de evitar bucles for para recorrer 
#   los datos, usando en su lugar funciones de numpy. La diferencia en eficiencia
#   es muy grande. 

# + Téngase en cuenta que el cálculo de la entropía cruzada no es necesario
#   para el entrenamiento, aunque si salida_epoch o early_stopping es True,
#   entonces si es necesario su cálculo. Tenerlo en cuenta para no calcularla
#   cuando no sea necesario.     

# * Definir la función sigmoide usando la función expit de scipy.special, 
#   para evitar "warnings" por "overflow":

#   from scipy.special import expit    
#
#   def sigmoide(x):
#      return expit(x)

# * Usar np.where para definir la entropía cruzada. 

# -------------------------------------------------------------

# Ejemplo, usando los datos del cáncer de mama (los resultados pueden variar):


# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)

# >>> lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer,yv_cancer)

# >>> lr_cancer.clasifica(Xp_cancer_n[24:27])
# array([0, 1, 0])   # Predicción para los ejemplos 24,25 y 26 

# >>> yp_cancer[24:27]
# array([0, 1, 0])   # La predicción anterior coincide con los valores esperado para esos ejemplos

# >>> lr_cancer.clasifica_prob(Xp_cancer_n[24:27])
# array([7.44297196e-17, 9.99999477e-01, 1.98547117e-18])



# Para calcular el rendimiento de un clasificador sobre un conjunto de ejemplos, usar la 
# siguiente función:
    
def rendimiento(clasif,X,y):
    return sum(clasif.clasifica(X)==y)/y.shape[0]

# Por ejemplo, los rendimientos sobre los datos (normalizados) del cáncer:
    
# >>> rendimiento(lr_cancer,Xe_cancer_n,ye_cancer)
# 0.9824561403508771

# >>> rendimiento(lr_cancer,Xp_cancer_n,yp_cancer)
# 0.9734513274336283




# Ejemplo con salida_epoch y early_stopping:

# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)

# >>> lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer_n,yv_cancer,salida_epoch=True,early_stopping=True)

# Inicialmente, en entrenamiento EC: 155.686323940485, rendimiento: 0.873972602739726.
# Inicialmente, en validación    EC: 43.38533009881579, rendimiento: 0.8461538461538461.
# Epoch 1, en entrenamiento EC: 32.7750241863029, rendimiento: 0.9753424657534246.
#          en validación    EC: 8.4952918658522,  rendimiento: 0.978021978021978.
# Epoch 2, en entrenamiento EC: 28.0583715052223, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.665719133490596, rendimiento: 0.967032967032967.
# Epoch 3, en entrenamiento EC: 26.857182744289368, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.09511082759361, rendimiento: 0.978021978021978.
# Epoch 4, en entrenamiento EC: 26.120803184993328, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.327991940213478, rendimiento: 0.967032967032967.
# Epoch 5, en entrenamiento EC: 25.66005010760342, rendimiento: 0.9808219178082191.
#          en validación    EC: 8.376171724729662, rendimiento: 0.967032967032967.
# Epoch 6, en entrenamiento EC: 25.329200890122557, rendimiento: 0.9808219178082191.
#          en validación    EC: 8.408704771704937, rendimiento: 0.967032967032967.
# PARADA TEMPRANA

# Nótese que para en el epoch 6 ya que desde la entropía cruzada obtenida en el epoch 3 
# sobre el conjunto de validación, ésta no se ha mejorado. 


# -----------------------------------------------------------------


from scipy.special import expit    
def sigmoide(x):
    return expit(x) 


class RegresionLogisticaMiniBatch():
   

    def _entropia_cruzada(self, y, y_pred):
        #Entropia de la forma de las transparencias de teoria
        #Le he añadido una pequeña constante porque me daba un warning de division by zero, dado que daba números demasiado pequeños
        #esto provocaba que apareciera inf en salida epoch.
        entropia = np.sum(np.where(y == 1, -np.log(y_pred + 1e-10), -np.log(1 - y_pred + 1e-10)))
        return entropia

    def __init__(self,rate=0.1,rate_decay=False,n_epochs=100,
                 batch_tam=64):
        self.rate = rate
        self.rate_decay = rate_decay
        self.n_epochs = n_epochs
        self.batch_tam = batch_tam
        self.clases = []


    def entrena(self,X,y,Xv=None,yv=None,n_epochs=100,salida_epoch=False,
                     early_stopping=False,paciencia=3):
        if Xv is None or yv is None:
            Xv = X
            yv = y
           
        #Extraemos las clases
        self.clases = np.unique(y).tolist()
        self.clases.sort()
        
        #Calculamos el numero de muestras, caracteristicas y clses
        num_filas, num_caracteristicas = X.shape
        
        #Inicializamos los parametros
        self.w = np.zeros(num_caracteristicas)
        self.bias = 0
        mejor_entropia = float('inf')
        rate_0 = self.rate
        
        for epoch in range(n_epochs):
        
            #Calculamos el rate_decay de ser necesario
            if self.rate_decay:
                # rate_n= (rate_0)*(1/(1+n)) 
                self.rate = rate_0*(1/(1+epoch))
            
            #Mezclamos los indices para la aleatoriedad
            indices_mezclados = np.random.permutation(num_filas)
            X = X[indices_mezclados]
            y = y[indices_mezclados]
            
            if salida_epoch and epoch == 0:
                entropia_entrenamiento = self._entropia_cruzada(y, sigmoide(np.dot(X, self.w) + self.bias))
                acc_entrenamiento = rendimiento(self, X, y)
                entropia_val = self._entropia_cruzada(yv, sigmoide(np.dot(Xv, self.w) + self.bias))
                acc_val = rendimiento(self, Xv, yv)
                print(f"Inicialmente, en entrenamiento EC: {entropia_entrenamiento}, rendimiento: {acc_entrenamiento}.")
                print(f"Inicialmente, en validación    EC: {entropia_val}, rendimiento: {acc_val}.")
                
            #Recorremos la muestra reordenada, tomando como salto el tamaño del batch
            for i in range(0, num_filas, self.batch_tam):
                
                X_batch = X[i:i+self.batch_tam]
                y_batch = y[i:i+self.batch_tam]
                
                #Calculammos el valor de la prediccion
                z = np.dot(X_batch, self.w) + self.bias
                y_pred = sigmoide(z)
                
                #Actualizamos los pesos y el w0
                self.w += self.rate * np.dot(X_batch.T, (y_batch - y_pred))
                self.bias += self.rate * np.sum(y_batch - y_pred)
                
            #Añadimos la salida de entrenamiento
            if salida_epoch:
                 #Calculamos la entropía cruzada y el acc
                  entropia_entrenamiento = self._entropia_cruzada(y, sigmoide(np.dot(X, self.w) + self.bias))
                  acc_entrenamiento = rendimiento(self, X, y)
                  entropia_val = self._entropia_cruzada(yv, sigmoide(np.dot(Xv, self.w) + self.bias))
                  acc_val = rendimiento(self, Xv, yv)
                  print(f"Epoch {epoch+1}, en entrenamiento EC: {entropia_entrenamiento}, rendimiento: {acc_entrenamiento}.")
                  print(f"         en validación    EC: {entropia_val}, rendimiento: {acc_val}.")
            
            #Definimos el early stopping
            if early_stopping:
                entropia_val = self._entropia_cruzada(yv, sigmoide(np.dot(Xv, self.w) + self.bias))
                if entropia_val < mejor_entropia:
                    mejor_entropia = entropia_val
                    contador_paciencia = 0
                else:
                    contador_paciencia += 1
                    if contador_paciencia >= paciencia:
                        print("PARADA TEMPRANA")
                        break
        
    def clasifica_prob(self, ejemplos):
        
        #Comprobamos que se haya entrenado previamente el clasificador
        if self.w is None or self.bias is None:
            raise ClasificadorNoEntrenado(ValueError("El clasificador no ha sido entrenado."))
        #Devolvemos la prediccion
        z = np.dot(ejemplos, self.w) + self.bias
        y_pred = sigmoide(z)
        return y_pred
    
    def clasifica(self, ejemplo):
        #Clasificamos en base a la prediccion
        prob = self.clasifica_prob(ejemplo)
        clases_pred = np.where(prob >= 0.5, self.clases[1], self.clases[0])
        return clases_pred





# ------------------------------------------------------------------------------





# =================================================
# EJERCICIO 4: IMPLEMENTACIÓN DE VALIDACIÓN CRUZADA
# =================================================



# Este jercicio puede servir para el ajuste de parámetros en los ejercicios posteriores, 
# pero si no se realiza, se podrían ajustar siguiendo el método "holdout" 
# implementado en el ejercicio 1


# Definir una función: 

#  rendimiento_validacion_cruzada(clase_clasificador,params,X,y,Xv=None,yv=None,n=5)

# que devuelve el rendimiento medio de un clasificador, mediante la técnica de
# validación cruzada con n particiones. Los arrays X e y son los datos y la
# clasificación esperada, respectivamente. El argumento clase_clasificador es
# el nombre de la clase que implementa el clasificador (como por ejemplo 
# la clase RegresionLogisticaMiniBatch). El argumento params es
# un diccionario cuyas claves son nombres de parámetros del constructor del
# clasificador y los valores asociados a esas claves son los valores de esos
# parámetros para llamar al constructor.

# INDICACIÓN: para usar params al llamar al constructor del clasificador, usar
# clase_clasificador(**params)  

# ------------------------------------------------------------------------------
# Ejemplo:
# --------
# Lo que sigue es un ejemplo de cómo podríamos usar esta función para
# ajustar el valor de algún parámetro. En este caso aplicamos validación
# cruzada, con n=5, en el conjunto de datos del cancer, para estimar cómo de
# bueno es el valor batch_tam=16 con rate_decay en regresión logística mini_batch.
# Usando la función que se pide sería (nótese que debido a la aleatoriedad, 
# no tiene por qué coincidir el resultado):

# >>> rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":16,"rate":0.01,"rate_decay":True},
#                                 Xe_cancer_n,ye_cancer,n=5)

# Partición: 1. Rendimiento:0.9863013698630136
# Partición: 2. Rendimiento:0.958904109589041
# Partición: 3. Rendimiento:0.9863013698630136
# Partición: 4. Rendimiento:0.9726027397260274
# Partición: 5. Rendimiento:0.9315068493150684
# >>> 0.9671232876712328




# El resultado es la media de rendimientos obtenidos entrenando cada vez con
# todas las particiones menos una, y probando el rendimiento con la parte que
# se ha dejado fuera. Las particiones DEBEN SER ALEATORIAS Y ESTRATIFICADAS. 
 
# Si decidimos que es es un buen rendimiento (comparando con lo obtenido para
# otros valores de esos parámetros), finalmente entrenaríamos con el conjunto de
# entrenamiento completo:

# >>> lr16=RegresionLogisticaMiniBatch(batch_tam=16,rate=0.01,rate_decay=True)
# >>> lr16.entrena(Xe_cancer_n,ye_cancer)

# Y daríamos como estimación final el rendimiento en el conjunto de prueba, que
# hasta ahora no hemos usado:
# >>> rendimiento(lr16,Xp_cancer_n,yp_cancer)
# 0.9646017699115044

#------------------------------------------------------------------------------


def rendimiento_validacion_cruzada(clase_clasificador,params,X,y,Xv=None,yv=None,n=5):
    assert n > 1, "El número de particiones debe ser mayor que 1"
    if Xv is None or yv is None:
        Xv = X
        yv = y
        
    #Obtenemos los indices de las diferentes clases para la estratificación
    #(Copiado en parte del ejercicio1)
    clases , conteo = np.unique(y,return_counts= True)
    #Obtenemos los indices por clase
    indices_clases = {c: np.where(y == c) for c in clases}
    for clase in indices_clases.keys():
        np.random.shuffle(indices_clases[clase][0])
        indices_clases[clase] = np.array_split(indices_clases[clase][0], n)
        
    #Acumulador de rendimientos para cada partición
    rendimientos = []
    
    for i in range(n):
        X_val = []
        y_val = []
        X_train = []
        y_train = []
        #Una vez hechas las particiones en los indices ordenados aleatoriamente solo dbeemos coger 1 para val y el resto para train por cada clase
        for clase in indices_clases.keys():
            X_val.extend(X[indices_clases[clase][i]])
            y_val.extend(y[indices_clases[clase][i]])
        #Obtenemos los indices de entrenamiento de forma que dejamos fuera la particion de validacion
            indices_train = np.concatenate(indices_clases[clase][:i] + indices_clases[clase][i+1:])
            X_train.extend(X[indices_train])
            y_train.extend(y[indices_train])
            
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        #Creamos el clasificador y medimos su rendimiento
        
        clasificador = clase_clasificador(**params)
        clasificador.entrena(X_train, y_train)

        rendimiento_part = rendimiento(clasificador, X_val, y_val)
        rendimientos.append(rendimiento_part)

        print(f"Partición {i+1}. Rendimiento: {rendimiento_part}")

    return np.mean(rendimientos)






# ===================================================
# EJERCICIO 5: APLICANDO LOS CLASIFICADORES BINARIOS
# ===================================================



# Usando la regeresión logística implementada en el ejercicio 2, obtener clasificadores 
# con el mejor rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama 
# - Críticas de películas en IMDB

# Ajustar los parámetros (tasa, rate_decay, batch_tam) para mejorar el rendimiento 
# (no es necesario ser muy exhaustivo, tan solo probar algunas combinaciones). 
# Si se ha hecho el ejercicio 4, usar validación cruzada para el ajuste 
# (si no, usar el "holdout" del ejercicio 1). 

# Mostrar el proceso realizado en cada caso, y los rendimientos finales obtenidos
# sobre un conjunto de prueba.     

# Mostrar también, para cada conjunto de datos, un ejemplo con salida_epoch, 
# en el que se vea cómo desciende la entropía cruzada y aumenta el 
# rendimiento durante un entrenamiento.     

# ----------------------------


#Empezaríamos cargando los datasets y dividiendolos en entrenamiento y prueba:
    
# >>> Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)
# >>> Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=0.2)
X_train_imdb=np.load("datos/imdb_sentiment/vect_train_text.npy")
X_test_imdb=np.load("datos/imdb_sentiment/vect_test_text.npy")
y_train_imdb=np.load("datos/imdb_sentiment/y_train_text.npy")
y_test_imdb=np.load("datos/imdb_sentiment/y_test_text.npy")

#Viendo las características imprimiendo algunas filas de los datasets vemos que el unico que tiene valores
# continuos es Xev_cancer por lo tanto es el que regularizaremos
# >>> normst_cancer=NormalizadorStandard()
# >>> normst_cancer.ajusta(Xev_cancer)
# >>> Xev_cancer_n=normst_cancer.normaliza(Xev_cancer)
# >>> Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)

#Una vez tenemos los datasets "preprocesados" comenzaremos ajustando parametros con el método de validación cruzada
#Probaré cada dataset en primer lugar con rate_decay true y rate_decay false


# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":16,"rate":0.01,"rate_decay":True},
#                                 Xev_cancer_n,yev_cancer,n=5)

# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":16,"rate":0.01,"rate_decay":False},
#                                 Xev_cancer_n,yev_cancer,n=5)

#Según la observacion es mejor uno u otro asi que para este dataset no es muy importante este parametro, por lo que lo dejare en False

# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":16,"rate":0.01,"rate_decay":True},
#                                 Xe_votos,ye_votos,n=5)

# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":16,"rate":0.01,"rate_decay":False},
#                                 Xe_votos,ye_votos,n=5)

#Mejor con False

# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":16,"rate":0.01,"rate_decay":True},
#                                 X_train_imdb,y_train_imdb,n=5)

# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":16,"rate":0.01,"rate_decay":False},
#                                 X_train_imdb,y_train_imdb,n=5)

#En este dataset tambien vemos que es mejor rate_decay = True

#Hecho esto pasaremos a comprobar el parametro batch_tam

# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":16,"rate":0.01,"rate_decay":False},
#                                 Xev_cancer_n,yev_cancer,n=5)

# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":32,"rate":0.01,"rate_decay":False},
#                                 Xev_cancer_n,yev_cancer,n=5)

# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":64,"rate":0.01,"rate_decay":False},
#                                 Xev_cancer_n,yev_cancer,n=5)

#Vemos que el mejor valor es bach_tam = 16, en esta observacion

# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":16,"rate":0.01,"rate_decay":False},
#                                 Xe_votos,ye_votos,n=5)

# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":32,"rate":0.01,"rate_decay":False},
#                                 Xe_votos,ye_votos,n=5)

# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":64,"rate":0.01,"rate_decay":False},
#                                 Xe_votos,ye_votos,n=5)

#En este caso el mejor valor es para batch_tam = 64

# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":16,"rate":0.01,"rate_decay":True},
#                                 X_train_imdb,y_train_imdb,n=5)

# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":32,"rate":0.01,"rate_decay":True},
#                                 X_train_imdb,y_train_imdb,n=5)

# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":64,"rate":0.01,"rate_decay":True},
#                                 X_train_imdb,y_train_imdb,n=5)

#Y, en este caso es mejor el batch_tam = 16

#Por último probaremos los valores de rate

# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":16,"rate":0.01,"rate_decay":False},
#                                 Xev_cancer_n,yev_cancer,n=5)

# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":16,"rate":0.1,"rate_decay":False},
#                                 Xev_cancer_n,yev_cancer,n=5)

#Vemos que es mejor en este caso rate de 0,01

# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":64,"rate":0.01,"rate_decay":False},
#                                 Xe_votos,ye_votos,n=5)

# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":64,"rate":0.1,"rate_decay":False},
#                                 Xe_votos,ye_votos,n=5)

#Vemos que es mejor en este caso rate de 0,01

# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":16,"rate":0.01,"rate_decay":True},
#                                 X_train_imdb,y_train_imdb,n=5)

# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":16,"rate":0.1,"rate_decay":True},
#                                 X_train_imdb,y_train_imdb,n=5)

#Vemos que es mejor en este caso rate de 0,01

#Por último vemos sus rendimientos finales sobre test.

# lr_cancer = RegresionLogisticaMiniBatch(batch_tam = 16,rate = 0.01,rate_decay = False)
# lr_cancer.entrena(Xev_cancer,yev_cancer,early_stopping = True, salida_epoch = True)
# rendimiento_cancer = rendimiento(lr_cancer, Xp_cancer, yp_cancer)
# print(rendimiento_cancer)
# 0.8141592920353983

# lr_votos = RegresionLogisticaMiniBatch(batch_tam = 64,rate = 0.01,rate_decay = False)
# lr_votos.entrena(Xe_votos,ye_votos,early_stopping = True, salida_epoch = True)
# rendimiento_votos = rendimiento(lr_votos, Xp_votos, yp_votos)
# print(rendimiento_votos)
# 0.9534883720930233

# lr_imdb = RegresionLogisticaMiniBatch(batch_tam = 16,rate = 0.01,rate_decay = True)
# lr_imdb.entrena(X_train_imdb,y_train_imdb,early_stopping = True, salida_epoch = True)
# rendimiento_imdb = rendimiento(lr_imdb, X_test_imdb,y_test_imdb)
# print(rendimiento_imdb)
# 0.805 

# =====================================================
# EJERCICIO 6: CLASIFICACIÓN MULTICLASE CON ONE vs REST
# =====================================================

# Se pide implementar un algoritmo de regresión logística para problemas de
# clasificación en los que hay más de dos clases, usando  la técnica One vs Rest. 


#  Para ello, implementar una clase  RL_OvR con la siguiente estructura, y que 
#  implemente un clasificador OvR (one versus rest) usando como base el
#  clasificador binario RegresionLogisticaMiniBatch


# class RL_OvR():

#     def __init__(self,rate=0.1,rate_decay=False,
#                   batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs=100,salida_epoch=False):

#        .......

#     def clasifica(self,ejemplos):

#        ......
            



#  Los parámetros de los métodos significan lo mismo que en el apartado
#  anterior, aunque ahora referido a cada uno de los k entrenamientos a 
#  realizar (donde k es el número de clases).
#  Por simplificar, supondremos que no hay conjunto de validación ni parada
#  temprana.  

 

#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------
# >>> Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)

# >>> rl_iris_ovr=RL_OvR(rate=0.001,batch_tam=8)

# >>> rl_iris_ovr.entrena(Xe_iris,ye_iris)

# >>> rendimiento(rl_iris_ovr,Xe_iris,ye_iris)
# 0.8333333333333334

# >>> rendimiento(rl_iris_ovr,Xp_iris,yp_iris)
# >>> 0.9
# --------------------------------------------------------------------

class RL_OvR():

    def __init__(self,rate=0.1,rate_decay=False,
                  batch_tam=64):
        
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.clasificadores = []

        
    def entrena(self,X,y,n_epochs=100,salida_epoch=False):
        
        #Extraemos las clases:
        clases = list(np.unique(y))
        clases.sort()
        
        for clase in clases:
            
            #Creamos una instancia del clasificador:
            clasificador = RegresionLogisticaMiniBatch(rate=self.rate, rate_decay=self.rate_decay,
                                                       n_epochs = n_epochs, batch_tam=self.batch_tam)

            # Transformamos los datos para aplicar OvR
            y_clase = np.where(y == clase, 1, 0)
            
            # Entrenamos el clasificador 
            clasificador.entrena(X, y_clase, salida_epoch=salida_epoch)
            
            # Agregamos el clasificador a la lista de clasificadores
            self.clasificadores.append(clasificador)

    
    def clasifica(self, ejemplos):
        #Comprobamos que se haya entrenado el clasificador
        if not self.clasificadores:
            raise ClasificadorNoEntrenado(ValueError("El clasificador no ha sido entrenado."))

        #Extraemos todas las probabilidades con el metodo del clasificador binario 
        probs = [clasificador.clasifica_prob(ejemplos) for clasificador in self.clasificadores]
        print(probs)
        #Con argmax devolvemos la clase que mayor probabilidad tiene.
        clases_pred = np.argmax(probs, axis=0)

        return clases_pred








            
# --------------------------------







# =================================
# EJERCICIO 7: CODIFICACIÓN ONE-HOT
# =================================


# Los conjuntos de datos en los que algunos atributos son categóricos (es decir,
# sus posibles valores no son numéricos, o aunque sean numéricos no hay una 
# relación natural de orden entre los valores) no se pueden usar directamente
# con los modelos de regresión logística, o con redes neuronales, por ejemplo.

# En ese caso es usual transformar previamente los datos usando la llamada
# "codificación one-hot". Básicamente, cada columna se reemplaza por k columnas
# en los que los valores psoibles son 0 o 1, y donde k es el número de posibles 
# valores del atributo. El valor i-ésimo del atributo se convierte en k valores
# (0 ...0 1 0 ...0 ) donde todas las posiciones son cero excepto la i-ésima.  

# Por ejemplo, si un atributo tiene tres posibles valores "a", "b" y "c", ese atributo 
# se reemplazaría por tres atributos binarios, con la siguiente codificación:
# "a" --> (1 0 0)
# "b" --> (0 1 0)
# "c" --> (0 0 1)    

# Definir una función:    
    
#     codifica_one_hot(X) 

# que recibe un conjunto de datos X (array de numpy) y devuelve un array de numpy
# resultante de aplicar la codificación one-hot a X.Por simplificar supondremos 
# que el array de entrada tiene todos sus atributos categóricos, y que por tanto 
# hay que codificarlos todos.

# Aplicar la función para obtener una codificación one-hot de los datos sobre
# concesión de prestamo bancario.     
  
# >>> Xc=np.array([["a",1,"c","x"],
#                  ["b",2,"c","y"],
#                  ["c",1,"d","x"],
#                  ["a",2,"d","z"],
#                  ["c",1,"e","y"],
#                  ["c",2,"f","y"]])
   
# >>> codifica_one_hot(Xc)
# 
# array([[1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
#        [0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0.],
#        [0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
#        [1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.],
#        [0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.],
#        [0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0.]])

# En este ejemplo, cada columna del conjuto de datos original se transforma en:
#   * Columna 0 ---> Columnas 0,1,2
#   * Columna 1 ---> Columnas 3,4
#   * Columna 2 ---> Columnas 5,6,7,8
#   * Columna 3 ---> Columnas 9, 10,11     

    
  

# -------- 


def codifica_one_hot(X):
    # Obtenemos la dimension de X
    num_filas, num_atributos = X.shape
    
    codificados = []
    
    for i in range(num_atributos):
        atributo = X[:, i]
        
        # Obtenemos los posibles valores
        valores = np.unique(atributo)
        
        # Creamos un array de 0s donde guardaremos la codificacion
        codificado = np.zeros((num_filas, len(valores)))
        
        # Iteramos sobre los valores posibles del atributo y asignamos 1 en la columna correspondiente
        for j, valor in enumerate(valores):
            codificado[:, j] = np.where(atributo == valor, 1, 0)
        
        # Agregamos el array codificado al acumulador
        codificados.append(codificado)
    
    # Concatenamos todos los arrays en 1
    resultado = np.concatenate(codificados, axis=1)
    
    return resultado









# =====================================================
# EJERCICIO 8: APLICACIONES DEL CLASIFICADOR MULTICLASE
# =====================================================


# ---------------------------------------------------------
# 8.1) Conjunto de datos de la concesión de crédito
# ---------------------------------------------------------

# Aplicar la implementación OvR Y one-hot de los ejercicios anteriores,
# para obtener un clasificador que aconseje la concesión, 
# estudio o no concesión de un préstamo, basado en los datos X_credito, y_credito. 

# Ajustar adecuadamente los parámetros (nuevamente, no es necesario ser demasiado 
# exhaustivo)

# ----------------------

#Comenzamos dividiendo el dataset en entrenamiento y prueba:
# Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.2)

#Por la naturaleza de los atributos le aplicaremos one-hot encoding
# Xe_credito_oh = codifica_one_hot(Xe_credito)
# Xp_credito_oh = codifica_one_hot(Xp_credito)

#Hecho esto vamos a pasar a comprobar combinaciones de parametros
#En el ejercicio 5 comprobé el mejor parametro, parametro a parametro y en este probaré con 
#combinaciones al azar.

# rendimiento_validacion_cruzada(RL_OvR,
#                                {"batch_tam":16,"rate":0.01,"rate_decay":True},
#                                 Xe_credito_oh,ye_credito,n=5)

# rendimiento_validacion_cruzada(RL_OvR,
#                                {"batch_tam":32,"rate":0.001,"rate_decay":False},
#                                 Xe_credito_oh,ye_credito,n=5)

# rendimiento_validacion_cruzada(RL_OvR,
#                                {"batch_tam":64,"rate":0.1,"rate_decay":True},
#                                 Xe_credito_oh,ye_credito,n=5)

#Vemos que nos da el mejor rendimiento, en esta observacion, sobre validacion con la ultima opcion.

#Por último comprobaremos sobre test.

# lr_credito = RL_OvR(batch_tam = 64, rate = 0.1, rate_decay = True)
# lr_credito.entrena(Xe_credito_oh,ye_credito, salida_epoch = True)
# rendimiento_credito = rendimiento(lr_credito, Xp_credito_oh, yp_credito)
# print(rendimiento_credito)
# 0.7131782945736435


# ---------------------------------------------------------
# 7.2) Clasificación de imágenes de dígitos escritos a mano
# ---------------------------------------------------------


#  Aplicar la implementación OvR anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en el
#  archivo digidata.zip que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador. 

#  Los datos están ya separados en entrenamiento, validación y prueba. En este
#  caso concreto, NO USAR VALIDACIÓN CRUZADA para ajustar, ya que podría
#  tardar bastante (basta con ajustar comparando el rendimiento en
#  validación). Si el tiempo de cómputo en el entrenamiento no permite
#  terminar en un tiempo razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test). 


# --------------------------------------------------------------------------

def lee_imagenes(ruta_fichero):
    imagenes = []
    with open(ruta_fichero, 'r') as f:
        lineas = [l.rstrip('\n') for l in f.readlines()]
        #Contamos que un digito son 28 lineas
        for i in range(0, len(lineas), 28):
            imagen = []
            for j in range(i, i+28):
                fila = np.where(np.isin(list(lineas[j]), ["+", "#"]), 1, 0)
                imagen.extend(fila)
            imagenes.append(imagen)
    return np.array(imagenes)
            

def lee_etiquetas(ruta_fichero):
    with open(ruta_fichero, 'r') as f:
        lineas = [l.rstrip('\n') for l in f.readlines()]
        etiquetas = [int(linea) for linea in lineas]
    return np.array(etiquetas)

# Comenzamos leyendo los archivos
# X_train = lee_imagenes('./datos/trainingimages')
# y_train = lee_etiquetas('./datos/traininglabels')
# X_val= lee_imagenes('./datos/validationimages')
# y_val = lee_etiquetas('./datos/validationlabels')
# X_test= lee_imagenes('./datos/testimages')
# y_test = lee_etiquetas('./datos/testlabels')

#Como los datos tienen una codificacion adecuada y estan ya divididos pasamos directamente al clasificador.

# lr_img = RL_OvR(batch_tam = 16, rate = 0.01, rate_decay = True)
# lr_img.entrena(X_train, y_train, salida_epoch = True)
# rendimiento_img = rendimiento(lr_img, X_test, y_test)
# print(rendimiento_img)
# 0.852

#Al parecer con el primer ajuste de parametros, obtenidos en base a la combinación que mejor solía funcionar en apartados 
#anteriores, ha resultado en un rendimiento que satisface el requisito.


