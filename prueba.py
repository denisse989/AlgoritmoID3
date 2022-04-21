# Algoritmo ID3 para construir un árbol de decisiones
import numpy as np
import math
#import uniout


#Crea un conjunto de datos
def createDataSet():
    dataSet = np.array([['juventud', 'No', 'No', 'No'],
                ['juventud', 'No', 'No', 'No'],
                ['juventud', 'si', 'No', 'si'],
                ['juventud', 'si', 'si', 'si'],
                ['juventud', 'No', 'No', 'No'],
                ['de edad mediana', 'No', 'No', 'No'],
                ['de edad mediana', 'No', 'No', 'No'],
                ['de edad mediana', 'si', 'si', 'si'],
                ['de edad mediana', 'No', 'si', 'si'],
                ['de edad mediana', 'No', 'si', 'si'],
                ['mayor', 'No', 'si', 'si'],
                ['mayor', 'No', 'si', 'si'],
                ['mayor', 'si', 'No', 'si'],
                ['mayor', 'si', 'No', 'si'],
                ['mayor', 'No', 'No', 'No']])
    features = ['años', 'tener un trabajo', 'Propia casa']
    return dataSet, features

#Calcular la entropía del conjunto de datos
def calcEntropy(dataSet):
    #Probabilidad primero
    labels = list(dataSet[:,-1])
    prob = {}
    entropy = 0.0
    for label in labels:
        prob[label] = (labels.count(label) / float(len(labels)))
    for v in prob.values():
        entropy = entropy + (-v * math.log(v,2))
    return entropy

#Conjunto de datos de partición
def splitDataSet(dataSet, i, fc):
    subDataSet = []
    for j in range(len(dataSet)):
        if dataSet[j, i] == str(fc):
            sbs = []
            sbs.append(dataSet[j, :])
            subDataSet.extend(sbs)
    subDataSet = np.array(subDataSet)
    return np.delete(subDataSet,[i],1)

#Calcule la ganancia de información, seleccione la mejor característica para dividir el conjunto de datos, es decir, devuelva el mejor índice de características
def chooseBestFeatureToSplit(dataSet):
    labels = list(dataSet[:, -1])
    bestInfoGain = 0.0   #Ganancia máxima de información
    bestFeature = -1   #*******
    #Extraiga la columna de características y la columna de etiquetas
    for i in range(dataSet.shape[1]-1):     #Columna
        #Calcule la probabilidad de cada categoría
        prob = {}
        featureCoulmnL = list(dataSet[:,i])
        for fcl in featureCoulmnL:
            prob[fcl] = featureCoulmnL.count(fcl) / float(len(featureCoulmnL))
        #Calcule la entropía de cada categoría
        new_entrony = {}    #Entropía de cada categoría
        condi_entropy = 0.0   #Entropía condicional
        featureCoulmn = set(dataSet[:,i])   #Columna de funciones
        for fc in featureCoulmn:
            subDataSet = splitDataSet(dataSet, i, fc)
            prob_fc = len(subDataSet) / float(len(dataSet))
            new_entrony[fc] = calcEntropy(subDataSet)   #Entropía de cada categoría
            condi_entropy = condi_entropy + prob[fc] * new_entrony[fc]    #Entropía condicional
        infoGain = calcEntropy(dataSet) - condi_entropy     #Calcular la ganancia de información
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#Si las características del conjunto de características están vacías, entonces T es un solo nodo, y la etiqueta de clase con el árbol de instancia más grande en el conjunto de datos D se usa como la etiqueta de clase del nodo, y se devuelve T
def majorityLabelCount(labels):
    labelCount = {}
    for label in labels:
        if label not in labelCount.keys():
            labelCount[label] = 0
        labelCount[label] += 1
    return max(labelCount)

#Construir árbol de decisión T
def createDecisionTree(dataSet, features):
    labels = list(dataSet[:,-1])
    #Si todas las instancias en el conjunto de datos pertenecen a la misma etiqueta de clase, T es un árbol de un solo nodo, y la etiqueta de clase se usa como etiqueta de clase del nodo, y se devuelve T
    if len(set(labels)) == 1:
        return labels[0]
    #Si las características del conjunto de características están vacías, entonces T es un solo nodo, y la etiqueta de clase con el árbol de instancia más grande en el conjunto de datos D se usa como la etiqueta de clase del nodo, y se devuelve T
    if len(dataSet[0]) == 1:
        return majorityLabelCount(labels)
    #De lo contrario, calcule la ganancia de información de cada característica en el conjunto de características para el conjunto de datos D de acuerdo con el algoritmo ID3, y seleccione la característica con la mayor ganancia de información, beatFeature
    bestFeatureI = chooseBestFeatureToSplit(dataSet)  #Subíndice de la mejor característica
    bestFeature = features[bestFeatureI]    #Mejor característica
    decisionTree = {bestFeature:{}} #Construya un árbol con la característica beatFeature con la mayor ganancia de información como nodo hijo
    
    del(features[bestFeatureI])    #Esta función se ha utilizado como un nodo secundario, elimínela para que pueda continuar construyendo el subárbol
    bestFeatureColumn = set(dataSet[:,bestFeatureI])
    for bfc in bestFeatureColumn:
        subFeatures = features[:]
        decisionTree[bestFeature][bfc] = createDecisionTree(splitDataSet(dataSet, bestFeatureI, bfc), subFeatures)
    return decisionTree

#Categorizar datos de prueba
def classify(testData, features, decisionTree):
    for key in decisionTree:
        index = features.index(key)
        testData_value = testData[index]
        subTree = decisionTree[key][testData_value]
        if type(subTree) == dict:
            result = classify(testData,features,subTree)
            return result
        else:
            return subTree


if __name__ == '__main__':
    dataSet, features = createDataSet()     #Crea un conjunto de datos
    decisionTree = createDecisionTree(dataSet, features)   #Construya un árbol de decisiones
    print ('decisonTree ',decisionTree)

    dataSet, features = createDataSet()     
    testData = ['mayor', 'si', 'No']
    result = classify(testData, features, decisionTree)  #Categorizar datos de prueba
    print ('Ya sea para dar',testData,'prestamo:',result)
    