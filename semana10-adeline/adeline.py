import numpy as np
import matplotlib
import matplotlib.pyplot as plt
class Utils():
    def activation(input):
        return np.where(input>=0,1,-1)
    def error(input,target):
        return target.reshape(1,1)-input
    def graficar(data,pesos,bias):
        print(pesos)
        print(bias)
        x = data[:, 0]
        y = data[:, 1]
        plt.figure(figsize=(7, 4))
        plt.plot(x, y, 'bo')
        plt.axvline(x=0, ymin=-1, ymax=1)
        plt.axhline(y=0, xmin=-1, xmax=1)
        lx = [-1, 1]
        ly = []
        ly.append((-pesos[0, 0] * -1) / pesos[1, 0] - bias / pesos[1, 0])
        ly.append((-pesos[0, 0] * 1) / pesos[1, 0] - bias / pesos[1, 0])
        print(lx)
        print(ly)
        plt.plot(lx, ly, 'r')
        plt.show()
class Layer():
    optimizer=0
    def __init__(self,in_features, out_features, bias=True):
        self.weights=np.random.rand(in_features,out_features)
        self.bias=None
        if bias:
            self.bias=np.random.rand(out_features)
    def __call__(self,input):
        output=np.dot(input,self.weights)
        if self.bias is not None:
            output+=self.bias.reshape((1,1))
        return output
    def optimize(self,input,error):
        varianza=(input*error)
        self.weights=self.weights+((varianza.reshape((1,3)).T)*0.3)
        if self.bias is not None:
            self.bias+=(error.reshape(1)*0.3)
        
class Network():
    def __init__(self):
        self.fc1=Layer(in_features=3,out_features=1)
    def forward(self,input,labels):
        #input (1,3)
        #labels (1,)
        pred=self.fc1(input)
        #pred (1,1)
        error=Utils.error(pred,labels)
        #error (1,1)
        self.fc1.optimize(input,error)
        return pred
        
def entrenamiento(datosEntrada, datosSalida):
    salida = np.array([[0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0]]).T
    network=Network()
    epoca=0
    while (not (np.array_equal(salida, datosSalida))):
        print("======EPOCA ",epoca,"========")
        epoca+=1
        salidaEpoca = []
        numeroFila = 0
        for fila_epoca in datosEntrada:
            pred=network.forward(fila_epoca.reshape((1,3)),datosSalida[numeroFila,:])
            salidaEpoca.append(pred)
            numeroFila+=1
        salida = np.array([salidaEpoca]).reshape(7,1)

    print("output:",salida)
    print("weights:",network.fc1.weights)
    #Utils.graficar(datosEntrada,network.fc1.weights,network.fc1.bias)
def main():
    datosEntrenamiento = np.array([[0,0,1],
                                [0,1,0],
                                [0,1,1],
                                [1,0,0],
                                [1,0,1],
                                [1,1,0],
                                [1,1,1]])
    salidaEsperada = np.array([[1.0,2.0,3.0,4.0,5.0,6.0,7.0]]).T 
    entrenamiento(datosEntrenamiento,salidaEsperada)
main()