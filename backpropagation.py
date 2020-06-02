import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math


class Backpropagation():
    def __init__(self,height,width):
        self.outputs=np.zeros((height,width))
    def add_output(self,column,output):
        self.outputs[0:output.size,column]=output
        
    def calculate_gradient_matrix_layer1(self,layer_num,fcl,error):
        learning_rate=0.9
        triangle=0
        for i in range(len(fcl.weights)):
            for j in range(len(fcl.weights[i])):
                triangle=error*(self.outputs[j,layer_num]*(1-self.outputs[j,layer_num]))
                fcl.gradients[i,j]=triangle*self.outputs[i,layer_num-1]*learning_rate
        #print(fcl.gradients)
        return triangle
    def calculate_gradient_matrix_layer2(self,fcl2,fcl1,layer_num,delta):
        learning_rate=0.9
        for i in range(len(fcl1.weights)):
            for j in range(len(fcl1.weights[i])):
                tmp=delta*fcl2.weights[j,0]*(self.outputs[j,layer_num]*(1-self.outputs[j,layer_num]))
                fcl1.gradients[i,j]=tmp*learning_rate*self.outputs[i,layer_num-1]
        
class Utils():
    def activation(input):
        return np.where(input>=0,1,-1)
    def error(input,target):
        return target.reshape(1,-1)-input
class Layer():
    optimizer=0
    def __init__(self,in_features, out_features, bias=True):
        self.weights=np.random.rand(in_features,out_features)
        self.gradients=np.zeros((in_features,out_features))
        self.bias=None
        if bias:
            self.bias=np.random.rand(1,out_features)
    def __call__(self,input):
        output=np.dot(input,self.weights)
        #print(output.shape)
        if self.bias is not None:
            output+=self.bias
        return output
    def optimize(self):
        self.weights+=self.gradients
        
class Network():
    def __init__(self):
        self.fc1=Layer(in_features=2,out_features=2,bias=False) #
        self.fc2=Layer(in_features=2,out_features=1,bias=False) #
        self.backward=Backpropagation(2,3) #
        self.layer_counter=0
    def forward(self,input,label): #
        #input (1,-1) rank 2
        #labels (1,) rank 1
        pred=self.fc1(input)
        output=self.activation(pred)

        pred=self.fc2(output)
        output=self.activation(pred)
        
        self.optimize(label,output)
        #error=Utils.error(pred,labels)
        #error (1,1)
        #self.fc1.optimize(input,error) #calculate matrix backpropagation
        # step() perform weights update
        self.layer_counter=0
        return output
    def __call__(self,input,labels):
        self.backward(self.layer_couter,input)
        self.forward(input,labels)
    def activation(self,pred): # (1,-1)
        self.layer_counter+=1
        pred=np.squeeze(pred,axis=0)
        tmp=[(1 / (1 + math.exp(-i))) for i in pred]
        output=np.expand_dims(tmp,axis=0)
        self.backward.add_output(self.layer_counter,output)
        return output
    def optimize(self,label,output): #
        delta=self.backward.calculate_gradient_matrix_layer1(self.layer_counter,self.fc2,label-output) 
        self.backward.calculate_gradient_matrix_layer2(self.fc2,self.fc1,self.layer_counter-1,delta)
        self.fc1.optimize()
        self.fc2.optimize()
        
def entrenamiento(datosEntrada, datosSalida):
    salida = np.array([[0.0, 0.0, 0.0, 0.0]]).T
    network=Network()
    epoca=0
    while (not (np.array_equal(salida, datosSalida))):
        print("======EPOCA ",epoca,"========")
        epoca+=1
        salidaEpoca = []
        numeroFila = 0
        for fila_epoca in datosEntrada:
            pred=network.forward(fila_epoca.reshape((1,-1)),datosSalida[numeroFila,:])
            salidaEpoca.append(pred)
            numeroFila+=1
        salida = np.array([salidaEpoca]).reshape(-1,1)
        print("salida",salida)
    #print("output:",salida)
    #print("weights:",network.fc1.weights)
    #Utils.graficar(datosEntrada,network.fc1.weights,network.fc1.bias)
def main():
    datosEntrenamiento = np.array([[0,0],
                                [1,0],
                                [0,1],
                                [1,1]])#7,3
    salidaEsperada = np.array([[0.0],
                               [1.0],
                               [1.0],
                               [0.0]]) # 7,1
    entrenamiento(datosEntrenamiento,salidaEsperada)
main()