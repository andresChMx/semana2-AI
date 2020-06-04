import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
class Backpropagation():
    def __init__(self,height,width):
        self.outputs=np.zeros((height,width))
    def add_output(self,column,output):
        self.outputs[0:output.size,column]=output
        
    def calculate_gradient_layer1(self,layer_num,fcl,error):
        learning_rate=0.3
        delta=0
        for i in range(len(fcl.weights)):
            for j in range(len(fcl.weights[i])):
                anext=(self.outputs[j,layer_num]*(1-self.outputs[j,layer_num]))
                delta=error*anext
                fcl.gradients[i,j]=delta*self.outputs[i,layer_num-1]*learning_rate
        for j in range(len(fcl.bias[0])):
            fcl.bias_gradient[0,j]=learning_rate*error*(self.outputs[j,layer_num]*(1-self.outputs[j,layer_num]))
        return delta
    def calculate_gradient_layer2(self,fcl2,fcl1,layer_num,delta):
        learning_rate=0.3
        for i in range(len(fcl1.weights)):
            for j in range(len(fcl1.weights[i])):
                tmp=delta*fcl2.weights[j,0]*(self.outputs[j,layer_num]*(1-self.outputs[j,layer_num]))
                fcl1.gradients[i,j]=tmp*learning_rate*self.outputs[i,layer_num-1]
        
        for j in range(len(fcl1.bias[0])):
            fcl1.bias_gradient[0,j]=delta*fcl2.weights[j,0]*(self.outputs[j,layer_num]*(1-self.outputs[j,layer_num]))*learning_rate
class Layer():
    def __init__(self,in_features, out_features, bias=True):
        self.weights=np.random.rand(in_features,out_features)
        self.gradients=np.zeros((in_features,out_features))
        self.bias=None
        if bias:
            self.bias=np.random.rand(1,out_features)
            self.bias_gradient=np.zeros((1,out_features))
    def __call__(self,input):
        output=np.dot(input,self.weights)
        if self.bias is not None:
            output+=self.bias
        return output
    def optimize(self):
        self.weights+=self.gradients
        if self.bias is not None:
            self.bias+=self.bias_gradient
        
class Network():
    def __init__(self):
        self.fc1=Layer(in_features=4,out_features=5) #
        self.fc2=Layer(in_features=5,out_features=1) #
        self.backward=Backpropagation(5,3) #
        self.layer_counter=0
    def forward(self,input,label,trainning=True): #
        pred=self.fc1(input)
        output=self.activation(pred)

        pred=self.fc2(output)
        output=self.activation(pred)
        
        if trainning:
            self.optimize(label,output)
            self.layer_counter=0
        return output
    def __call__(self,input,labels):
        self.backward.add_output(self.layer_counter,input)
        res=self.forward(input,labels)
        return res
    def activation(self,pred): # (1,-1)
        self.layer_counter+=1
        pred=np.squeeze(pred,axis=0)
        tmp=[(1 / (1 + math.exp(-i))) for i in pred]
        output=np.expand_dims(tmp,axis=0)
        self.backward.add_output(self.layer_counter,output)
        return output
    def optimize(self,label,output): #
        delta=self.backward.calculate_gradient_layer1(self.layer_counter,self.fc2,(label-output)) 
        self.backward.calculate_gradient_layer2(self.fc2,self.fc1,self.layer_counter-1,delta)
        self.fc1.optimize()
        self.fc2.optimize()
    
        
def entrenamiento(datosEntrada, datosSalida):
    salida = np.array([[0.0, 0.0, 0.0, 0.0]]).T
    network=Network()
    epoca=60
    while (not (np.array_equal(salida, datosSalida)) and epoca>0):
        epoca-=1
        salidaEpoca = []
        numeroFila = 0
        for fila_epoca in datosEntrada:
            pred=network(fila_epoca.reshape((1,-1)),datosSalida[numeroFila,:])
            salidaEpoca.append(pred)
            numeroFila+=1
        salida = np.array([salidaEpoca]).reshape(-1,1)
    print("salida",salida)
def scale_data(data):
    for i in range(len(data[0])):
        max_val=np.max(data[:,i])
        min_val=np.min(data[:,i])
        data[:,i]=(data[:,i]-min_val)/(max_val-min_val)
    return data
def main():
    datosEntrenamiento = np.array([5.1,3.5,1.4,0.2,4.9,3.0,1.4,0.2,4.7,3.2,1.3,0.2,4.6,3.1,1.5,0.2,5.0,3.6,1.4,0.2,5.4,3.9,1.7,0.4,
4.6,3.4,1.4,0.3,5.0,3.4,1.5,0.2,4.4,2.9,1.4,0.2,4.9,3.1,1.5,0.1,5.4,3.7,1.5,0.2,4.8,3.4,1.6,0.2,4.8,3.0,1.4,0.1,4.3,3.0,1.1,0.1,
5.8,4.0,1.2,0.2,5.7,4.4,1.5,0.4,5.4,3.9,1.3,0.4,5.1,3.5,1.4,0.3,5.7,3.8,1.7,0.3,5.1,3.8,1.5,0.3,5.4,3.4,1.7,0.2,5.1,3.7,1.5,0.4,
4.6,3.6,1.0,0.2,5.1,3.3,1.7,0.5,4.8,3.4,1.9,0.2,5.0,3.0,1.6,0.2,5.0,3.4,1.6,0.4,5.2,3.5,1.5,0.2,5.2,3.4,1.4,0.2,4.7,3.2,1.6,0.2,
4.8,3.1,1.6,0.2,5.4,3.4,1.5,0.4,5.2,4.1,1.5,0.1,5.5,4.2,1.4,0.2,4.9,3.1,1.5,0.1,5.0,3.2,1.2,0.2,5.5,3.5,1.3,0.2,4.9,3.1,1.5,0.1,4.4,3.0,1.3,0.2,
5.1,3.4,1.5,0.2,5.0,3.5,1.3,0.3,4.5,2.3,1.3,0.3,4.4,3.2,1.3,0.2,5.0,3.5,1.6,0.6,5.1,3.8,1.9,0.4,4.8,3.0,1.4,0.3,5.1,3.8,1.6,0.2,
4.6,3.2,1.4,0.2,5.3,3.7,1.5,0.2,5.0,3.3,1.4,0.2,7.0,3.2,4.7,1.4,6.4,3.2,4.5,1.5,6.9,3.1,4.9,1.5,5.5,2.3,4.0,1.3,6.5,2.8,4.6,1.5,
5.7,2.8,4.5,1.3,6.3,3.3,4.7,1.6,4.9,2.4,3.3,1.0,6.6,2.9,4.6,1.3,5.2,2.7,3.9,1.4,5.0,2.0,3.5,1.0,5.9,3.0,4.2,1.5,6.0,2.2,4.0,1.0,6.1,2.9,4.7,1.4,
5.6,2.9,3.6,1.3,6.7,3.1,4.4,1.4,5.6,3.0,4.5,1.5,5.8,2.7,4.1,1.0,6.2,2.2,4.5,1.5,5.6,2.5,3.9,1.1,5.9,3.2,4.8,1.8,6.1,2.8,4.0,1.3,
6.3,2.5,4.9,1.5,6.1,2.8,4.7,1.2,6.4,2.9,4.3,1.3,6.6,3.0,4.4,1.4,6.8,2.8,4.8,1.4,6.7,3.0,5.0,1.7,6.0,2.9,4.5,1.5,5.7,2.6,3.5,1.0,
5.5,2.4,3.8,1.1,5.5,2.4,3.7,1.0,5.8,2.7,3.9,1.2,6.0,2.7,5.1,1.6,5.4,3.0,4.5,1.5,6.0,3.4,4.5,1.6,6.7,3.1,4.7,1.5,6.3,2.3,4.4,1.3,
5.6,3.0,4.1,1.3,5.5,2.5,4.0,1.3,5.5,2.6,4.4,1.2,6.1,3.0,4.6,1.4,5.8,2.6,4.0,1.2,5.0,2.3,3.3,1.0,5.6,2.7,4.2,1.3,5.7,3.0,4.2,1.2,
5.7,2.9,4.2,1.3,6.2,2.9,4.3,1.3,5.1,2.5,3.0,1.1,5.7,2.8,4.1,1.3,6.3,3.3,6.0,2.5,5.8,2.7,5.1,1.9,7.1,3.0,5.9,2.1,6.3,2.9,5.6,1.8,
6.5,3.0,5.8,2.2,7.6,3.0,6.6,2.1,4.9,2.5,4.5,1.7,7.3,2.9,6.3,1.8,6.7,2.5,5.8,1.8,7.2,3.6,6.1,2.5,6.5,3.2,5.1,2.0,6.4,2.7,5.3,1.9,
6.8,3.0,5.5,2.1,5.7,2.5,5.0,2.0,5.8,2.8,5.1,2.4,6.4,3.2,5.3,2.3,6.5,3.0,5.5,1.8,7.7,3.8,6.7,2.2,7.7,2.6,6.9,2.3,6.0,2.2,5.0,1.5,6.9,3.2,5.7,2.3,
5.6,2.8,4.9,2.0,7.7,2.8,6.7,2.0,6.3,2.7,4.9,1.8,6.7,3.3,5.7,2.1,7.2,3.2,6.0,1.8,6.2,2.8,4.8,1.8,6.1,3.0,4.9,1.8,6.4,2.8,5.6,2.1,
7.2,3.0,5.8,1.6,7.4,2.8,6.1,1.9,7.9,3.8,6.4,2.0,6.4,2.8,5.6,2.2,6.3,2.8,5.1,1.5,6.1,2.6,5.6,1.4,7.7,3.0,6.1,2.3,6.3,3.4,5.6,2.4,
6.4,3.1,5.5,1.8,6.0,3.0,4.8,1.8,6.9,3.1,5.4,2.1,6.7,3.1,5.6,2.4,6.9,3.1,5.1,2.3,5.8,2.7,5.1,1.9,6.8,3.2,5.9,2.3,6.7,3.3,5.7,2.5,6.7,3.0,5.2,2.3,
6.3,2.5,5.0,1.9,6.5,3.0,5.2,2.0,6.2,3.4,5.4,2.3,5.9,3.0,5.1,1.8])

    salidaEsperada = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,
2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0])
    datosEntrenamiento=datosEntrenamiento.reshape((150,4))
    salidaEsperada=salidaEsperada.reshape((150,1))
    
    datosEntrenamiento=scale_data(datosEntrenamiento)
    salidaEspedara=scale_data(salidaEsperada)
    entrenamiento(datosEntrenamiento,salidaEsperada)
main()
