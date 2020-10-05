# importação das bibliotecas necessárias

# pybrain
from pybrain.datasets.supervised import SupervisedDataSet 
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer


# gráficos 
import matplotlib.pyplot as plt
import numpy as np

# função para carregar os dados de treinamento
def getData( path ):
    #Open file
    file = open( path, "r" )
    
    data = []    
    
    for linha in file:        # obtem cada linha do arquivo
      linha = linha.rstrip()  # remove caracteres de controle, \n
      digitos = linha.split(" ")  # pega os dígitos
      for numero in digitos:   # para cada número da linha
        data.append( numero )  # add ao vetor de dados  
    
    file.close()
    return data


# configurando a rede neural artificial e o dataSet de treinamento
network = buildNetwork( 45, 400, 400, 2 )    # define network 
dataSet = SupervisedDataSet( 45, 2 )  # define dataSet

'''
arquivos = ['1.txt', '1a.txt', '1b.txt', '1c.txt',
            '1d.txt', '1e.txt', '1f.txt']
'''  
arquivos = ['0.1.txt','0.2.txt','0.3.txt','1.1.txt','1.2.txt','1.3.txt','2.1.txt','2.2.txt','3.1.txt','3.2.txt','3.3.txt','4.1.txt','4.2.txt','4.3.txt','5.1.txt','5.2.txt','5.3.txt',
            '6.1.txt','6.2.txt','6.3.txt','7.1.txt','7.2.txt','7.3.txt','8.1.txt','8.2.txt','8.3.txt','9.1.txt','9.2.txt','9.3.txt']
# a resposta do número
resposta = [[0],[0],[0],[1],[1],[1],[2],[2],[2],[3],[3],[3],[4],[4],[4],[5],[5],[5],[6],[6],[6],[7],[7],[7],[8],[8],[8],[9],[9],[9] ]
#resposta = [[1], [1], [1], [1], [1], [1], [1]] 

i = 0
for arquivo in arquivos:           # para cada arquivo de treinamento
    data =  getData( arquivo )            # pegue os dados do arquivo
    dataSet.addSample( data, resposta[i] )  # add dados no dataSet
    i = i + 1


# trainer
trainer = BackpropTrainer( network, dataSet )
error = 1
iteration = 0
outputs = []
file = open("outputs.txt", "w") # arquivo para guardar os resultados

while error > 0.001: # 10 ^ -3
    error = trainer.train()
    outputs.append( error )
    iteration += 1    
    print ( iteration, error )
    file.write( str(error)+"\n" )

file.close()

# Fase de teste
arquivos = ['0- test.txt', '1- test.txt', '2- test.txt', '3- test.txt', '4- test.txt', '5- test.txt', '6- test.txt', '7- test.txt', '8- test.txt', '9- test.txt']
for arquivo in arquivos:
    data =  getData( arquivo )
    print ( network.activate( data ) )


# plot graph
plt.ioff()
plt.plot( outputs )
plt.xlabel('Iterações')
plt.ylabel('Erro Quadrático')
plt.show()

