import numpy as np
import matplotlib.pyplot as plt

#Velocidad Inicial
v0=1
#h = magnitud en la que recorremos la matriz
h=1
#Ancho malla
nx=5
#Alto malla
ny=5
#matriz U
U = np.zeros((ny,nx), float)
#velocidad inicial 
v0 = 1
# Viga 1
inicio = 4
alto1 = 4
ancho1 = 3
# Viga 2
inicio2 = 7
alto2 = 4
ancho2 = 3


#NOTA: J son columnas e 

#Rellenar Obtaculos dentro de la matriz
# def rellenar(m1):
#     for i in range(nx - alto1, nx):
#         for j in range(inicio, inicio + ancho1):
#             m1[i, j] = 1

#     for i in range(0, alto2):
#          for j in range(inicio2, inicio2 + ancho2):
#              m1[i, j] = 1
# rellenar(U)      

arreglo = np.ones((nx,ny), float)

# matriz, indice en i, indice en j -> array
# Proposito : Devuelve los valores que debería tener la posicion, segun el valor del cuadrante superior, inferior, trasero y delantero
def array(u,i,j):
    
    fila = np.zeros((ny,nx))
    fila[i,j] = 1
    if(j == 0):       
        fila[i,j+1] = 1/4 - 1/8*u[i,j]       
    elif(j+1 >= nx):
        fila[i,j-1] = 1/4 + 1/8*u[i,j]   
    else:
        fila[i,j+1] = 1/4 - 1/8*u[i,j]
        fila[i,j-1] = 1/4 + 1/8*u[i,j]   

    if(i == 0):
        fila[i+1,j] = 1/4
    elif(i+1 >= ny):
        fila[i-1,j] = 1/4
    else:
        fila[i+1,j] = 1/4
        fila[i-1,j] = 1/4
        
    return fila
    
#derivada(arreglo,0,0)

matriz1 = array(arreglo,1,0)
#matriz = array(derivada(arreglo,0,0)[0],derivada(arreglo,0,0)[1],derivada(arreglo,0,0)[2],derivada(arreglo,0,0)[3],derivada(arreglo,0,0)[4],0,1)

def concatArrays(matriz):
    fila = []
    for i in range(nx):
        fila = np.concatenate((fila, matriz[i]), axis=None)
    return fila 

#arr = [concatArrays(matriz1),concatArrays(matriz)]
print(matriz1)