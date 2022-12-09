import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
#CONSTANTES
#Velocidad Inicial
v0=1
#h
h=1
#Número de iteraciones
nIter=1
#Omega
omg=1
#Ancho malla
nx=10
#Alto malla
ny=10
#Alto Viga
altoViga=2
#Ancho Viga
anchoViga=2
#Punto de la malla donde inicia la viga en x
inicioVigaX=int((nx-anchoViga)/2)
#Punto de la malla donde termina la viga en x
finalVigaX=int((nx+anchoViga)/2)
#Punto de la malla donde termina la viga en y (la viga nace en ny)
finalVigaY=ny-altoViga
#Velocidad en el eje x
vx=np.zeros((ny,nx))
#Velocidad en el eje y
vy=np.zeros_like(vx)
matrizRedondeada=np.zeros_like(vx)
#Presión
constantePresión=0.5
p=np.zeros_like(vx)

#Relajación Vx
for k in range(nIter+1):
    #Condiciones de frontera
    #inlet
    for j in range(ny):
        vx[j][0]=vx[j][1]
        vy[j][0]=0
    #outlet
    for j in range(ny):
        vx[j][nx-1]=vx[j][nx-2]
        vy[j][nx-1]=vy[j][nx-2]
    #surface
    for i in range(nx): 
        vx[0][i]=v0
        vy[0][i]=0
    #E line
    for i in range(inicioVigaX-1):
        vx[ny-1][i]=0
        vy[ny-1][i]=0
    #A line
    for i in range(finalVigaX+2,nx):
        vx[ny-1][i]=0
        vy[ny-1][i]=0
    #Viga
    for j in range(finalVigaY,ny):
        for i in range(inicioVigaX,finalVigaX+1):
            vx[j][i]=0
            vy[j][i]=0
    #Viga's back
    for j in range(finalVigaY,ny):
        vx[j][inicioVigaX-1]=0
        vy[j][inicioVigaX-1]=(-2*(vx[j][inicioVigaX-2]-vx[j][inicioVigaX-1]))/(h*h)
    #Viga's top
    for i in range(inicioVigaX,finalVigaX+1):
        vx[finalVigaY-1][i]=0
        vy[finalVigaY-1][i]=(-2*(vx[finalVigaY-2][i]-vx[finalVigaY-1][i]))/(h*h)
    #Viga's front
    for j in range(finalVigaY,ny):
        vx[j][finalVigaX+1]=0
        vy[j][finalVigaX+1]=(-2*(vx[j][finalVigaX+2]-vx[j][finalVigaX+1]))/(h*h)
    for j in range(ny):
        for i in range(nx):
            matrizRedondeada[j][i]=round(vx[j][i],2)
    print("Pre relajación:\n",vx)

    #Código relajación Vx
    if k<nIter:
        for j in range(1,ny-1):
            for i in range(1,nx-1):
                r=0.25*(vx[j][i+1]+vx[j][i-1]+vx[j+1][i]+vx[j-1][i]-(h/2)*(vx[j][i]*(vx[j][i+1]-vx[j][i-1])+vy[j][i]*(vx[j+1][i]-vx[j-1][i])+(p[j][i+1]-p[j][i-1])))-vx[j][i]
                vx[j][i]=vx[j][i]+omg*r
    
    print("Iteración número:",k)
#Relajación Vy
for k in range (nIter+1):
    #inlet
    for j in range(ny):
        vy[j][0]=0
    #outlet
    for j in range(ny):
        vy[j][nx-1]=vy[j][nx-2]
    #surface
    for i in range(nx):
        vy[0][i]=0
    #E line
    for i in range(inicioVigaX-1):
        vy[ny-1][i]=0
    #A line
    for i in range(finalVigaX+2,nx):
        vy[ny-1][i]=0
    #Viga
    for j in range(finalVigaY,ny):
        for i in range(inicioVigaX,finalVigaX+1):
            vy[j][i]=0
    #Viga's back
    for j in range(finalVigaY,ny):
        #vy[finalVigaY][inicioVigaX-1]=v0/2
        vy[j][inicioVigaX-1]=(-2*(vx[j][inicioVigaX-2]-vx[j][inicioVigaX-1]))/(h*h)
        
    #Viga's top
    for i in range(inicioVigaX,finalVigaX+1):
        #vy[finalVigaY-1][i]=0
        vy[finalVigaY-1][i]=(-2*(vx[finalVigaY-2][i]-vx[finalVigaY-1][i]))/(h*h)
        
    #Viga's front
    for j in range(finalVigaY,ny):
        #vy[finalVigaY][finalVigaX+1]=-v0/2
        vy[j][finalVigaX+1]=(-2*(vx[j][finalVigaX+2]-vx[j][finalVigaX+1]))/(h*h)
    print("Vy pre relajación:\n",vy)
    #Código relajación Vy
    if k<nIter:
        for j in range(1,ny-1):
            for i in range(1,nx-1):
                r=0.25*(vy[j][i+1]+vy[j][i-1]+vy[j+1][i]+vy[j-1][i]-(h/2)*(vx[j][i]*(vy[j][i+1]-vy[j][i-1])+vy[j][i]*(vy[j+1][i]-vy[j-1][i])+(p[j+1][i]-p[j-1][i])))-vy[j][i]
                vy[j][i]=vy[j][i]+omg*r
    print("Iteración número:",k)
#redondeo
for j in range(ny):
    for i in range(nx):
        matrizRedondeada[j][i]=round(vx[j][i],2)
print("vx:\n",matrizRedondeada)
print("vy:\n",vy)


def prueba(m):
    dfAGraficar=pd.DataFrame(m)
    plt.matshow(dfAGraficar)
    plt.colorbar()
prueba(vx)
prueba(vy)
#grafica
x=np.zeros_like(vx)
y=np.zeros_like(vx)
for j in range(ny):
    for i in range(nx):
        x[j][i]=i
        y[j][i]=ny-j
u=vx
v=vy

plt.figure()
plt.quiver(x,y,u,v)
plt.show()
sb.heatmap(vy)