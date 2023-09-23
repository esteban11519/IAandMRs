import numpy as np
import random
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
import glob, os

#plt.ion()

size = 180, 120

image=io.imread("D:/jpriva/Descargas/faptitud.png")/255.0

print("- Dimensiones de la imagen:")
print(image.shape)

imageRed = image[:,:,2]
print(imageRed.shape)

plt.imshow(imageRed,vmin=0,vmax=1)
plt.title("Canal Rojo")
plt.show()

#plt.imshow(image,vmin=0,vmax=1)
#plt.show()

#****************************************************
numCruce = 4
k = 2
x=120
y=180
l = 10
M = 0
pobNueva = np.zeros((k, y, x), dtype=np.int)
pobVieja = np.zeros((k, y, x), dtype=np.int)
apt = np.zeros((k, y, x), dtype='f')
prs = np.zeros((k, y, x), dtype='f')
aptitudMayor = np.zeros((y,x), dtype='f')
aptMayorIndex = np.zeros((y,x), dtype=np.int)


def imprimir():
    plt.imshow((pobNueva[0,:,:]/255.0),vmin=0,vmax=1)
    plt.show()

def generar():
    global pobNueva
    pobNueva = np.random.randint(256, size=(k, y, x))

def evalua():
    global aptMayorIndex
    a=0
    aptitudMayor = np.zeros((y,x), dtype='f')
    aptMayorIndex = np.zeros((y,x), dtype=np.int)
    for matrix in pobNueva:
        b=0
        for row in matrix:
            c=0
            for element in row:
                apt[a][b][c] = 255-abs(element-int(imageRed[b][c]*255))
                if apt[a][b][c]>aptitudMayor[b][c]:
                    aptitudMayor[b][c] = apt[a][b][c]
                    aptMayorIndex[b][c] = a
                c=c+1
            b=b+1
        a=a+1

def ordenar():
    global pobNueva
    a=0
    for row in aptMayorIndex:
        b=0
        for element in row:
            pobNueva[0][a][b] = pobVieja[element][a][b]
            b=b+1
        a=a+1
        
def seleccion():
    global pobNueva
    pobNueva[1:,:,:] = np.random.randint(256, size=(k-1, y, x))

generar()
evalua()
imprimir()


while M<15:
    pobVieja = np.copy(pobNueva)
    ordenar()
    imprimir()
    Image.fromarray(pobNueva[0,:,:].astype('uint8'), mode='L').save('D:/jpriva/OneDrive/Docs/UNAL/Inteligencia Artificial/Trabajo segunda parte/outfile'+str(M)+'.png')
    seleccion()
    evalua()
    M=M+1


