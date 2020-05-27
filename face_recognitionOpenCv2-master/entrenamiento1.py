import cv2
import os
import numpy as np
from PIL import Image
import pickle

cascPath = "C:/Users/Ramses Moreno/Documents/GitHub/Proyecto_Desarrollo_Proyectos_inteligentes/face_recognitionOpenCv2-master/Cascades/haarcascade_frontalface_alt2.xml"
##Creamos la variable
faceCascade = cv2.CascadeClassifier(cascPath)

#Reconocimiento con opencv hacemos uso de la variable cv2 que tomamos de openCV cargamos el metodo para el reconocimiento.
reconocimiento = cv2.face.LBPHFaceRecognizer_create()

#Hacemos uso de os para jugar con las rutas para obtener partde de los archivos para depues pasarlo a las etiquetas 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")


current_id = 0
etiquetas_id = {}
y_etiquetas = []
x_entrenamiento = []

#haciendo un barrido y lectura de nuestros archivos que terminene con .jpg o .png
for root, dirs, archivos in os.walk(image_dir):
    for archivo in archivos:
        if archivo.endswith("png") or archivo.endswith("jpg"):
            pathImagen = os.path.join(root,archivo)
            etiqueta = os.path.basename(root).replace(" ", "-")#.lower()
            #print(etiqueta,pathImagen)

            #Creando las etiquetas
            if not etiqueta in etiquetas_id:                
                etiquetas_id[etiqueta] = current_id
                current_id += 1   
                #incrementando las vamos guardado en nuestro array de etiqueas de ID         
            id_ = etiquetas_id[etiqueta]
            #print(etiquetas_id)

            #abrimos la imagen
            pil_image = Image.open(pathImagen).convert("L")
            #le damos un tamaño
            tamanio = (550,550)
            imagenFinal = pil_image.resize(tamanio, Image.ANTIALIAS)
            image_array = np.array(pil_image,"uint8")
            #reconocemos los rostros haciendo uso de faceCascade como lo hiciemos en el dataset
            rostros = faceCascade.detectMultiScale(image_array, 1.5, 5)

            #Recorremos los rotros otra vez 
            for (x,y,w,h) in rostros:
                roi = image_array[y:y+h, x:x+w]
                x_entrenamiento.append(roi)
                y_etiquetas.append(id_)


with open("labels.pickle",'wb') as f:
    #Abrimos label y en ese archivo almacenamos las etiquetas con los ID
    pickle.dump(etiquetas_id, f)
#Hacemos el etranimiento en base al array convertdio a un array de numpy junto con las etiquetas y creamos el archivo entrenamiento.yml
reconocimiento.train(x_entrenamiento, np.array(y_etiquetas))
reconocimiento.save("entrenamiento.yml")