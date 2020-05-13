import cv2
import os
import numpy as np
from PIL import Image
import pickle

cascPath = "C:/Users/Ramses Moreno/Documents/GitHub/Proyecto_Desarrollo_Proyectos_inteligentes/face_recognitionOpenCv2-master/Cascades/haarcascade_frontalface_alt2.xml"
##Creamos la variable
faceCascade = cv2.CascadeClassifier(cascPath)

#reconocimiento con opencv
reconocimiento = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")


current_id = 0
etiquetas_id = {}
y_etiquetas = []
x_entrenamiento = []

