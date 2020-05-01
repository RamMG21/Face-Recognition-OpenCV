import cv2

## creamos una variable para abrir la camara de la maquina el 0 es el ID de la maquina
## 0 es por defecto a menos que sea una camara ajena a la maquina
web_cam = cv2.VideoCapture(0)

##Ruta de nuestro achivo XML que nos ayuda a detectar rosotros de frente
cascPath = "C:/Users/Ramses Moreno/Desktop/face_recognitionOpenCv2-master/Cascades/haarcascade_frontalface_alt2.xml"
##Creamos una variable hacemos uso de OpenCV
##Le damos el metodo y le mandamos la ruta del archivo donde identificaremos los rostros
faceCascade = cv2.CascadeClassifier(cascPath)


##el contador nos dira la cantidad de imagenes que se encuentren en 
## en el set de datos
count = 0



## Cuando todo esta hecho, liberamos la captura
web_cam.release()
##Destruccion de las ventanas
cv2.destroyAllWindows()