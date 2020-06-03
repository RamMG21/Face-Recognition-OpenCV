import cv2

## creamos una variable para abrir la camara de la maquina el 0 es el ID de la maquina
## 0 es por defecto a menos que sea una camara ajena a la maquina
web_cam = cv2.VideoCapture(0)

##Ruta de nuestro achivo XML que nos ayuda a detectar rosotros de frente
cascPath = "C:/Users/Ramses Moreno/Documents/GitHub/Proyecto_Desarrollo_Proyectos_inteligentes/face_recognitionOpenCv2-master/Cascades/haarcascade_frontalface_alt2.xml"

##Creamos una variable hacemos uso de OpenCV
##Le damos el metodo y le mandamos la ruta del archivo donde identificaremos los rostros
faceCascade = cv2.CascadeClassifier(cascPath)


##el contador nos dira la cantidad de imagenes que se encuentren en 
## en el set de datos
count = 0

while(True):
    ##Creamos una variable para la imagen de la ventada donde abrira la camara
    _, imagen_marco = web_cam.read()


    ##Creamos una escala de grises para pasarla al marco
    grises = cv2.cvtColor(imagen_marco, cv2.COLOR_BGR2GRAY)

    ## Detectamos los rostros faceCascade es el clasificador de OpenCV y detectMultiScale es el metodo de OpenCv 
    ##solo recive como para,etro la escala de grises
    rostro = faceCascade.detectMultiScale(grises, 1.5, 5)

    ##Cuatro variables para encerrar el rostro en un recuadro
    for(x,y,w,h) in rostro:
        ##Pintamos el recuadro con OpenCV y le eviamos la imagen del cuadro
        cv2.rectangle(imagen_marco, (x,y), (x+w, y+h), (255,0,0), 4)
        ##Iniciamos el contador y lo hacemos incremental
        count += 1
       
       ##Vamos a escribir en la ruta 
        cv2.imwrite("C:/Users/Ramses Moreno/Documents/GitHub/Proyecto_Desarrollo_Proyectos_inteligentes/face_recognitionOpenCv2-master/images/Ramses"+str(count)+".jpg", grises[y:y+h, x:x+w])
      
        ##Se abre el marco para mostrar la imagen de la camara
        cv2.imshow("Creando Dataset", imagen_marco)
    ##Validacion para romper el ciclo precionando la letra q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ##El contador llega a 400 nos guardara esa cantidad de imagenes
    elif count >= 400:
        break


## Cuando todo esta hecho, liberamos la captura
web_cam.release()
##Destruccion de las ventanas
cv2.destroyAllWindows()