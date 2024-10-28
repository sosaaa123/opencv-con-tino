import cv2
import imutils
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#Ponemos a la haarcascade como variable
cap = cv2.VideoCapture(0)#Nuestra camara
image = cv2.imread('tino.png', cv2.IMREAD_UNCHANGED)
#image.shape =  (448, 900, 3)
#print('image.shape = ', image.shape)
#cv2.imshow('image', image[:,:,0])
while True:
    ret, frame = cap.read()#Capturamos el frame y comprobamos si la captura fue exitosa
    if not ret:
        break

    grises = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cara = face_cascade.detectMultiScale(grises,1.1,4)
    for(x,y,w,h) in cara:
        cv2.rectangle(frame, (x,y ), (x+w, y+h), (0,0,255), 2)

        resized_image = imutils.resize(image, width=w)
        filas_image = resized_image.shape[0]
        col_image = w
        porcion_alto = filas_image // 10
        if y - filas_image >= 0:
            n_frame = frame[y - filas_image + porcion_alto: y + porcion_alto, x: x + w]


            mask = resized_image[:,:,3]
            mask_inv = cv2.bitwise_not(mask)

            bg_black = cv2.bitwise_and(resized_image, resized_image, mask=mask)
            bg_black = bg_black[:,:,0:3]
            bg_frame = cv2.bitwise_and(n_frame,n_frame, mask=mask_inv)

            #Sumar las dos imagenes
            result = cv2.add(bg_black, bg_frame)
            frame[y - filas_image + porcion_alto: y + porcion_alto, x: x + w] = result
            #cv2.imshow('result',result)

    cv2.imshow('Frame', frame)
    k = cv2.waitKey(30)
    if k == 27:
        break
cap.release()