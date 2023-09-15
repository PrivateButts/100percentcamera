from imutils import face_utils
import dlib
import cv2

# Vamos inicializar um detector de faces (HOG) para então
# let's go code an faces detector(HOG) and after detect the
# landmarks on this detected face

# p = our pre-treined model directory, on my case, it's on the same script's diretory.
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)

while True:
    # Getting out image by webcam
    _, image = cap.read()
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get faces into webcam's image
    rects = detector(gray, 1)
    faces = dlib.full_object_detections()
    for detection in rects:
        faces.append(predictor(image, detection))

    if len(faces) == 0:
        continue

    image = dlib.get_face_chip(image, faces[0], size=720, padding=0)

    # Show the image
    cv2.imshow("Output", image)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
