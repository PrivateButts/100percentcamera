import cv2
import dlib
import numpy as np
from imutils import face_utils

# Initialize the face detector and facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    _, image = cap.read()

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    rects = detector(gray, 1)

    # Loop over the face detections
    for i, rect in enumerate(rects):
        # Determine the facial landmarks for the face region
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Define desired output coordinates
        desired_left_eye = (0.35, 0.35)
        desired_face_width = 256
        desired_face_height = 256

        if len(shape) < 46:  # Ensure that the landmarks for the eyes are detected
            continue

        # Get the center between the two eyes
        left_eye_center = (
            (shape[39][0] + shape[36][0]) // 2,
            (shape[39][1] + shape[36][1]) // 2,
        )
        right_eye_center = (
            (shape[45][0] + shape[42][0]) // 2,
            (shape[45][1] + shape[42][1]) // 2,
        )

        # Compute the angle between the eye centroids
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # Compute the desired right eye x-coordinate based on the desired x-coordinate of the left eye
        desired_right_eye_x = 1.0 - desired_left_eye[0]

        # Determine the scale of the new resulting image by taking the ratio of the distance between eyes
        dist = np.sqrt((dX**2) + (dY**2))
        desired_dist = desired_right_eye_x - desired_left_eye[0]
        desired_dist *= desired_face_width
        scale = desired_dist / dist

        # Compute center (x, y)-coordinates (i.e., the median point) between the two eyes in the input image
        if len(shape) > 45:  # Ensure that the landmarks for the eyes are detected
            left_eye_center = (
                (shape[39][0] + shape[36][0]) // 2,
                (shape[39][1] + shape[36][1]) // 2,
            )
            right_eye_center = (
                (shape[45][0] + shape[42][0]) // 2,
                (shape[45][1] + shape[42][1]) // 2,
            )
            eyes_center = (
                (left_eye_center[0] + right_eye_center[0]) // 2,
                (left_eye_center[1] + right_eye_center[1]) // 2,
            )
        else:
            continue  # Skip the current iteration if the landmarks are not detected

        # Grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # Update the translation component of the matrix
        tX = desired_face_width * 0.5
        tY = desired_face_height * desired_left_eye[1]
        M[0, 2] += tX - eyes_center[0]
        M[1, 2] += tY - eyes_center[1]

        # Apply the affine transformation
        (w, h) = (desired_face_width, desired_face_height)
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        # Display the aligned face
        cv2.imshow("Face", output)

    # Break the loop if 'esc' key is pressed
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
