import cv2
import numpy as np
import mediapipe as mp

# Initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils


def __display(image, results, time_show):
    # Display
    # Drawing the Facial Landmarks
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,

        mp_drawing.DrawingSpec(
            color=(255,0,255),
            thickness=1,
            circle_radius=1
        ),

        mp_drawing.DrawingSpec(
            color=(0,255,255),
            thickness=1,
            circle_radius=1
        )
    )

    # Drawing Right hand Landmarks
    mp_drawing.draw_landmarks(
        image, 
        results.right_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS
    )

    # Drawing Left hand Landmarks
    mp_drawing.draw_landmarks(
        image, 
        results.left_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS
    )

    # Drawing Pose Landmarks
    mp_drawing.draw_landmarks(
        image, 
        results.pose_landmarks, 
        mp_holistic.POSE_CONNECTIONS
    )
    
    # Display the resulting image
    cv2.imshow("Landmarks", image)

    if cv2.waitKey(time_show):
        pass


def getLandmarks(image: np.array, show_landmarks=True, time_show=5):
    """
    Take `image` as image input and return x, y coordinate array of 543 landmarks.
    """
    x_list = []
    y_list = []

    image = cv2.resize(image, (800, 600))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Making predictions using holistic model
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #
    # Extract point
    #
    # Extract face
    if results.face_landmarks:
        for idx, landmark in enumerate(results.face_landmarks.landmark):
            x = float(landmark.x)
            y = float(landmark.y)

            x_list.append(x)
            y_list.append(y)

    else:
        x_list.append(0 for i in range(468))
        y_list.append(0 for i in range(468))

    
    if results.left_hand_landmarks:
        for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
            x = float(landmark.x)
            y = float(landmark.y)

            x_list.append(x)
            y_list.append(y)

    else:
        x_list.append(0 for i in range(21))
        y_list.append(0 for i in range(21))


    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            x = float(landmark.x)
            y = float(landmark.y)

            x_list.append(x)
            y_list.append(y)

    else:
        x_list.append(0 for i in range(33))
        y_list.append(0 for i in range(33))    


    if results.right_hand_landmarks:
        for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
            x = float(landmark.x)
            y = float(landmark.y)

            x_list.append(x)
            y_list.append(y)

    else:
        x_list.append(0 for i in range(21))
        y_list.append(0 for i in range(21))

    final = np.array(x_list + y_list)

    if show_landmarks:
        __display(image, results, time_show)

    return final