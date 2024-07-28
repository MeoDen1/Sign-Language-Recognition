import cv2
import numpy as np
import mediapipe as mp

class HolisticModel:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic_model = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def apply_result(self, image, results):
        """
        Return image reuslt
        `image`: input image
        `result`: image's landmarks
        """
        self.mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            self.mp_holistic.FACEMESH_CONTOURS,

            self.mp_drawing.DrawingSpec(
                color=(255,0,255),
                thickness=1,
                circle_radius=1
            ),

            self.mp_drawing.DrawingSpec(
                color=(0,255,255),
                thickness=1,
                circle_radius=1
            )
        )

        # Drawing Right hand Landmarks
        self.mp_drawing.draw_landmarks(
            image, 
            results.right_hand_landmarks, 
            self.mp_holistic.HAND_CONNECTIONS
        )

        # Drawing Left hand Landmarks
        self.mp_drawing.draw_landmarks(
            image, 
            results.left_hand_landmarks, 
            self.mp_holistic.HAND_CONNECTIONS
        )

        # Drawing Pose Landmarks
        self.mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            self.mp_holistic.POSE_CONNECTIONS
        )

        return image

        
    def get_landmarks(self, image: np.array):
        """
        Get full x and y of 543 landmarks of image:
        `image`: input image \n
        Return: 
        `features`: an numpy array with shape (543 * 2, )
        `image`: processed image

        """
        x_list = []
        y_list = []

        # Making predictions using holistic model
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = self.holistic_model.process(image)
        image.flags.writeable = True

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

        return final, self.apply_result(image, results)
    