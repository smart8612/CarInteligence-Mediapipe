import mediapipe as mp
import cv2
import numpy as np
import pickle
import pandas as pd


def renderHolistic(mp_drawing, mp_holistic, results):
    # 1. Draw face landmarks
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )

    # 2. Right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )

    # 3. Left Hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )

    # 4. Pose Detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    modelFileName = input("Write Model File: ")

    with open(modelFileName, 'rb') as f:
        global model
        model = pickle.load(f)

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(0)

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Make Detections
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                # Rendering Face Mesh and Pose (Mediapipe Holistic Solution)
                renderHolistic(mp_drawing, mp_holistic, results)

                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                row = pose_row + face_row

                realtime_x = pd.DataFrame([row])
                body_language_class = model.predict(realtime_x)[0]
                body_language_prob = model.predict_proba(realtime_x)[0]
                print(body_language_class, body_language_prob)

                coords_left = tuple(np.multiply(
                    np.array(
                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x,
                         results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y))
                    , [1280, 720]).astype(int))

                coords_right = tuple(np.multiply(
                    np.array(
                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x,
                         results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y))
                    , [1280, 720]).astype(int))

                cv2.rectangle(image,
                              (coords_left[0], coords_left[1]+5),
                              (coords_left[0]+len(body_language_class)*20, coords_left[1]-35),
                              (255, 255, 0), -1)
                cv2.putText(image, body_language_class, coords_left,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.rectangle(image,
                              (coords_right[0], coords_right[1]+5),
                              (coords_right[0] + len(str(round(body_language_prob[np.argmax(body_language_prob)], 2))) * 20, coords_right[1]-35),
                              (245, 117, 16), -1)
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), coords_right,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            except:
                print("Holistic Detection Fail")

            cv2.imshow('Raw Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("finish")
                break

    cap.release()
    cv2.destroyAllWindows()
