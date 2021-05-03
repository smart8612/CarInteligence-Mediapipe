import mediapipe as mp
import cv2
import csv


def initCSVFile(filename, results):
    num_coords = len(results.pose_landmarks.landmark) + len(results.face_landmarks.landmark)

    landmarks = ["class"]
    for val in range(1, num_coords + 1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
    print(landmarks)

    with open(filename, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)


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


if __name__ == '__main__':
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

            # Rendering Face Mesh and Pose (Mediapipe Holistic Solution)
            renderHolistic(mp_drawing, mp_holistic, results)

            # Show Image
            cv2.imshow('Initialize coords.csv Dataset file By pressing \'q\'', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                try:
                    initCSVFile('coords.csv', results)
                    break
                except:
                    print("Holistic Detection Fail")
                    continue

    cap.release()
    cv2.destroyAllWindows()