import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


class HandNotFoundError(Exception):
    pass


def get_middle_finger_mcp_point(cv_image):
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:
        # image = cv2.flip(cv_image, 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        # Print handedness and draw hand landmarks on the image.
        # print("Handedness:", results.multi_handedness)
        if not results.multi_hand_landmarks:
            raise HandNotFoundError()
        image_height, image_width, _ = cv_image.shape
        # annotated_image = cv_image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            # print('hand_landmarks:', hand_landmarks)
            # print(
            #     f'MIDDLE_FINGER_MCP coordinates: (',
            #     f'{}, '
            #     f'{)'
            # )
            mid_x = int(
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
                * image_width
            )
            mid_y = int(
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
                * image_height
            )
            return mid_x, mid_y
    raise HandNotFoundError()
