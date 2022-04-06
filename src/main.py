import sys
import cv2
import mediapipe as mp

import midas
import yolo
import hand_track


def main(image_filename):
    index = yolo.get_target_point(image_filename)
    print(index)

    cv_image = cv2.imread(image_filename)
    estimatedDepth = midas.inference_depth(cv_image)
    depth = estimatedDepth[index[0], index[1]]
    print(f"depth: {depth}")
    x, y = hand_track.get_middle_finger_mcp_point(cv_image)
    hand_depth = estimatedDepth[x, y]
    print(f"hand_depth: {hand_depth}")


if __name__ == "__main__":
    filename = sys.argv[1]
    main(filename)
