import cv2
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter("models/lite-model_midas_v2_1_small_1_lite_1.tflite")
interpreter.allocate_tensors()


def inference_depth(cv_image):
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]["shape"]
    inputHeight, inputWidth, channels, = (
        input_shape[1],
        input_shape[2],
        input_shape[3],
    )

    output_details = interpreter.get_output_details()
    output_shape = output_details[0]["shape"]
    outputHeight, outputWidth = output_shape[1], output_shape[2]

    img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # Input values should be from -1 to 1 with a size of 128 x 128 pixels for the fornt model
    # and 256 x 256 pixels for the back model
    img_input = cv2.resize(
        img, (inputWidth, inputHeight), interpolation=cv2.INTER_CUBIC
    ).astype(np.float32)

    # Scale input pixel values to -1 to 1
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_input = ((img_input / 255.0 - mean) / std).astype(np.float32)
    img_input = img_input[np.newaxis, :, :, :]

    # Peform inference
    interpreter.set_tensor(input_details[0]["index"], img_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    output = output.reshape(outputHeight, outputWidth)

    # Normalize estimated depth to have values between 0 and 255
    depth_min = output.min()
    depth_max = output.max()
    normalizedDisparity = (255 * (output - depth_min) / (depth_max - depth_min)).astype(
        "uint8"
    )

    # Resize disparity map to the sam size as the image inference
    estimatedDepth = cv2.resize(
        normalizedDisparity, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC
    )
    return estimatedDepth
