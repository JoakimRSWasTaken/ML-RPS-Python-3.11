# Runs Python 3.10

import json
import cv2
import numpy as np
import tensorflow as tf
#from ai_edge_litert import interpreter
from keras.src.saving import load_model
from tensorflow.keras.applications.convnext import preprocess_input

interpreter = tf.lite.Interpreter(model_path='convnext_best_rps_model_ever.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

lstm_model = load_model('user_move_prediction_model.keras', compile=False, safe_mode=False)

with open('class_names.json') as f:
    class_names = json.load(f)

def predict(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0).astype(np.float32) # (? hvorfor 32?) Vi laver et 224x224x3 billede om til et 1x224x224x3 billede
    img = preprocess_input(img)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke() # Her laver vi inference (når man predicter data i ML verdenen)
    output = interpreter.get_tensor(output_details[0]['index'])
    class_index = np.argmax(output[0])
    confidence = output[0][class_index] * 100 # Vi ganger med 100 for at få det i procent
    return class_names[class_index], confidence # Returnerer to værdier, navnet på klassen og confidence


def predict_next_move(move_history):
    """
    Predicts the next Rock-Paper-Scissors move using the LSTM model.

    Args:
        move_history: List of last 5 moves (strings: 'Rock', 'Paper', 'Scissors')

    Returns:
        Tuple of (predicted_move, confidence)
    """
    move_to_encoding = {
        'Rock': [1, 0, 0],
        'Paper': [0, 1, 0],
        'Scissors': [0, 0, 1]
    }
    encoding_to_move = {
        0: 'Rock',
        1: 'Paper',
        2: 'Scissors'
    }

    if len(move_history) < 5:
        return None, 0

    encoded_sequence = np.array([move_to_encoding[move] for move in move_history[-5:]], dtype=np.float32)
    encoded_sequence = np.expand_dims(encoded_sequence, axis=0)

    output = lstm_model.predict(encoded_sequence, verbose=0)

    predicted_index = np.argmax(output[0])
    confidence = output[0][predicted_index] * 100

    return encoding_to_move[predicted_index], confidence



vid = cv2.VideoCapture(0)

move_history = []

while (True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    class_name, confidence = predict(frame)
    display_text = "Object: {}  Confidence: {:.2f}%".format(class_name, confidence)


    # If needed, convert the frame to grayscale
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.putText(frame, 'You are currently showing a ' + display_text + 'Press Space to Play!', (10,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,(0,0,0), 2, 2)

    # Display the resulting frame
    cv2.imshow('Camera feed', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):  # Spacebar pressed
        move_history.append(class_name)
        if len(move_history) >= 5:
            predicted_move, pred_confidence = predict_next_move(move_history)
            print(f"Predicted next move: {predicted_move} (Confidence: {pred_confidence:.2f}%)")

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()