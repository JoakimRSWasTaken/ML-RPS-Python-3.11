# Runs Python 3.10

import json
import cv2
import numpy as np
import tensorflow as tf
#from ai_edge_litert import interpreter
from keras.src.saving import load_model
from tensorflow.keras.applications.convnext import preprocess_input

interpreter = tf.lite.Interpreter(model_path='convnext_best_rps_model_ever_v0.0.1.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

lstm_model = load_model('user_move_prediction_model.keras', compile=False, safe_mode=False)

with open('class_names.json') as f:
    class_names = json.load(f)


def inference(frame):
    """
    Performs inference on a frame using the ConvNeXt model.

    Args:
        frame: Video frame from camera

    Returns:
        Tuple of (class_name, confidence)
    """
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    img = preprocess_input(img)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    class_index = np.argmax(output[0])
    confidence = output[0][class_index] * 100
    return class_names[class_index], confidence


def predict_next_move(move_history):
    """
    Predicts the next Rock-Paper-Scissors move using the LSTM model.

    Args:
        move_history: List of last 5 moves (strings: 'rock', 'paper', 'scissors')

    Returns:
        Tuple of (predicted_move, confidence)
    """
    move_to_encoding = {
        'paper': [0, 0, 1],
        'rock': [1, 0, 0],
        'scissors': [0, 1, 0]
    }
    encoding_to_move = {
        0: 'paper',
        1: 'rock',
        2: 'scissors'
    }

    if len(move_history) < 5:
        return None, 0

    encoded_sequence = np.array([move_to_encoding[move] for move in move_history[-5:]], dtype=np.float32)
    encoded_sequence = np.expand_dims(encoded_sequence, axis=0)

    output = lstm_model.predict(encoded_sequence, verbose=0)

    predicted_index = np.argmax(output[0])
    confidence = output[0][predicted_index] * 100

    return encoding_to_move[predicted_index], confidence

def play_game(predicted_move, pred_confidence, move_history):
    """
    Plays a round of Rock-Paper-Scissors against the user.

    Args:
        predicted_move: The move predicted by the LSTM model (or None)
        pred_confidence: Confidence of the prediction
        move_history: List of user's moves

    Returns:
        Tuple of (ai_move, result_text, winner_text)
    """
    move_counter = {
        'rock': 'paper',
        'paper': 'scissors',
        'scissors': 'rock'
    }

    # Choose AI move based on prediction
    if predicted_move is None:
        ai_move = np.random.choice(['rock', 'paper', 'scissors'])
        result_text = "AI chose randomly: {}".format(ai_move)
    else:
        ai_move = move_counter[predicted_move.lower()]
        result_text = "Prediction: {} (confidence: {:.2f}%) | AI chose: {}".format(
            predicted_move, pred_confidence, ai_move
        )

    # Determine winner
    winner_text = ""
    if len(move_history) > 0:
        user_move = move_history[-1].lower()
        if user_move == ai_move:
            winner_text = "It's a tie!"
        elif (user_move == 'rock' and ai_move == 'paper') or \
                (user_move == 'paper' and ai_move == 'scissors') or \
                (user_move == 'scissors' and ai_move == 'rock'):
            winner_text = "AI wins!"
        else:
            winner_text = "You win!"

    return ai_move, result_text, winner_text



vid = cv2.VideoCapture(0)

move_history = []
game_color = (0, 255, 0)
last_result_text = ""
last_winner_text = ""

while True:
    ret, frame = vid.read()

    class_name, confidence = inference(frame)
    display_text = "You are currently showing " + "{}  Confidence: {:.2f}%".format(class_name, confidence)

    cv2.putText(frame, display_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60, (0, 0, 0), 2, 2)

    prompt_text = "Press Space to Play!"
    cv2.putText(frame, prompt_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60, (0, 0, 0), 2, 2)

    # Always display the last game result if one exists
    if last_result_text:
        cv2.putText(frame, last_result_text, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (0, 255, 0), 2, 2)

    if last_winner_text:
        cv2.putText(frame, last_winner_text, (220, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.80, (0, 0, 255), 3, 3)

    cv2.imshow('Camera feed', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        move_history.append(class_name)
        if len(move_history) >= 5:
            predicted_move, pred_confidence = predict_next_move(move_history)
            ai_move, last_result_text, last_winner_text = play_game(predicted_move, pred_confidence, move_history)
        else:
            ai_move, last_result_text, last_winner_text = play_game(None, 0, move_history)

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
