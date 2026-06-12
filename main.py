# Runs Python 3.10

import json
import cv2
import numpy as np
import tensorflow as tf
from keras.src.saving import load_model
from tensorflow.keras.applications.convnext import preprocess_input

# Initialize ConvNeXt model for gesture recognition
interpreter = tf.lite.Interpreter(model_path='convnext_best_rps_model_ever_v0.0.1.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize LSTM model for sequence prediction
lstm_model = load_model('user_move_prediction_model.keras', compile=False, safe_mode=False)

vid = cv2.VideoCapture(0)

move_history = []

# Screen text variables
last_lstm_result_text = ""
last_lstm_winner_text = ""
last_simple_result_text = ""
last_simple_winner_text = ""

# Statistics tracking
lstm_games = []  # Format: {'user': move, 'ai': move, 'result': text}
simple_games = []  # Format: {'user': move, 'ai': move, 'result': text}

with open('class_names.json') as f:
    class_names = json.load(f)


def inference(frame):
    """Performs inference on a frame using the ConvNeXt model."""
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


def predict_next_move_lstm(history):
    """Predicts the next move using the LSTM model (requires last 5 moves)."""
    move_to_encoding = {'paper': [0, 0, 1], 'rock': [1, 0, 0], 'scissors': [0, 1, 0]}
    encoding_to_move = {0: 'paper', 1: 'rock', 2: 'scissors'}

    if len(history) < 5:
        return None, 0

    encoded_sequence = np.array([move_to_encoding[move.lower()] for move in history[-5:]], dtype=np.float32)
    encoded_sequence = np.expand_dims(encoded_sequence, axis=0)

    output = lstm_model.predict(encoded_sequence, verbose=0)
    predicted_index = np.argmax(output[0])
    confidence = output[0][predicted_index] * 100

    return encoding_to_move[predicted_index], confidence


def predict_next_move_simple(history):
    """Predicts the least-used move so far (user's weakness)."""
    move_counts = {'rock': 0, 'paper': 0, 'scissors': 0}

    for move in history:
        m = move.lower()
        if m in move_counts:
            move_counts[m] += 1

    if len(history) < 5:
        return None, move_counts

    # Predicts the move the user plays LEAST, assuming AI should counter that or track pattern
    predicted_move = min(move_counts, key=move_counts.get)
    return predicted_move, move_counts


def determine_winner(user_move, ai_move):
    """Standard Rock-Paper-Scissors rules generator."""
    u = user_move.lower()
    a = ai_move.lower()

    if u == a:
        return "It's a tie!"
    if (u == 'rock' and a == 'paper') or \
            (u == 'paper' and a == 'scissors') or \
            (u == 'scissors' and a == 'rock'):
        return "AI wins!"
    return "You win!"


# AI move counter dictionary
move_counter = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}

while True:
    ret, frame = vid.read()
    if not ret:
        break

    class_name, confidence = inference(frame)

    # UI Text Overlays
    display_text = f"Current gesture: {class_name} ({confidence:.2f}%)"
    cv2.putText(frame, display_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
    cv2.putText(frame, "Press SPACE to play round", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)

    # Display LSTM results
    if last_lstm_result_text:
        cv2.putText(frame, last_lstm_result_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"LSTM Result: {last_lstm_winner_text}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 2)

    # Display Simple results
    if last_simple_result_text:
        cv2.putText(frame, last_simple_result_text, (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
        cv2.putText(frame, f"Simple Result: {last_simple_winner_text}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 200, 0), 2)

    cv2.imshow('Rock Paper Scissors AI', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' '):
        # 1. GET PREDICTIONS (Based on history BEFORE the new move is registered)
        lstm_pred, lstm_conf = predict_next_move_lstm(move_history)
        simple_pred, counts = predict_next_move_simple(move_history)

        # 2. DETERMINE AI MOVES
        # LSTM Choice
        if lstm_pred is None:
            lstm_ai_move = np.random.choice(['rock', 'paper', 'scissors'])
            last_lstm_result_text = f"LSTM: Random choice -> AI: {lstm_ai_move}"
        else:
            lstm_ai_move = move_counter[lstm_pred]
            last_lstm_result_text = f"LSTM: Pred {lstm_pred} ({lstm_conf:.1f}%) -> AI: {lstm_ai_move}"

        # Simple Choice
        if simple_pred is None:
            simple_ai_move = np.random.choice(['rock', 'paper', 'scissors'])
            last_simple_result_text = f"Simple: Random choice -> AI: {simple_ai_move}"
        else:
            simple_ai_move = move_counter[simple_pred]
            last_simple_result_text = f"Simple: Pred {simple_pred} -> AI: {simple_ai_move}"

        # 3. REGISTER THE USER'S ACTUAL MOVE
        user_actual_move = class_name
        move_history.append(user_actual_move)

        # 4. EVALUATE WINNERS
        last_lstm_winner_text = determine_winner(user_actual_move, lstm_ai_move)
        last_simple_winner_text = determine_winner(user_actual_move, simple_ai_move)

        # 5. TRACK HISTORY FOR STATISTICS
        lstm_games.append({'user': user_actual_move, 'ai': lstm_ai_move, 'result': last_lstm_winner_text})
        simple_games.append({'user': user_actual_move, 'ai': simple_ai_move, 'result': last_simple_winner_text})

        # 6. CALCULATE WIN RATES
        total_rounds = len(lstm_games)
        lstm_ai_wins = sum(1 for g in lstm_games if g['result'] == "AI wins!")
        simple_ai_wins = sum(1 for g in simple_games if g['result'] == "AI wins!")

        lstm_win_rate = (lstm_ai_wins / total_rounds * 100) if total_rounds > 0 else 0
        simple_win_rate = (simple_ai_wins / total_rounds * 100) if total_rounds > 0 else 0

        # 7. PRINT COMPARISON TO CONSOLE
        print("\n" + "=" * 50)
        print(f"ROUND #{total_rounds} SUMMARY")
        print(f"User played: {user_actual_move.upper()}")
        print("-" * 50)
        print(f"🤖 LSTM MODEL:")
        print(f"   Prediction: {str(lstm_pred).upper()} | AI Move: {lstm_ai_move.upper()}")
        print(f"   Outcome:    {last_lstm_winner_text}")
        print(f"   AI Winrate: {lstm_win_rate:.1f}% ({lstm_ai_wins}/{total_rounds})")
        print("-" * 50)
        print(f"📊 SIMPLE FREQUENCY MODEL:")
        print(f"   Prediction: {str(simple_pred).upper()} | AI Move: {simple_ai_move.upper()}")
        print(f"   Outcome:    {last_simple_winner_text}")
        print(f"   AI Winrate: {simple_win_rate:.1f}% ({simple_ai_wins}/{total_rounds})")
        print("=" * 50)

vid.release()
cv2.destroyAllWindows()