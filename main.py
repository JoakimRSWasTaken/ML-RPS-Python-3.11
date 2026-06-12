# Runs Python 3.10

import json
import cv2
import numpy as np
import tensorflow as tf
from keras.src.saving import load_model
from tensorflow.keras.applications.convnext import preprocess_input

# Initialize LSTM model for sequence prediction
# lstm_model = load_model('user_move_prediction_model.keras', compile=False, safe_mode=False)
# dnn_model = load_model('dnn_user_move_prediction_model.keras', compile=False, safe_mode=False)

# --- MODEL SELECTION AT STARTUP ---
print("=" * 50)
print("SELECT SEQUENCE PREDICTION MODEL")
print("1: LSTM Model (user_move_prediction_model.keras)")
print("2: DNN Model (dnn_user_move_prediction_model.keras)")
print("=" * 50)

model_choice = input("Enter your choice (1 or 2): ").strip()

# Initialize variables based on user selection
if model_choice == "2":
    print("\nLoading DNN model...")
    ml_model = load_model('dnn_user_move_prediction_model.keras', compile=False, safe_mode=False)
    selected_model_name = "DNN"
else:
    # Default choice if user presses 1 or enters anything else
    print("\nLoading LSTM model...")
    ml_model = load_model('user_move_prediction_model.keras', compile=False, safe_mode=False)
    selected_model_name = "LSTM"


# Initialize ConvNeXt model for gesture recognition
interpreter = tf.lite.Interpreter(model_path='convnext_best_rps_model_ever_v0.0.1.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



vid = cv2.VideoCapture(0)

move_history = []

# Screen text variables
last_ml_model_result_text = ""
last_ml_model_winner_text = ""
last_simple_result_text = ""
last_simple_winner_text = ""

# Statistics tracking
ml_model_games = []  # Format: {'user': move, 'ai': move, 'result': text}
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


def predict_next_move_ml_model(model, history):
    """Predicts the next move using the ML model (requires last 5 moves)."""
    move_to_encoding = {'paper': [0, 0, 1], 'rock': [1, 0, 0], 'scissors': [0, 1, 0]}
    encoding_to_move = {0: 'paper', 1: 'rock', 2: 'scissors'}

    if len(history) < 5:
        return None, 0

    encoded_sequence = np.array([move_to_encoding[move.lower()] for move in history[-5:]], dtype=np.float32)
    encoded_sequence = np.expand_dims(encoded_sequence, axis=0)

    output = model.predict(encoded_sequence, verbose=0)
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

    # cv2.putText()
    cv2.putText(frame, display_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.putText(frame, "Press SPACE to play round", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    # Display LSTM results
    # if last_lstm_result_text:
    #     cv2.putText(frame, last_lstm_result_text, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 189), 2)
    #     cv2.putText(frame, f"LSTM Result: {last_lstm_winner_text}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
    #                 (255, 0, 189), 2)

    # Display Active ML Model Results
    if last_ml_model_result_text:
        cv2.putText(frame, last_ml_model_result_text, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 189), 2)
        # Dynamically displays "LSTM Result:" or "DNN Result:" based on selection
        cv2.putText(frame, f"{selected_model_name} Result: {last_ml_model_winner_text}", (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 0, 189), 2)


    # Display Simple results
    if last_simple_result_text:
        cv2.putText(frame, last_simple_result_text, (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 24, 69), 2)
        cv2.putText(frame, f"Simple Result: {last_simple_winner_text}", (15, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (225, 24, 69), 2)

    cv2.imshow('Rock Paper Scissors AI', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' '):
        # 1. GET PREDICTIONS (Based on history BEFORE the new move is registered)
        ml_model_pred, ml_model_conf = predict_next_move_ml_model(ml_model, move_history)
        simple_pred, counts = predict_next_move_simple(move_history)

        # 2. DETERMINE AI MOVES
        # Model Choice
        if ml_model_pred is None:
            ml_model_ai_move = np.random.choice(['rock', 'paper', 'scissors'])
            last_ml_model_result_text = f"{selected_model_name}: Random choice -> AI: {ml_model_ai_move}"
        else:
            ml_model_ai_move = move_counter[ml_model_pred]
            last_ml_model_result_text = f"{selected_model_name}: Pred {ml_model_pred} ({ml_model_conf:.1f}%) -> AI: {ml_model_ai_move}"

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
        last_ml_model_winner_text = determine_winner(user_actual_move, ml_model_ai_move)
        last_simple_winner_text = determine_winner(user_actual_move, simple_ai_move)

        # 5. TRACK HISTORY FOR STATISTICS
        ml_model_games.append({'user': user_actual_move, 'ai': ml_model_ai_move, 'result': last_ml_model_winner_text})
        simple_games.append({'user': user_actual_move, 'ai': simple_ai_move, 'result': last_simple_winner_text})

        # 6. CALCULATE WIN RATES
        total_rounds = len(ml_model_games)
        ml_model_ai_wins = sum(1 for g in ml_model_games if g['result'] == "AI wins!")
        simple_ai_wins = sum(1 for g in simple_games if g['result'] == "AI wins!")

        ml_model_win_rate = (ml_model_ai_wins / total_rounds * 100) if total_rounds > 0 else 0
        simple_win_rate = (simple_ai_wins / total_rounds * 100) if total_rounds > 0 else 0

        print("\n" + "=" * 50)
        print(f"ROUND #{total_rounds} SUMMARY")
        print(f"User played: {user_actual_move.upper()}")
        print("-" * 50)
        print(f"🤖 {selected_model_name} MODEL:")
        print(f"   Prediction: {str(ml_model_pred).upper()} | AI Move: {ml_model_ai_move.upper()}")
        print(f"   Outcome:    {last_ml_model_winner_text}")
        print(f"   AI Winrate: {ml_model_win_rate:.1f}% ({ml_model_ai_wins}/{total_rounds})")
        print("-" * 50)
        print(f"📊 SIMPLE FREQUENCY MODEL:")
        print(f"   Prediction: {str(simple_pred).upper()} | AI Move: {simple_ai_move.upper()}")
        print(f"   Outcome:    {last_simple_winner_text}")
        print(f"   AI Winrate: {simple_win_rate:.1f}% ({simple_ai_wins}/{total_rounds})")
        print("=" * 50)

vid.release()
cv2.destroyAllWindows()