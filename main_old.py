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

vid = cv2.VideoCapture(0)

move_history = []
game_color = (0, 255, 0)

last_lstm_result_text = ""
last_lstm_winner_text = ""
last_simple_result_text = ""
last_simple_winner_text = ""
last_simple_prediction_text = ""

# game_results = []
lstm_game_results = []
simple_game_results = []

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


def predict_next_move_simple(move_history):
    """
    Non-ML predictor based on move frequency.
    - Counts all user moves
    - Starts predicting after 5 moves
    - Predicts the least-used move (user's weakness)

    Args:
        move_history: List of user's moves

    Returns:
        Tuple of (predicted_move, confidence, move_counts)
    """
    move_counts = {
        'rock': 0,
        'paper': 0,
        'scissors': 0
    }

    for move in move_history:
        m = move.lower()
        if m in move_counts:
            move_counts[m] += 1

    if len(move_history) < 5:
        return None, move_counts

    predicted_move = min(move_counts, key=move_counts.get)
    total = sum(move_counts.values())
    #confidence = ((total - move_counts[predicted_move]) / total * 100.0) if total > 0 else 0.0

    return predicted_move, move_counts

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


def update_game_history(winner_text, move_history, predicted_move, ai_move):
    """Tracks game outcomes and maintains win rate statistics.

    Args:
        winner_text: Result of the game ("You win!", "AI wins!", "It's a tie!")
        move_history: List of user's moves
        predicted_move: The move predicted by LSTM
        ai_move: The move chosen by AI

    Returns:
        Updated game_results list
    """
    game_outcome = {
        'user_move': move_history[-1] if move_history else 'None',
        'ai_move': ai_move,
        'result': winner_text,
        'predicted': predicted_move
    }
    return game_outcome

def determine_winner(user_move, ai_move):
    user_move = user_move.lower()
    ai_move = ai_move.lower()

    if user_move == ai_move:
        return "It's a tie!"
    if (user_move == 'rock' and ai_move == 'paper') or \
            (user_move == 'paper' and ai_move == 'scissors') or \
            (user_move == 'scissors' and ai_move == 'rock'):
        return "AI wins!"
    return "You win!"


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

    # if last_result_text:
    #     cv2.putText(frame, last_result_text, (10, 75),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.60, (0, 255, 0), 2, 2)
    #
    # if last_simple_prediction_text:
    #     cv2.putText(frame, last_simple_prediction_text, (10, 135),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.60, (255, 200, 0), 2, 2)
    #
    # if last_winner_text:
    #     cv2.putText(frame, last_winner_text, (220, 450),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 1.80, (0, 0, 255), 3, 3)

    cv2.putText(frame, last_lstm_result_text, (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2, 2)

    if last_lstm_winner_text:
        cv2.putText(frame, last_lstm_winner_text, (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 200, 255), 2, 2)

    if last_simple_result_text:
        cv2.putText(frame, last_simple_result_text, (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 200, 0), 2, 2)

    if last_simple_winner_text:
        cv2.putText(frame, last_simple_winner_text, (10, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 120, 0), 2, 2)

    cv2.imshow('Camera feed', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    # elif key == ord(' '):
    #     move_history.append(class_name)
    #
    #     if len(move_history) >= 5:
    #         lstm_predicted_move, lstm_pred_confidence = predict_next_move(move_history)
    #         ai_move, last_result_text, last_winner_text = play_game(lstm_predicted_move, lstm_pred_confidence, move_history)
    #     else:
    #         ai_move, last_result_text, last_winner_text = play_game(None, 0, move_history)
    #
    #     # Track game outcome
    #     game_results.append(
    #         update_game_history(last_winner_text, move_history, lstm_predicted_move if len(move_history) >= 5 else None,
    #                             ai_move))
    #
    #     # Print statistics
    #     wins = sum(1 for g in game_results if g['result'] == "You win!")
    #     total_games = len(game_results)
    #     win_rate = (wins / total_games * 100) if total_games > 0 else 0
    #
    #     print("move_histoy: ", move_history)
    #     print(f"\n=== Win Rate: {win_rate:.1f}% ({wins}/{total_games}) ===")
    #     print("Last 5 games:")
    #     for i, game in enumerate(game_results[-5:], 1):
    #         print(f"  {i}. You: {game['user_move']} | AI: {game['ai_move']} | {game['result']}")
    elif key == ord(' '):
        move_history.append(class_name)

        # Get LSTM prediction
        if len(move_history) >= 5:
            lstm_predicted_move, lstm_confidence = predict_next_move(move_history)
        else:
            lstm_predicted_move, lstm_confidence = None, 0

        # Get simple frequency prediction
        simple_predicted_move, move_counts = predict_next_move_simple(move_history)

        # Use LSTM prediction for AI move (existing logic)
        ai_move, last_result_text, last_winner_text = play_game(lstm_predicted_move, lstm_confidence, move_history)

        # For display of simple-predictor's AI move, compute counter
        move_counter = {
            'rock': 'paper',
            'paper': 'scissors',
            'scissors': 'rock'
        }
        if simple_predicted_move is None:
            simple_ai_move = "N/A"
        else:
            simple_ai_move = move_counter.get(simple_predicted_move.lower(), "N/A")

        if simple_predicted_move is None:
            last_simple_prediction_text = "Simple: (needs 5 moves)"
        else:
            last_simple_prediction_text = f"Simple: Pred {simple_predicted_move} -> AI: {simple_ai_move}"

        # Track game outcome
        game_results.append(
            update_game_history(last_winner_text, move_history, lstm_predicted_move, ai_move)
        )
        last_simple_result_text = "Simple: (needs5 moves)"
        last_simple_winner_text = ""
    else:
        last_simple_result_text = f"Simple: Pred {simple_predicted_move} -> AI: {simple_ai_move}"
        last_simple_winner_text = determine_winner(class_name, simple_ai_move)

        simple_game_results.append(
            update_game_history(last_simple_winner_text, move_history, simple_predicted_move, simple_ai_move)
        )

        # Print statistics with both predictions (unchanged)
        wins = sum(1 for g in game_results if g['result'] == "You win!")
        total_games = len(game_results)
        win_rate = (wins / total_games * 100) if total_games > 0 else 0

        print("move_history: ", move_history)
        print(f"\n=== Win Rate: {win_rate:.1f}% ({wins}/{total_games}) ===")
        print(f"LSTM Prediction: {lstm_predicted_move} (confidence: {lstm_confidence:.2f}%)")
        print(f"Simple Prediction: {simple_predicted_move}")
        print(
            f"Move counts -> rock: {move_counts['rock']} | paper: {move_counts['paper']} | scissors: {move_counts['scissors']}")
        print("Last 5 games:")
        for i, game in enumerate(game_results[-5:], 1):
            print(f"  {i}. You: {game['user_move']} | AI: {game['ai_move']} | {game['result']}")

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
