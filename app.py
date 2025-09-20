from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Answer keys as arrays (0->A, 1->B, 2->C, 3->D)
ANSWER_KEY_A = [
    0,2,2,2,2,0,2,2,2,2,
    0,3,3,0,1,0,2,3,1,1,
    0,3,1,0,2,1,2,1,3,2,
    2,0,2,2,0,1,2,1,3,3,
    1,2,2,0,2,1,2,2,3,1,
    1,2,0,1,2,1,0,0,1,1,
    2,2,0,2,2,1,2,2,2,1,
    1,0,1,1,2,1,1,1,1,2,
    1,0,2,1,2,1,1,1,2,2,
    0,1,2,1,2,1,2,0,2,2
]

ANSWER_KEY_B = [
    0, 2, 2, 2, 2, 0, 2, 2, 2, 2,
    0, 3, 3, 0, 1, 0, 2, 3, 1, 1,
    0, 3, 1, 0, 2, 1, 2, 1, 3, 2,
    2, 0, 2, 2, 0, 1, 2, 1, 3, 3,
    1, 2, 2, 0, 2, 1, 2, 2, 3, 1,
    1, 2, 0, 1, 2, 1, 0, 0, 1, 1,
    2, 2, 0, 2, 2, 1, 2, 2, 2, 1,
    1, 0, 1, 1, 2, 1, 1, 1, 1, 2,
    1, 0, 2, 1, 2, 1, 1, 1, 2, 2,
    0, 1, 2, 1, 2, 1, 2, 0, 2, 2
]

def evaluate_omr(image_path, set_choice='A'):
    if set_choice.upper() == 'A':
        ANSWER_KEY = ANSWER_KEY_A
    else:
        ANSWER_KEY = ANSWER_KEY_B

    if not ANSWER_KEY:
        return None, f"Answer key for set {set_choice} not found."

    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None, "Error: Could not load image."

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Find contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # Sort contours top-to-bottom
        questionCnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])

        # Assume 4 bubbles per question
        bubbles_per_question = 4
        questions = [questionCnts[i:i+bubbles_per_question] for i in range(0, len(questionCnts), bubbles_per_question)]
        questions = questions[:len(ANSWER_KEY)]  # Limit to answer key length

        correct_count = 0
        results = []

        for q_index, q_bubbles in enumerate(questions):
            if len(q_bubbles) != bubbles_per_question:
                continue

            q_bubbles = sorted(q_bubbles, key=lambda c: cv2.boundingRect(c)[0])
            bubble_values = []
            for bubble in q_bubbles:
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [bubble], -1, 255, -1)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                bubble_values.append(cv2.countNonZero(mask))

            chosen_option = int(np.argmax(bubble_values))
            correct_option = ANSWER_KEY[q_index]
            is_correct = chosen_option == correct_option
            if is_correct:
                correct_count += 1

            results.append({
                "question": q_index + 1,
                "chosen": chosen_option,
                "correct": is_correct
            })

        return {"score": correct_count, "details": results}, None

    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    set_choice = request.form.get('set_choice', 'A')
    file_path = os.path.join(UPLOAD_FOLDER, "omr.jpg")
    file.save(file_path)

    result, error = evaluate_omr(file_path, set_choice=set_choice)
    if error:
        return jsonify({"error": error}), 500
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
