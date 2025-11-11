from flask import Flask, render_template, Response
import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

app = Flask(__name__, static_url_path='/static', static_folder='static')

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils



def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):

    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])


def draw_styled_landmarks(image, results):
    
                             
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                             )
    
# Initialize detection variables
sequence = []
sentence = []
threshold = 0.96

DATA_PATH = os.path.join('Project_Set') 

# Actions that we try to detect
actions = np.array(['HELLO','THANKS','I LOVE YOU'])
# actions = np.array(['N'])
# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30



model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.load_weights('Project.h5')



colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (255, 0, 0),(245, 117, 16)]  # Adding a new color tuple

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return output_frame



def generate_frames():
    global sequence
    global sentence
    global video_feed_active
    sequence = []
    sentence = []

    while True:
        cap = cv2.VideoCapture(0)
        width, height = 1400, 1000
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                if not video_feed_active:
                    break

                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)

                # Your prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                res = None
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])

                    # Viz probabilities
                    image = prob_viz(res, actions, image, colors)

                draw_styled_landmarks(image, results)

                # cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                # cv2.putText(image, ' '.join(sentence), (3, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                _, buffer = cv2.imencode('.jpg', image)
                if not _:
                    continue
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()
        cv2.destroyAllWindows()


@app.route('/start_video',methods=['POST'])
def start_video():
    global video_feed_active
    video_feed_active = True
    return 'Video feed started.'

@app.route('/stop_video',methods=['POST'])
def stop_video():
    global video_feed_active
    global cap
    global sentence
    global sequence
    video_feed_active = False
    sequence = []
    sentence = []
    return "Video feed stopped."


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
