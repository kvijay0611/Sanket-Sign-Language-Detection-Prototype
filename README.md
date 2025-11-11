# Sanket — Sign Language Detection (Prototype)

A lightweight Flask web app that loads a pre-trained deep-learning model to detect sign language gestures and display the prediction from a simple browser UI.

This repository is intended as a **prototype** for quick demos and learning, not as a production service.

---

## ✨ Features

- Web UI built with **Flask** templates (`templates/`) and static assets (`static/`)
- Loads a pre-trained Keras/TensorFlow model from `Project.h5`
- Simple image preprocessing (OpenCV / NumPy) before inference
- Single-file backend (`app.py`) that exposes the inference endpoint
- Easy local run with `pip` and `Flask`

> Model architecture and training code are not included here; this repo focuses on **deployment & demo** of the trained model.

---

## 📁 Repository Structure

```
Sanket-Sign-Language-Detection-Prototype/
├─ app.py                # Flask app: loads model, routes, inference
├─ Project.h5            # Trained Keras/TensorFlow model
├─ requirements.txt      # Python dependencies
├─ templates/            # Jinja2 templates (HTML)
└─ static/               # CSS/JS/images
```

---

## 🚀 Quickstart

### 1) Clone and set up a virtual environment
```bash
git clone https://github.com/kvijay0611/Sanket-Sign-Language-Detection-Prototype.git
cd Sanket-Sign-Language-Detection-Prototype

# (Recommended) create a venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2) Install dependencies
If a `requirements.txt` is present, simply run:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> If installation fails on systems without a full toolchain, install build essentials (e.g., Windows Build Tools + CMake, or `build-essential` on Debian/Ubuntu).

### 3) Launch the app
```bash
# Common Flask entry pattern
export FLASK_APP=app.py         # (Windows PowerShell: $env:FLASK_APP = "app.py")
export FLASK_ENV=development    # optional, enables auto-reload
flask run

# or directly
python app.py
```
The server will print a local URL (typically `http://127.0.0.1:5000`). Open it in your browser.

---

## 🧠 Inference Pipeline (high level)

1. **Upload/Provide an image** from the UI (or webcam frame if implemented).
2. **Preprocess**: resize/normalize to the model’s expected input shape.
3. **Model predicts** a class label for the sign/gesture.
4. **Result displayed** on the web page with the predicted label and (optionally) score.

> Exact input size and class names depend on the model embedded in `Project.h5`. Check `app.py` for the preprocessing steps and label mapping.

---

## 🔧 Configuration

Common places to tweak:
- **Model path**: update the path to `Project.h5` if you move it.
- **Class labels**: ensure the label list / encoder in `app.py` matches the model outputs.
- **Max upload size** & allowed file types: adjust in Flask config if you add file uploads.
- **CORS**: enable only if you plan to call the API from a separate frontend.

---

## 📦 Packaging & Deployment

- **Local demo**: run with `flask run` or `python app.py`
- **Production**: behind a WSGI server (e.g., `gunicorn`) and a reverse proxy (e.g., Nginx)
- **Environment variables**: store secrets and configs outside source (e.g., `.env` + `python-dotenv`)

Example (Linux):
```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

---

## 🧪 Testing Tips

- Verify that sample images yield consistent predictions.
- Confirm the model input shape in `app.py` matches the trained model.
- Add basic unit tests for preprocessing and label decoding if you extend the repo.

---

## ⚠️ Notes

- `Project.h5` is a binary model file; Git LFS is recommended if it becomes large.
- If you retrain the model, keep the **label ordering** consistent with this app.
- This is a prototype; **do not** use it for critical decisions or accessibility services without rigorous validation.

---

## 🙏 Acknowledgements

- Flask for Python web microservices
- TensorFlow/Keras for deep learning
- OpenCV/NumPy for image processing
