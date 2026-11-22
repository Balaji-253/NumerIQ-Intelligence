# Handwritten Digit Recognizer

End-to-end MNIST CNN project: training, evaluation, FastAPI inference, Streamlit UI, tests, and Docker. CPU supported.

## Setup (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train and evaluate
```powershell
python -m src.train --epochs 5 --batch-size 128
python -m src.evaluate
```
MNIST downloads automatically into `data/` on the first training run. The best checkpoint is saved to `artifacts/mnist_cnn.pt`.

## Run the application
Terminal 1:
```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```
Open `http://127.0.0.1:8000/docs`.

Terminal 2:
```powershell
streamlit run ui/app.py
```
Open the local URL Streamlit prints. Draw or upload a digit, then click Recognize.

## Tests
```powershell
pytest -q
```

## Docker
Train the model first, then:
```powershell
docker compose up --build
```
UI: `http://localhost:8501`; API docs: `http://localhost:8000/docs`.

## Model
Input 1x28x28 grayscale; Conv(1→32) + ReLU + Pool; Conv(32→64) + ReLU + Pool; FC(3136→128→10). Cross-entropy loss, Adam optimizer.

The softmax score is a model confidence score, not a calibrated probability. MNIST is a benchmark dataset; this is an educational project, not a certified OCR system. Do not expose the development API publicly without security hardening.
