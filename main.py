from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import joblib
import numpy as np
from PIL import Image
import io

# Load trained model
model = joblib.load("model.pkl")

app = FastAPI()

@app.get("/")
def root():
    return {"message": "ML Disease Classifier API is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
        image = image.resize((64, 64))  # adjust size as per your training
        img_array = np.array(image).flatten().reshape(1, -1)

        # Predict
        prediction = model.predict(img_array)
        return {"prediction": str(prediction[0])}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
