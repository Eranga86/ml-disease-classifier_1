from fastapi import FastAPI, File, UploadFile
import uvicorn
import joblib
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load your trained Random Forest model
# Make sure model.pkl is in the same folder as this file
model = joblib.load("best_paddy_rf_model.pkl")

# Example preprocessing (adjust based on how you trained your model)
def preprocess_image(image: Image.Image):
    image = image.resize((128, 128))  # resize to fixed size
    image = np.array(image) / 255.0   # normalize pixel values
    image = image.flatten().reshape(1, -1)  # flatten for RandomForest
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Preprocess
        processed = preprocess_image(image)

        # Run prediction
        prediction = model.predict(processed)[0]

        return {"prediction": str(prediction)}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
