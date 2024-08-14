import io
import pickle
import numpy as np
import uvicorn
from keras.models import load_model
import PIL.Image
import PIL.ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
with open('model3.pkl', 'rb') as f:
  model = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/prdict/")
async def predict_image(file:UploadFile = File(...)):
  contents = await file.read()
  image = PIL.Image.open(io.BytesIO(contents)).convert('RGB')
  image = PIL.ImageOps.fit(image, (299, 299), PIL.Image.Resampling.LANCZOS)
  image_array = np.asarray(image)
  image_array = np.expand_dims(image_array, axis=0)
  image_array = image_array.astype('float32')
  image_array /= 255.0
  prediction = model.predict(image_array)
  if prediction[0] > 0.5:
    return {"prediction": "dog"}
  else:
    return {"prediction": "cat"}


