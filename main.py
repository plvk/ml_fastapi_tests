import logging
from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

class Item(BaseModel):
    text: str

app = FastAPI()
classifier = pipeline("sentiment-analysis")
logger = logging.getLogger(__name__)

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/predict/", response_model=dict)
def predict(item: Item):
    try:
        prediction = classifier(item.text)[0]
        logger.info(f"Prediction made for text: {item.text}")
        return prediction
    except Exception as e:
        logger.exception("An error occurred while making a prediction")
        return {"error": "An error occurred while making a prediction"}
