import logging
import os
import sys
from operator import itemgetter

import spacy
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"model")

class Input(BaseModel):
    sentence: str

# load the model
try:
    nlp = spacy.load(model_path)
    logging.info("Loaded spacy model")
except Exception as e:
    logging.error("Cannot load spacy model")
    sys.exit(-1)

@app.get("/emotion")
def get_emotion(obj: Input):
    doc = nlp(obj.sentence)
    result = max(doc.cats.items(), key=itemgetter(1))
    return {"emotion": result[0], "score": result[1]}

if __name__ == "__main__":
    uvicorn.run("api:app")
