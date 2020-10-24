import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/{id}")
def home(id):
    return {"Hello": str(id)+"World"}

if __name__ == "__main__":
    uvicorn.run("fastapi_test:app")