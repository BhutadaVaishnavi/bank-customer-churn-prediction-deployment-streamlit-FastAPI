from fastapi import FastAPI
import numpy as np
import pickle
from pydantic import BaseModel

# Load trained model
model = pickle.load(open("churn_model.pkl", "rb"))

app = FastAPI()

# Define input structure
class CustomerData(BaseModel):
    CreditScore: float
    Gender: int
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Geo_France: int
    Geo_Germany: int
    Geo_Spain: int


@app.get("/")
def home():
    return {"message": "Bank Customer Churn Prediction API"}


@app.post("/predict")
def predict(data: CustomerData):

    input_data = np.array([[
        data.CreditScore,
        data.Gender,
        data.Age,
        data.Tenure,
        data.Balance,
        data.NumOfProducts,
        data.HasCrCard,
        data.IsActiveMember,
        data.EstimatedSalary,
        data.Geo_France,
        data.Geo_Germany,
        data.Geo_Spain
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        result = "Customer will churn"
    else:
        result = "Customer will stay"

    return {"prediction": result}