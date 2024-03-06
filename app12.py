# Bring in lightweight dependencies
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import joblib 
from pathlib import Path

app = FastAPI()

class ScoringItem(BaseModel): 
    # YearsAtCompany: float #/ 1, // Float value 
    # EmployeeSatisfaction: float #0.01, // Float value 
    # Position:str # "Non-Manager", # Manager or Non-Manager
    # Salary: int #4.0 // Ordinal 1,2,3,4,5

    #####

    Transaction_ID: object
    Customer_ID: object
    Date: object
    Total_Items: int
    Unique_Items: int
    Total_Sales: float
    Discounted_Sales: float
    Browsing_Duration_minutes: float
    Number_of_Clicks: int
    Age: int
    Gender: object
    Region: object
    Marital_Status: object
    Education: object
    Household_Income: float
    Loyalty_Card: int
    Loyalty_Points: float

# Load preprocessor and model from disk
preprocessor= joblib.load(Path('artifacts\data_transformation\preprocessor.joblib'))
model = joblib.load(Path('artifacts/model_trainer/model.joblib'))




@app.post('/')
async def scoring_endpoint(item:ScoringItem): 
    columns =['Transaction ID', 'Customer ID', 'Date', 'Total Items', 'Unique Items',
       'Total Sales', 'Discounted Sales', 'Browsing Duration (minutes)',
       'Number of Clicks', 'Age', 'Gender', 'Region',
       'Marital Status', 'Education', 'Household Income', 'Loyalty Card',
       'Loyalty Points'] 
    data = pd.DataFrame([item.dict().values()], columns=columns)

    # Feature engineering steps
    data['Discount Percentage'] = ((data['Total Sales'] - data['Discounted Sales']) / data['Total Sales']) * 100
    data['Unique Items per Total Item'] = data['Unique Items'] / data['Total Items']
    data['Month'] = pd.to_datetime(data['Date']).dt.month

    # Drop unnecessary columns
    data.drop(columns=['Customer ID', 'Transaction ID','Date'], inplace=True)

    # Convert data types
    data['Month'] = data['Month'].astype(str)
    data['Loyalty Card'] = data['Loyalty Card'].astype(str)
    # Transform data using preprocessor
    data = preprocessor.transform(data)
    # Make predictions using the model
    yhat = model.predict(data)

    # yhat = model.predict(df)
    return {"prediction":int(yhat)}



#uvicorn app:app --reload
# https://www.youtube.com/watch?v=C82lT9cWQiA&t=420s

#json
# {
#     "Transaction_ID": "123",
#     "Customer_ID": "123",
#     "Date": "2020-03-22",
#     "Total_Items": 12,
#     "Unique_Items": 3,
#     "Total_Sales": 123.0,
#     "Discounted_Sales": 12.0,
#     "Browsing_Duration_minutes": 12.0,
#     "Number_of_Clicks": 12,
#     "Age": 12,
#     "Gender": "Male",
#     "Region": "Rural",
#     "Marital_Status": "Married",
#     "Education": "Graduate",
#     "Household_Income": 123.0,
#     "Loyalty_Card": 1,
#     "Loyalty_Points": 123.0
# }