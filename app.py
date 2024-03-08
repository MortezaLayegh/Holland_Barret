from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline
import pandas as pd
import joblib 
from pathlib import Path
#test

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 

@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            Transaction_ID = request.form['Transaction_ID']
            Customer_ID = request.form['Customer_ID']
            Date = request.form['Date']
            Total_Items = int(request.form['Total_Items'])
            Unique_Items = int(request.form['Unique_Items'])
            Total_Sales = float(request.form['Total_Sales'])
            Discounted_Sales = float(request.form['Discounted_Sales'])
            Browsing_Duration_minutes = float(request.form['Browsing_Duration_minutes'])
            Number_of_Clicks = int(request.form['Number_of_Clicks'])
            Age = int(request.form['Age'])
            Gender = request.form['Gender']
            Region = request.form['Region']
            Marital_Status = request.form['Marital_Status']
            Education = request.form['Education']
            Household_Income = float(request.form['Household_Income'])
            Loyalty_Card = int(request.form['Loyalty_Card'])
            Loyalty_Points = float(request.form['Loyalty_Points'])

            columns =['Transaction ID', 'Customer ID', 'Date', 'Total Items', 'Unique Items',
            'Total Sales', 'Discounted Sales', 'Browsing Duration (minutes)',
            'Number of Clicks', 'Age', 'Gender', 'Region',
            'Marital Status', 'Education', 'Household Income', 'Loyalty Card',
            'Loyalty Points'] 

            data = pd.DataFrame([[Transaction_ID, Customer_ID, Date, Total_Items, Unique_Items, Total_Sales, Discounted_Sales, Browsing_Duration_minutes,
                    Number_of_Clicks, Age, Gender, Region, Marital_Status, Education, Household_Income, Loyalty_Card, Loyalty_Points]], columns=columns)

            preprocessor = joblib.load(Path('artifacts/data_transformation/preprocessor.joblib'))
            model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

            # Feature engineering steps
            data['Discount Percentage'] = ((data['Total Sales'] - data['Discounted Sales']) / data['Total Sales']) * 100
            data['Unique Items per Total Item'] = data['Unique Items'] / data['Total Items']
            data['Month'] = pd.to_datetime(data['Date']).dt.month

            # Drop unnecessary columns
            data.drop(columns=['Customer ID', 'Transaction ID', 'Date'], inplace=True)

            # Convert data types
            data['Month'] = data['Month'].astype(str)
            data['Loyalty Card'] = data['Loyalty Card'].astype(str)

            # Transform data using preprocessor
            data = preprocessor.transform(data)

            # Make predictions using the model
            predict = model.predict(data)
            
            return render_template('results.html', prediction=str(predict))

        except Exception as e:
            print('The Exception message is: ', e)
            return 'Something went wrong!'

    else:
        return render_template('index.html')

if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=8080, debug=True)
    app.run(host="0.0.0.0", port=8080)



