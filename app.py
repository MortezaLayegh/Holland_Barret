from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


# @app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
# def index():
#     if request.method == 'POST':
#         try:
#             #  reading the inputs given by the user
#             fixed_acidity =float(request.form['fixed_acidity'])
#             volatile_acidity =float(request.form['volatile_acidity'])
#             citric_acid =float(request.form['citric_acid'])
#             residual_sugar =float(request.form['residual_sugar'])
#             chlorides =float(request.form['chlorides'])
#             free_sulfur_dioxide =float(request.form['free_sulfur_dioxide'])
#             total_sulfur_dioxide =float(request.form['total_sulfur_dioxide'])
#             density =float(request.form['density'])
#             pH =float(request.form['pH'])
#             sulphates =float(request.form['sulphates'])
#             alcohol =float(request.form['alcohol'])
       
         
#             data = [fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol]
#             data = np.array(data).reshape(1, 11)
            
#             obj = PredictionPipeline()
#             predict = obj.predict(data)

#             return render_template('results.html', prediction = str(predict))

#         except Exception as e:
#             print('The Exception message is: ',e)
#             return 'something is wrong'

#     else:
#         return render_template('index.html')
    



#######################
    
@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            # reading the inputs given by the user
            total_items = float(request.form['total_items'])
            unique_items = float(request.form['unique_items'])
            total_sales = float(request.form['total_sales'])
            discounted_sales = float(request.form['discounted_sales'])
            browsing_duration = float(request.form['browsing_duration'])
            number_of_clicks = float(request.form['number_of_clicks'])
            age = float(request.form['age'])
            household_income = float(request.form['household_income'])
            loyalty_points = float(request.form['loyalty_points'])
            discount_percentage = float(request.form['discount_percentage'])
            unique_items_per_total_item = float(request.form['unique_items_per_total_item'])
            month = request.form['month']
            
            gender = request.form['gender']
            region = request.form['region']
            marital_status = request.form['marital_status']
            education = request.form['education']
            loyalty_card = request.form['loyalty_card']

    
            data = [total_items, unique_items, total_sales, discounted_sales, browsing_duration, number_of_clicks, age,gender,
                    region, marital_status, education, household_income, loyalty_card, loyalty_points, discount_percentage, unique_items_per_total_item, month]
            data = np.array(data).reshape(1, 11)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port = 8080, debug=True)
    app.run(host="0.0.0.0", port = 8080)



#############################
    
from fastapi import FastAPI, File, UploadFile



app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
def index():
	return RedirectResponse(url="/docs")





@app.get("/train")
def training():
    os.system("python main.py")
    return "Training Successful!" 
