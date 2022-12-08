import numpy as np
import pandas as pd
import pickle
from tkinter import Y
from pandas import DataFrame
from flask import Flask, request, jsonify, render_template
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import spearmanr
import json
import plotly
import plotly.express as px
import random
import math

dataset = pd.read_csv("model/Institute-of-Computer-Studies-Graduate-Tracer-Study-2021-2022-Responses(ALTERED).csv")

ordinal_encoder = OrdinalEncoder()

IT_FEATURES = [
     'Sex',
     'Shiftee',
     'ComProg_1_1st',
     'ComProg_1_2nd',
     'ComProg_2_1st',
     'ComProg_2_2nd',
     'Intro_to_Computing_1st',
     'Intro_to_Computing_2nd',
     'Info_Management_1st',
     'Info_Management_2nd',
     'Operating_System_1st',
     'Operating_System_2nd',
     'Elective_1_1st',
     'Elective_1_2nd',
     'Elective_2_1st',
     'Elective_2_2nd',
     'Elective_3_1st',
     'Elective_3_2nd',
     'Data_Structures_1st',
     'Data_Structures_2nd',
     'Application_Dev_and_Emerging_Tech_1st',
     'Application_Dev_and_Emerging_Tech_2nd',
     'Human_and_Computer_Integration_1st',
     'Human_and_Computer_Integration_2nd',
     'Practicum_Industry_Immersion_1st',
     'Practicum_Industry_Immersion_2nd',
     'Integrative_Programming_and_Tech_1st',
     'Integrative_Programming_and_Tech_2nd',
     'System_Integration_and_Architecture_1st',
     'System_Integration_and_Architecture_2nd',
     'Information_Assurance_and_Security_1_1st',
     'Information_Assurance_and_Security_1_2nd',
     'Information_Assurance_and_Security_2_1st',
     'Information_Assurance_and_Security_2_2nd',
     'Software_Engineering_1st',
     'Software_Engineering_2nd',
     'Networking_1_1st',
     'Networking_1_2nd',
     'Networking_2_1st',
     'Networking_2_2nd',
     'WebProg_1st',
     'WebProg_2nd'
]
CS_FEATURES = [
     'Sex',
     'Shiftee',
     'ComProg_1_1st',
     'ComProg_1_2nd',
     'ComProg_2_1st',
     'ComProg_2_2nd',
     'Intro_to_Computing_1st',
     'Intro_to_Computing_2nd',
     'Info_Management_1st',
     'Info_Management_2nd',
     'Operating_System_1st',
     'Operating_System_2nd',
     'Elective_1_1st',
     'Elective_1_2nd',
     'Elective_2_1st',
     'Elective_2_2nd',
     'Elective_3_1st',
     'Elective_3_2nd',
     'Data_Structures_1st',
     'Data_Structures_2nd',
     'Application_Dev_and_Emerging_Tech_1st',
     'Application_Dev_and_Emerging_Tech_2nd',
     'Human_and_Computer_Integration_1st',
     'Human_and_Computer_Integration_2nd',
     'Practicum_Industry_Immersion_1st',
     'Practicum_Industry_Immersion_2nd',
     'Digital_Design_1st',
     'Digital_Design_2nd',
     'Architecture_and_Organization_1st',
     'Architecture_and_Organization_2nd',
     'Programming_Languages_1st',
     'Programming_Languages_2nd',
     'Modelling_and_Simulation_1st',
     'Modelling_and_Simulation_2nd',
     'Information_Assurance_and_Security_1st',
     'Information_Assurance_and_Security_1_2nd',
     'Software_Engineering_1_1st',
     'Software_Engineering_1_2nd',
     'Software_Engineering_2_1st',
     'Software_Engineering_2_2nd',
     'Network_Management_1st',
     'Network_Management_2nd',
     'Advance_Database_1st',
     'Advance_Database_2nd',
     'WebProg_1st',
     'WebProg_2nd'
]

TARGET = 'Suggested_job_role'

Cat_Y = dataset[TARGET]
X_IT = dataset[IT_FEATURES]
X_CS = dataset[CS_FEATURES]

X_IT = X_IT.replace(np.nan, 0)
X_CS = X_CS.replace(np.nan, 0)

X_IT['Sex'] = ordinal_encoder.fit_transform(X_IT[['Sex']])
X_CS['Sex'] = ordinal_encoder.fit_transform(X_CS[['Sex']])
percent = "%"
#   CREATE FLASK APP
flask_app = Flask(__name__)
#   MAIN JOB ROLE MODEL FOR BOTH IT CS
model_IT = pickle.load(open("model/ProjectModel_IT.pkl", "rb"))
model_CS = pickle.load(open("model/ProjectModel_CS.pkl", "rb"))
#   TOP 5 COURSES SUGGESTION FOR BOTH IT CS
model_ITsuggest = pickle.load(open("model/IT_SUGGESTEDcourse.pkl", "rb"))
model_CSsuggest = pickle.load(open("model/CS_SUGGESTEDcourse.pkl", "rb"))


@flask_app.route("/")

def Home():
    # Graph One
    # df = px.data.medals_wide()
    # dfTest = pd.read_csv(dfCon)
    # fig1 = px.bar(df, x="nation", y=["gold", "silver", "bronze"], title="Wide-Form Input")
    # fig1 = px.histogram(dfTest, y="humidity", title="Wide-Form Input")
    # graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    dfCon = "model/Institute-of-Computer-Studies-Graduate-Tracer-Study-2021-2022-Responses(ALTERED).csv"
    df = pd.read_csv(dfCon)
    
    # Graph three
    fig3 = px.histogram(df, y="Degree_Completed", title="Degree Completed Population of 2018 - 2022")
    graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Graph one
    fig1 = px.bar(df, x="Suggested_job_role", y=["Shiftee", "Units"], title="Shiftee and Units Frequency for Job Role")
    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Graph four
    fig4 = px.bar(df, x="Sex", y=["Shiftee", "Units"], title="Shiftee and Units Frequency for Sex")
    graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Graph five
    fig5 = px.bar(df, x="Suggested_job_role", y=['Digital_Design_1st',
     'Architecture_and_Organization_1st',
     'Programming_Languages_1st',
     'Modelling_and_Simulation_1st',
     'Information_Assurance_and_Security_1st',
     'Software_Engineering_1_1st',
     'Software_Engineering_2_1st',
     'Network_Management_1st',
     'Advance_Database_1st',
     'WebProg_1st'], title="Exclusive Computer Science Courses Frequency for Job Role")
    graph5JSON = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Graph six
    fig6 = px.bar(df, x="Suggested_job_role", y=['Integrative_Programming_and_Tech_1st',
     'System_Integration_and_Architecture_1st',
     'Information_Assurance_and_Security_1_1st',
     'Information_Assurance_and_Security_2_1st',
     'Software_Engineering_1st',
     'Networking_1_1st',
     'Networking_2_1st',
     'WebProg_1st'], title="Exclusive Information Technology Courses Frequency for Job Role")
    graph6JSON = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Graph two
    dataIRIS = px.data.iris()
    fig2 = px.scatter_3d(df, x='Practicum_Industry_Immersion_1st', y='Data_Structures_1st', z='Operating_System_1st',
              color='Suggested_job_role',  title="Scatter Plot")
    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template("index.html", graph1JSON=graph1JSON, graph2JSON=graph2JSON, graph3JSON=graph3JSON, graph4JSON=graph4JSON, graph5JSON=graph5JSON, graph6JSON=graph6JSON) # 

@flask_app.route("/ITend")
def ITend_view():
    return render_template("ITend.html")

@flask_app.route("/CSend")
def CSend_view():
    return render_template("CSend.html")

@flask_app.route("/predictCS")
def CS_view():
    return render_template("predictCS.html")

@flask_app.route("/predictIT")
def IT_view():
    return render_template("predictIT.html")

@flask_app.route("/predict_IT", methods = ["POST"])

def predict_IT():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    new_Xdata_IT = X_IT.sample(4)
    new_Ydata_IT = Cat_Y[new_Xdata_IT.index.values]
    pred_IT = model_IT.predict(features)
    suggestIT = model_ITsuggest.predict(pred_IT)

    def recall(new_Ydata_IT, pred_IT, K):
            act_set = set(new_Ydata_IT)
            pred_set = set(pred_IT[:K])
            result = round(len(act_set & pred_set) / float(len(act_set)), 2)
            return result
    
    for K in range(0, 3):
        predIT = pred_IT[0]
        actual = new_Ydata_IT
        prediction = ["Software Engineer / Programmer", "Technical Support Specialist", "Academician", "Administrative Assistant"]
        random.shuffle(prediction)
        prediction = [prediction.replace(predIT, '0')for prediction in prediction]
        prediction.append(predIT)
        
        fetch1 = recall(actual, prediction, K)
        if fetch1 == 1.0:
            fetch1 = 100
        elif fetch1 == 0.75:
            fetch1 = 75
        elif fetch1 == 0.67:
            fetch1 = 67 
        elif fetch1 == 0.5:
            fetch1 = 50
        elif fetch1 == 0.33:
            fetch1 = 33
        elif fetch1 == 0.25:
            fetch1 = 25
        else:
            fetch1 = 0
            
        fetch2 = recall(actual, prediction, K-1)
        if fetch2 == 1.0:
            fetch2 = 100
        elif fetch2 == 0.75:
            fetch2 = 75
        elif fetch2 == 0.67:
            fetch2 = 67
        elif fetch2 == 0.5:
            fetch2 = 50
        elif fetch2 == 0.33:
            fetch2 = 33
        elif fetch2 == 0.25:
            fetch2 = 25
        else:
            fetch2 = 0
            
        fetch3 = recall(actual, prediction, K-2)
        if fetch3 == 1.0:
            fetch3 = 100
        elif fetch3 == 0.75:
            fetch3 = 75
        elif fetch3 == 0.67:
            fetch3 = 67
        elif fetch3 == 0.5:
            fetch3 = 50
        elif fetch3 == 0.33:
            fetch3 = 33
        elif fetch3 == 0.25:
            fetch3 = 25
        else:
            fetch3 = 0
            
        fetch4 = recall(actual, prediction, K-3)
        if fetch4 == 1.0:
            fetch4 = 100
        elif fetch4 == 0.75:
            fetch4 = 75
        elif fetch4 == 0.67:
            fetch4 = 67
        elif fetch4 == 0.5:
            fetch4 = 50
        elif fetch4 == 0.33:
            fetch4 = 33
        elif fetch4 == 0.25:
            fetch4 = 25
        else:
            fetch4 = 0
            
        fetchPred1 = prediction[0]
        fetchPred2 = prediction[-1]
        fetchPred3 = prediction[-2]
        fetchPred4 = prediction[-3]
        
        # job_label1 = "{}".format(f"{prediction[K-1]} = {fetch2}%") if fetch2 == 100 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetch2 == 75 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetch2 == 67 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetch2 == 50 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetch2 == 33 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetch2 == 25 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" else "",
        # prediction_label4 = "Additionally, there is a good likelihood that the job stated on the left side will lead to your first career." if fetch2 == fetch1 and fetchPred1 == '0' or fetch2 == fetch3 and fetchPred3 == '0' or fetch2 == fetch4 and fetchPred4 == '0' else "",
        # prediction_label5 = "Additionally, there is a good likelihood that the job stated on the left side will lead to your first career." if fetch1 > 0.0 and fetch2 != fetch1 or fetch3 > 0.0 and fetch2 != fetch3 or fetch4 > 0.0 and fetch2 != fetch4 else "",
        # fetchPred1 == '0' and fetchPred2 == fetchPred2 and fetchPred3 == fetchPred3 and fetchPred4 == fetchPred4 or fetchPred2 == '0' and fetchPred3 == fetchPred3 and fetchPred4 == fetchPred4 and fetchPred1 == fetchPred1 or fetchPred3 == '0' and fetchPred4 == fetchPred4 and fetchPred1 == fetchPred1 and fetchPred2 == fetchPred2 or fetchPred4 == '0' and fetchPred1 == fetchPred1 and fetchPred2 == fetchPred2 and fetchPred3 == fetchPred3   
        return render_template("ITend.html", 
                               prediction_text1 = "" if fetch1 == 0 or fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetchPred1 == '0' else "{}".format(f"{prediction[K]} : {fetch1}%"), 
                               prediction_text2 = "" if fetch2 == 0 or fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetchPred2 == '0' else "{}".format(f"{prediction[K-1]} : {fetch2}%"),
                               prediction_text3 = "" if fetch3 == 0 or fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetchPred3 == '0' else "{}".format(f"{prediction[K-2]} : {fetch3}%"),
                               prediction_text4 = "" if fetch4 == 0 or fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetchPred4 == '0' else "{}".format(f"{prediction[K-3]} : {fetch4}%"),
                               prediction_label1 = "" if fetch2 == 0 or fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 == 1.0 and fetchPred2 == "Administrative Assistant" else "{}".format(f"{prediction[K-1]} is a more likely career path for you. Congratulations!"),
                            #    prediction_label2 = "- Snap, there is still a good probability that you will find your first job in an industry related to IT or computer science." if fetchPred2 == "Administrative Assistant" and fetch1 >= 0 and fetchPred1 == "0" or fetchPred2 == "Administrative Assistant" and fetch3 >= 0 and fetchPred3 == "0" or fetchPred2 == "Administrative Assistant" and fetch4 >= 0 and fetchPred4 == "0" else "",
                               prediction_label3 = "{}".format(f"Nothing follows...") if fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 else "",
                               prediction_label4 = "- In addition to the possibilities mentioned above, there is still a significant chance that you will be hired for your first job in one or more of the positions listed on the left side." if fetch1 > 0 and fetchPred1 !="0" or fetch3 > 0 and fetchPred3 !="0" or fetch4 > 0 and fetchPred4 !="0" or fetchPred2 == "Administrative Assistant" else "",
                               job_label1 = "{}".format(f"{prediction[K-1]} = {fetch2}%") if fetch2 <= 100 and fetch1 >= 0 and fetch3 >= 0 and fetch4 >= 0 and fetchPred2 == "Administrative Assistant" else "",
                               job_label2 = "Predicted IT/CS Related Job(s)" if fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch1 == 0 and fetch2 <= 100 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" else "Predicted IT/CS Related Job(s)",
                            #    job_label3 = "-- NOT APPLICABLE --" if fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch1 == 0 and fetch2 <= 100 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" else "Predicted IT/CS Related Job(s)",
                            #    label_text1 = "To increase the likelihood of landing IT/CS-related jobs, the below courses on the left side must be improved.~~" if fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 else "",
                               label_text2 = "- To increase the likelihood of landing IT/CS-related jobs, the below courses on the left side must be improved." if fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 >= 0 and fetch3 >= 0 and fetch4 >= 0 and fetchPred2 == "Administrative Assistant" or fetch2 <= 100 and fetch1 >= 0 and fetch3 >= 0 and fetch4 >= 0 and fetchPred2 == '0' or fetch1 == 0 and fetch2 <= 100 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" else "",
                            #    course_suggestion1 = "{}".format(suggestCS.tolist()) if fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 else "",
                               course_suggestion2 = "{}".format(suggestIT.tolist()) if fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 >= 0 and fetch3 >= 0 and fetch4 >= 0 and fetchPred2 == "Administrative Assistant" or fetch2 <= 100 and fetch1 >= 0 and fetch3 >= 0 and fetch4 >= 0 and fetchPred2 == '0' or fetch1 == 0 and fetch2 <= 100 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" else ""
                               )

@flask_app.route("/predict_CS", methods = ["POST"])
def predict_CS():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    new_Xdata_CS = X_CS.sample(4)
    new_Ydata_CS = Cat_Y[new_Xdata_CS.index.values]
    pred_CS = model_CS.predict(features)
    suggestCS = model_CSsuggest.predict(pred_CS)

    def recall(new_Ydata_CS, pred_CS, K):
            act_set = set(new_Ydata_CS)
            pred_set = set(pred_CS[:K])
            result = round(len(act_set & pred_set) / float(len(act_set)), 2)
            return result
        
    for K in range(0, 3):
        
        predCS = pred_CS[0]
        actual = new_Ydata_CS
        prediction = ["Software Engineer / Programmer", "Technical Support Specialist", "Academician", "Administrative Assistant"]
        random.shuffle(prediction)
        prediction = [prediction.replace(predCS, '0')for prediction in prediction]
        prediction.append(predCS)
        
        fetch1 = recall(actual, prediction, K)
        if fetch1 == 1.0:
            fetch1 = 100
        elif fetch1 == 0.75:
            fetch1 = 75
        elif fetch1 == 0.67:
            fetch1 = 67 
        elif fetch1 == 0.5:
            fetch1 = 50
        elif fetch1 == 0.33:
            fetch1 = 33
        elif fetch1 == 0.25:
            fetch1 = 25
        else:
            fetch1 = 0
            
        fetch2 = recall(actual, prediction, K-1)
        if fetch2 == 1.0:
            fetch2 = 100
        elif fetch2 == 0.75:
            fetch2 = 75
        elif fetch2 == 0.67:
            fetch2 = 67
        elif fetch2 == 0.5:
            fetch2 = 50
        elif fetch2 == 0.33:
            fetch2 = 33
        elif fetch2 == 0.25:
            fetch2 = 25
        else:
            fetch2 = 0
            
        fetch3 = recall(actual, prediction, K-2)
        if fetch3 == 1.0:
            fetch3 = 100
        elif fetch3 == 0.75:
            fetch3 = 75
        elif fetch3 == 0.67:
            fetch3 = 67
        elif fetch3 == 0.5:
            fetch3 = 50
        elif fetch3 == 0.33:
            fetch3 = 33
        elif fetch3 == 0.25:
            fetch3 = 25
        else:
            fetch3 = 0
            
        fetch4 = recall(actual, prediction, K-3)
        if fetch4 == 1.0:
            fetch4 = 100
        elif fetch4 == 0.75:
            fetch4 = 75
        elif fetch4 == 0.67:
            fetch4 = 67
        elif fetch4 == 0.5:
            fetch4 = 50
        elif fetch4 == 0.33:
            fetch4 = 33
        elif fetch4 == 0.25:
            fetch4 = 25
        else:
            fetch4 = 0
            
        fetchPred1 = prediction[0]
        fetchPred2 = prediction[-1]
        fetchPred3 = prediction[-2]
        fetchPred4 = prediction[-3]
        
        # job_label1 = "{}".format(f"{prediction[K-1]} = {fetch2}%") if fetch2 == 100 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetch2 == 75 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetch2 == 67 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetch2 == 50 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetch2 == 33 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetch2 == 25 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" else "",
        # prediction_label4 = "Additionally, there is a good likelihood that the job stated on the left side will lead to your first career." if fetch2 == fetch1 and fetchPred1 == '0' or fetch2 == fetch3 and fetchPred3 == '0' or fetch2 == fetch4 and fetchPred4 == '0' else "",
        # prediction_label5 = "Additionally, there is a good likelihood that the job stated on the left side will lead to your first career." if fetch1 > 0.0 and fetch2 != fetch1 or fetch3 > 0.0 and fetch2 != fetch3 or fetch4 > 0.0 and fetch2 != fetch4 else "",
        # fetchPred1 == '0' and fetchPred2 == fetchPred2 and fetchPred3 == fetchPred3 and fetchPred4 == fetchPred4 or fetchPred2 == '0' and fetchPred3 == fetchPred3 and fetchPred4 == fetchPred4 and fetchPred1 == fetchPred1 or fetchPred3 == '0' and fetchPred4 == fetchPred4 and fetchPred1 == fetchPred1 and fetchPred2 == fetchPred2 or fetchPred4 == '0' and fetchPred1 == fetchPred1 and fetchPred2 == fetchPred2 and fetchPred3 == fetchPred3   
        return render_template("CSend.html", 
                               prediction_text1 = "" if fetch1 == 0 or fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetchPred1 == '0' else "{}".format(f"{prediction[K]} : {fetch1}%"), 
                               prediction_text2 = "" if fetch2 == 0 or fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetchPred2 == '0' else "{}".format(f"{prediction[K-1]} : {fetch2}%"),
                               prediction_text3 = "" if fetch3 == 0 or fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetchPred3 == '0' else "{}".format(f"{prediction[K-2]} : {fetch3}%"),
                               prediction_text4 = "" if fetch4 == 0 or fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetchPred4 == '0' else "{}".format(f"{prediction[K-3]} : {fetch4}%"),
                               prediction_label1 = "" if fetch2 == 0 or fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 == 1.0 and fetchPred2 == "Administrative Assistant" else "{}".format(f"{prediction[K-1]} is a more likely career path for you. Congratulations!"),
                            #    prediction_label2 = "- Snap, there is still a good probability that you will find your first job in an industry related to IT or computer science." if fetchPred2 == "Administrative Assistant" and fetch1 >= 0 and fetchPred1 == "0" or fetchPred2 == "Administrative Assistant" and fetch3 >= 0 and fetchPred3 == "0" or fetchPred2 == "Administrative Assistant" and fetch4 >= 0 and fetchPred4 == "0" else "",
                               prediction_label3 = "{}".format(f"Nothing follows...") if fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 else "",
                               prediction_label4 = "- In addition to the possibilities mentioned above, there is still a significant chance that you will be hired for your first job in one or more of the positions listed on the left side." if fetch1 > 0 and fetchPred1 !="0" or fetch3 > 0 and fetchPred3 !="0" or fetch4 > 0 and fetchPred4 !="0" or fetchPred2 == "Administrative Assistant" else "",
                               job_label1 = "{}".format(f"{prediction[K-1]} = {fetch2}%") if fetch2 <= 100 and fetch1 >= 0 and fetch3 >= 0 and fetch4 >= 0 and fetchPred2 == "Administrative Assistant" else "",
                               job_label2 = "Predicted IT/CS Related Job(s)" if fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch1 == 0 and fetch2 <= 100 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" else "Predicted IT/CS Related Job(s)",
                            #    job_label3 = "-- NOT APPLICABLE --" if fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch1 == 0 and fetch2 <= 100 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" else "Predicted IT/CS Related Job(s)",
                            #    label_text1 = "To increase the likelihood of landing IT/CS-related jobs, the below courses on the left side must be improved.~~" if fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 else "",
                               label_text2 = "- To increase the likelihood of landing IT/CS-related jobs, the below courses on the left side must be improved." if fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 >= 0 and fetch3 >= 0 and fetch4 >= 0 and fetchPred2 == "Administrative Assistant" or fetch2 <= 100 and fetch1 >= 0 and fetch3 >= 0 and fetch4 >= 0 and fetchPred2 == '0' or fetch1 == 0 and fetch2 <= 100 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" else "",
                            #    course_suggestion1 = "{}".format(suggestCS.tolist()) if fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 else "",
                               course_suggestion2 = "{}".format(suggestCS.tolist()) if fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 >= 0 and fetch3 >= 0 and fetch4 >= 0 and fetchPred2 == "Administrative Assistant" or fetch2 <= 100 and fetch1 >= 0 and fetch3 >= 0 and fetch4 >= 0 and fetchPred2 == '0' or fetch1 == 0 and fetch2 <= 100 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" else ""
                               )
        
if __name__ == "__main__":
    flask_app.run(debug=False)
