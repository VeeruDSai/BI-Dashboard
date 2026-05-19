import os
import json
import requests
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# ==========================================
# ML MODEL SETUP
# ==========================================
data_path = "cleaned_placement_dataset.csv"
feature_cols = [
    'gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'degree_p', 'workex', 
    'etest_p', 'mba_p', 'hsc_s_Commerce', 'hsc_s_Science', 
    'degree_t_Others', 'degree_t_Sci&Tech', 'specialisation_Mkt&HR'
]

clf = None
reg = None
imputer = None

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    X = df[feature_cols]
    y_status = df['status']
    
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf.fit(X, y_status)
    
    placed_df = df[df['status'] == 1]
    X_placed = placed_df[feature_cols]
    y_salary = placed_df['salary']
    
    reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    reg.fit(X_placed, y_salary)
    
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X)

# ==========================================
# ENDPOINTS
# ==========================================
@app.route("/api/predict", methods=["POST"])
def predict():
    if clf is None:
        return jsonify({"error": "Model not trained. Dataset missing."}), 500
        
    data = request.json
    
    # Extract
    is_cgpa = data.get('is_cgpa', False)
    multiplier = 9.5 if is_cgpa else 1.0
    
    def get_val(key):
        val = data.get(key)
        if val == "" or val is None:
            return None
        return float(val)
        
    ssc_p_val = get_val('ssc_p')
    hsc_p_val = get_val('hsc_p')
    degree_p_val = get_val('degree_p')
    mba_p_val = get_val('mba_p')
    
    ssc_p = ssc_p_val * multiplier if ssc_p_val is not None else None
    hsc_p = hsc_p_val * multiplier if hsc_p_val is not None else None
    degree_p = degree_p_val * multiplier if degree_p_val is not None else None
    mba_p = mba_p_val * multiplier if mba_p_val is not None else None

    gender = data.get('gender', 'Not Specified')
    ssc_b = data.get('ssc_b', 'Not Specified')
    hsc_b = data.get('hsc_b', 'Not Specified')
    hsc_s = data.get('hsc_s', 'Not Specified')
    degree_t = data.get('degree_t', 'Not Specified')
    specialisation = data.get('specialisation', 'Not Specified')
    workex = data.get('workex', 'Not Specified')
    etest_p = get_val('etest_p')
    ug_exam = data.get('ug_exam', 'Not Specified')

    # Mapping
    input_data = {col: np.nan for col in feature_cols}
    if gender != "Not Specified": input_data['gender'] = 1.0 if gender == "Male" else 0.0
    if ssc_p is not None: input_data['ssc_p'] = float(ssc_p)
    if ssc_b != "Not Specified": input_data['ssc_b'] = 1.0 if ssc_b in ["CBSE", "ICSE"] else 0.0
    if hsc_p is not None: input_data['hsc_p'] = float(hsc_p)
    if hsc_b != "Not Specified": input_data['hsc_b'] = 1.0 if hsc_b in ["CBSE", "ISC"] else 0.0
    if degree_p is not None: input_data['degree_p'] = float(degree_p)
    if workex != "Not Specified": input_data['workex'] = 1.0 if workex == "Yes" else 0.0
    if etest_p is not None: input_data['etest_p'] = float(etest_p)
    if mba_p is not None: input_data['mba_p'] = float(mba_p)
    
    if hsc_s != "Not Specified":
        input_data['hsc_s_Commerce'] = 1.0 if hsc_s == "Commerce" else 0.0
        input_data['hsc_s_Science'] = 1.0 if "Science" in hsc_s else 0.0
        
    if degree_t != "Not Specified":
        if "B.E/B.Tech" in degree_t or "B.Sc" in degree_t:
            input_data['degree_t_Sci&Tech'] = 1.0
            input_data['degree_t_Others'] = 0.0
        elif degree_t == "Other":
            input_data['degree_t_Sci&Tech'] = 0.0
            input_data['degree_t_Others'] = 1.0
        else:
            input_data['degree_t_Sci&Tech'] = 0.0
            input_data['degree_t_Others'] = 0.0
            
    if specialisation != "Not Specified":
        input_data['specialisation_Mkt&HR'] = 1.0 if specialisation == "Mkt&HR" else 0.0

    input_df = pd.DataFrame([input_data])
    input_imputed = imputer.transform(input_df)

    bonus_prob = 0.0
    if ug_exam in ["KCET", "COMEDK"] and "CS/IT" in degree_t:
        bonus_prob = 0.08
        
    status_prob = float(min(1.0, clf.predict_proba(input_imputed)[0][1] + bonus_prob))
    
    if status_prob > 0.5:
        base_salary = reg.predict(input_imputed)[0]
        if ug_exam in ["COMEDK", "KCET"] and "CS/IT" in degree_t:
            base_salary *= 1.45
        elif "Karnataka" in ssc_b or "Karnataka" in hsc_b:
            base_salary *= 1.15
            
        return jsonify({
            "placed": True,
            "probability": status_prob,
            "salary_lower": int(base_salary * 0.88),
            "salary_upper": int(base_salary * 1.15)
        })
    else:
        return jsonify({
            "placed": False,
            "probability": status_prob
        })

@app.route("/api/jobs", methods=["GET"])
def get_jobs():
    try:
        res = requests.get('https://www.themuse.com/api/public/jobs?location=India&page=1', timeout=5)
        if res.status_code == 200:
            return jsonify(res.json().get('results', [])[:6])
    except:
        pass
    return jsonify([])

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    api_key = data.get('api_key', '')
    user_query = data.get('query', '')
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.0-pro')
            response = model.generate_content(f"You are an expert career counselor for Indian students, specializing in Karnataka colleges, KCET, and COMEDK. Answer this query: {user_query}")
            return jsonify({"response": response.text, "type": "gemini"})
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    else:
        q = user_query.lower()
        ans = "That's a great question! Securing a high rank in KCET/COMEDK and maintaining a CGPA > 8.0 unlocks the best opportunities in Bengaluru. Provide a Gemini API key for dynamic AI answers!"
        if "rvce" in q or "bmsce" in q or "msrit" in q or "top" in q:
            ans = "Top colleges in Bengaluru like RVCE, BMSCE, and MSRIT typically require a KCET rank under 2000 or a COMEDK rank under 1500 for CS. Average packages range from 15-20 LPA."
        elif "kcet" in q and "comedk" in q:
            ans = "KCET is for Karnataka domicile students (lower fees), while COMEDK is open to all Indian students (Pan-India, higher fees) offering access to the same private colleges."
        elif "fee" in q or "cost" in q:
            ans = "Through KCET, fees range from ₹80K to ₹1 Lakh/year. Through COMEDK, fees range from ₹2.5 to ₹3 Lakhs/year."
            
        return jsonify({"response": ans, "type": "local"})

if __name__ == "__main__":
    app.run(port=8000, debug=True)
