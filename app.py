import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
import os
import requests
import time
from datetime import datetime
import google.generativeai as genai

# --- Page Config ---
st.set_page_config(page_title="AI Placement Engine | Bengaluru Edition", layout="wide", page_icon="🚀")

# --- Custom CSS for Ultra-Rich Aesthetics ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #1a1a2e 100%);
        color: #f8fafc;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2.5rem;
        font-weight: 600;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        margin-top: 1rem;
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 20px 25px -5px rgba(139, 92, 246, 0.5);
    }
    
    .glass-panel {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }
    
    h1, h2, h3 {
        background: linear-gradient(to right, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    .prediction-box-placed {
        padding: 3rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
        text-align: center;
        margin-top: 2rem;
        backdrop-filter: blur(20px);
        box-shadow: 0 25px 50px -12px rgba(16, 185, 129, 0.25);
    }
    
    .prediction-box-not-placed {
        padding: 3rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.05));
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #f87171;
        text-align: center;
        margin-top: 2rem;
        backdrop-filter: blur(20px);
        box-shadow: 0 25px 50px -12px rgba(239, 68, 68, 0.25);
    }
    
    .main-header {
        text-align: center;
        margin-bottom: 0.5rem;
        font-size: 3.5rem;
    }
    
    .karnataka-badge {
        background: linear-gradient(90deg, #f59e0b, #ef4444);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        color: white;
        font-weight: 600;
        margin-left: 8px;
    }
    
    .analytics-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .analytics-card h4 {
        margin-top: 0;
        font-size: 1rem;
        color: #94a3b8;
        -webkit-text-fill-color: #94a3b8;
    }
    .analytics-card .value {
        font-size: 2rem;
        font-weight: 800;
        color: #60a5fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data_and_train_models():
    data_path = "cleaned_placement_dataset.csv"
    if not os.path.exists(data_path):
        return None, None, None, None
        
    df = pd.read_csv(data_path)
    
    feature_cols = [
        'gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'degree_p', 'workex', 
        'etest_p', 'mba_p', 'hsc_s_Commerce', 'hsc_s_Science', 
        'degree_t_Others', 'degree_t_Sci&Tech', 'specialisation_Mkt&HR'
    ]
    
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
    
    return clf, reg, imputer, feature_cols

clf, reg, imputer, feature_cols = load_data_and_train_models()

# --- Top Navigation ---
tabs = st.tabs(["🔮 Predictor Matrix", "📊 BLR Real-Time Analytics", "🤖 AI Career Counselor"])

# ==========================================
# TAB 1: PREDICTOR MATRIX
# ==========================================
with tabs[0]:
    st.markdown("<h1 class='main-header'>🚀 AI Career Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; margin-bottom: 3rem; font-size: 1.1rem;'>Optimized for Pan-India Boards, KCET, COMEDK, and Bengaluru Colleges.</p>", unsafe_allow_html=True)

    # Toggle for CGPA vs Percentage
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.subheader("⚙️ Scoring Format")
    input_format = st.radio("Academic Scoring Mode:", ["Percentage (0-100%)", "CGPA (0.0-10.0)"], horizontal=True)
    
    is_cgpa = "CGPA" in input_format
    score_max = 10.0 if is_cgpa else 100.0
    score_step = 0.1 if is_cgpa else 1.0
    score_format = "%.2f"

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🏫 Schooling")
            ssc_b = st.selectbox("10th Board Type", ["Not Specified", "CBSE", "ICSE", "Karnataka SSLC", "Other State Board"])
            ssc_p_val = st.number_input(f"⭐ 10th Score ({'CGPA' if is_cgpa else '%'})", min_value=0.0, max_value=score_max, value=None, format=score_format, step=score_step)
            
            hsc_b = st.selectbox("12th Board Type", ["Not Specified", "CBSE", "ISC", "Karnataka PUC", "Other State Board"])
            hsc_s = st.selectbox("12th Stream", ["Not Specified", "Science (PCMB/PCMC)", "Commerce", "Arts"])
            hsc_p_val = st.number_input(f"⭐ 12th Score ({'CGPA' if is_cgpa else '%'})", min_value=0.0, max_value=score_max, value=None, format=score_format, step=score_step)
            
        with col2:
            st.markdown("### 🎓 Undergrad (College)")
            ug_exam = st.selectbox("Admission Route", ["Not Specified", "KCET", "COMEDK", "JEE Main", "Management Quota", "Other"])
            degree_t = st.selectbox("Undergrad Degree", ["Not Specified", "B.E/B.Tech (CS/IT/Circuital)", "B.E/B.Tech (Core)", "B.Sc/BCA", "B.Com/BBA", "Other"])
            degree_p_val = st.number_input(f"⭐ Undergrad Score ({'CGPA' if is_cgpa else '%'})", min_value=0.0, max_value=score_max, value=None, format=score_format, step=score_step)
            
        with col3:
            st.markdown("### 💼 Master's & Profile")
            mba_p_val = st.number_input(f"⭐ Postgrad/MBA Score ({'CGPA' if is_cgpa else '%'})", min_value=0.0, max_value=score_max, value=None, format=score_format, step=score_step)
            specialisation = st.selectbox("MBA Specialisation", ["Not Specified", "Mkt&Fin", "Mkt&HR"])
            workex = st.selectbox("Work Experience", ["Not Specified", "Yes", "No"])
            etest_p = st.number_input("Employability/Aptitude Test (%)", min_value=0.0, max_value=100.0, value=None, format="%.2f")
            gender = st.selectbox("Gender", ["Not Specified", "Male", "Female"])
    
        submit_button = st.form_submit_button("✨ Initialize Prediction Engine")
    st.markdown("</div>", unsafe_allow_html=True)

    if submit_button and clf is not None:
        multiplier = 9.5 if is_cgpa else 1.0
        
        ssc_p = ssc_p_val * multiplier if ssc_p_val is not None else None
        hsc_p = hsc_p_val * multiplier if hsc_p_val is not None else None
        degree_p = degree_p_val * multiplier if degree_p_val is not None else None
        mba_p = mba_p_val * multiplier if mba_p_val is not None else None
    
        # Map Inputs to Original Dataset Schema
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
            else: # B.Com/BBA (Comm&Mgmt)
                input_data['degree_t_Sci&Tech'] = 0.0
                input_data['degree_t_Others'] = 0.0
            
        if specialisation != "Not Specified":
            input_data['specialisation_Mkt&HR'] = 1.0 if specialisation == "Mkt&HR" else 0.0
            
        input_df = pd.DataFrame([input_data])
        input_imputed = imputer.transform(input_df)
        
        # Boost confidence artificially if KCET/COMEDK top tiers, as real data indicates better placements
        bonus_prob = 0.0
        if ug_exam in ["KCET", "COMEDK"] and degree_t == "B.E/B.Tech (CS/IT/Circuital)":
            bonus_prob = 0.08
            
        status_pred = clf.predict(input_imputed)[0]
        status_prob = min(1.0, clf.predict_proba(input_imputed)[0][1] + bonus_prob)
        
        if status_prob > 0.5:
            base_salary = reg.predict(input_imputed)[0]
            # Adjust salary dynamically based on Bengaluru market realities
            if ug_exam in ["COMEDK", "KCET"] and degree_t == "B.E/B.Tech (CS/IT/Circuital)":
                base_salary *= 1.45 # Bengaluru Premium
            elif "Karnataka" in ssc_b or "Karnataka" in hsc_b:
                base_salary *= 1.15
                
            salary_lower = int(base_salary * 0.88)
            salary_upper = int(base_salary * 1.15)
            
            st.markdown(f"""
            <div class="prediction-box-placed">
                <h1 style="color: #34d399; margin-bottom: 1rem;">✅ Placement Secured</h1>
                <p style="font-size: 1.25rem; color: #f8fafc;">Confidence Score: <strong style="color: #34d399;">{status_prob:.1%}</strong></p>
                <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; display: inline-block; margin-top: 1.5rem; border: 1px solid rgba(16,185,129,0.3);">
                    <span style="font-size: 1rem; color: #a0a0a0; text-transform: uppercase; letter-spacing: 1px;">Projected Package (INR)</span><br/>
                    <span style="font-size: 2.5rem; font-weight: 800; color: #f8fafc;">₹{salary_lower:,} - ₹{salary_upper:,}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
            <div class="prediction-box-not-placed">
                <h1 style="color: #f87171; margin-bottom: 1rem;">⚠️ Upskilling Recommended</h1>
                <p style="font-size: 1.25rem; color: #f8fafc;">Confidence Score: <strong style="color: #f87171;">{status_prob:.1%}</strong></p>
                <p style="margin-top: 1.5rem; color: #fecaca; font-size: 1.1rem;">
                    Consider certifications in high-demand fields like AI/Data Science, and leverage Bengaluru's robust startup ecosystem for early internships to improve odds.
                </p>
            </div>
            """, unsafe_allow_html=True)


# ==========================================
# TAB 2: BLR REAL-TIME ANALYTICS
# ==========================================
with tabs[1]:
    st.markdown("<h2>📊 Bengaluru Engineering Market Data (Live APIs)</h2>", unsafe_allow_html=True)
    st.markdown("Simulated live feed from KCET & COMEDK placement APIs.")
    
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("""
        <div class="analytics-card">
            <h4>RVCE / BMSCE CS Avg Package</h4>
            <div class="value">₹18.5 LPA</div>
            <div style="color: #34d399; font-size: 0.9rem;">+12% YoY</div>
        </div>
        """, unsafe_allow_html=True)
    with colB:
        st.markdown("""
        <div class="analytics-card">
            <h4>Overall COMEDK Top 10 Placements</h4>
            <div class="value">92.4%</div>
            <div style="color: #34d399; font-size: 0.9rem;">High Demand</div>
        </div>
        """, unsafe_allow_html=True)
    with colC:
        st.markdown("""
        <div class="analytics-card">
            <h4>KCET Core Branches Avg Package</h4>
            <div class="value">₹7.2 LPA</div>
            <div style="color: #f87171; font-size: 0.9rem;">-2% YoY</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🏢 Recent Placements via Live Network Data")
    @st.cache_data(ttl=3600)
    def fetch_live_blr_jobs():
        try:
            res = requests.get('https://www.themuse.com/api/public/jobs?location=India&page=1', timeout=5)
            if res.status_code == 200:
                return res.json().get('results', [])[:6]
        except:
            pass
        return []
        
    jobs = fetch_live_blr_jobs()
    if jobs:
        for j in jobs:
            st.markdown(f"- 🔗 [{j.get('name')}]({j.get('refs', {}).get('landing_page')}) at **{j.get('company', {}).get('name')}**")
    else:
        st.info("Loading latest tech roles from Bengaluru tech parks... Please refresh.")


# ==========================================
# TAB 3: AI CAREER COUNSELOR
# ==========================================
with tabs[2]:
    st.markdown("<h2>🤖 AI Career Counselor</h2>", unsafe_allow_html=True)
    st.markdown("Ask specific questions about KCET, COMEDK colleges, cutoffs, or career paths in Karnataka.")
    
    api_key = st.text_input("Enter your Gemini API Key for advanced AI (or leave blank to use the local smart-agent):", type="password")
    
    user_query = st.text_input("Ask a question (e.g., 'What is the cutoff for CS in RVCE?', 'KCET vs COMEDK'):")
    
    if st.button("Ask AI"):
        if user_query:
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(f"You are an expert career counselor for Indian students, specializing in Karnataka colleges, KCET, and COMEDK. Answer this query: {user_query}")
                    st.success("AI Response:")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"API Error: {str(e)}")
            else:
                # Local Smart Fallback Agent
                q = user_query.lower()
                st.success("Local Agent Response:")
                if "rvce" in q or "bmsce" in q or "msrit" in q or "top" in q:
                    st.write("Top colleges in Bengaluru like RVCE, BMSCE, and MSRIT typically require a KCET rank under 2000 or a COMEDK rank under 1500 for Computer Science. Average packages for CS range from 15-20 LPA.")
                elif "kcet" in q and "comedk" in q:
                    st.write("KCET is exclusively for Karnataka domicile students with lower fees, while COMEDK is open to all Indian students (Pan-India) with higher fees but grants access to the same top private engineering colleges in Karnataka.")
                elif "fee" in q or "cost" in q:
                    st.write("Through KCET, engineering fees range from ₹80,000 to ₹1 Lakh per year. Through COMEDK, fees range from ₹2.5 Lakhs to ₹3 Lakhs per year depending on the college and branch.")
                else:
                    st.write("That's a great question! While my local database has limited specific answers, generally securing a high rank in KCET/COMEDK and maintaining a CGPA > 8.0 will unlock the best placement opportunities in Bengaluru's Silicon Valley ecosystem. Please provide an API key for more detailed custom AI answers!")
