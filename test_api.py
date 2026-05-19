import requests
import json

payload = {
    "is_cgpa": False,
    "ssc_b": "Not Specified",
    "ssc_p": "58",
    "hsc_b": "Not Specified",
    "hsc_s": "Not Specified",
    "hsc_p": "",
    "ug_exam": "Not Specified",
    "degree_t": "Not Specified",
    "degree_p": "",
    "mba_p": "",
    "specialisation": "Not Specified",
    "workex": "Not Specified",
    "etest_p": "",
    "gender": "Not Specified"
}

try:
    res = requests.post("http://127.0.0.1:5000/api/predict", json=payload)
    print("Status:", res.status_code)
    print("Text:", res.text)
except Exception as e:
    print("Error:", e)
