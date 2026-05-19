const API_URL = 'http://127.0.0.1:8000';

function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    
    document.getElementById(tabId).classList.add('active');
    document.querySelector(`button[onclick="switchTab('${tabId}')"]`).classList.add('active');

    if(tabId === 'analytics' && document.getElementById('jobsList').innerHTML.includes('Loading')) {
        loadJobs();
    }
}

function toggleFormat() {
    const isCGPA = document.querySelector('input[name="format"]:checked').value === 'cgpa';
    const labels = document.querySelectorAll('.lbl-fmt');
    const inputs = ['ssc_p', 'hsc_p', 'degree_p', 'mba_p'];
    
    labels.forEach(lbl => lbl.innerText = isCGPA ? '(CGPA)' : '(%)');
    inputs.forEach(id => {
        const el = document.getElementById(id);
        el.max = isCGPA ? "10.0" : "100.0";
        if(el.value) {
            // Rough conversion for UX if they switch
            el.value = isCGPA ? (parseFloat(el.value)/9.5).toFixed(2) : (parseFloat(el.value)*9.5).toFixed(2);
        }
    });
}

document.getElementById('predictForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const is_cgpa = document.querySelector('input[name="format"]:checked').value === 'cgpa';
    
    const payload = {
        is_cgpa: is_cgpa,
        ssc_b: document.getElementById('ssc_b').value,
        ssc_p: document.getElementById('ssc_p').value,
        hsc_b: document.getElementById('hsc_b').value,
        hsc_s: document.getElementById('hsc_s').value,
        hsc_p: document.getElementById('hsc_p').value,
        ug_exam: document.getElementById('ug_exam').value,
        degree_t: document.getElementById('degree_t').value,
        degree_p: document.getElementById('degree_p').value,
        mba_p: document.getElementById('mba_p').value,
        specialisation: document.getElementById('specialisation').value,
        workex: document.getElementById('workex').value,
        etest_p: document.getElementById('etest_p').value,
        gender: document.getElementById('gender').value
    };

    const resBox = document.getElementById('resultBox');
    resBox.style.display = 'block';
    resBox.innerHTML = '<p>Computing AI Matrix...</p>';

    try {
        const res = await fetch(`${API_URL}/api/predict`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        
        if(data.error) {
            resBox.innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
            return;
        }

        const prob = (data.probability * 100).toFixed(1);

        if(data.placed) {
            resBox.className = 'result-box placed';
            resBox.innerHTML = `
                <h1 style="color: #34d399; background: none; -webkit-text-fill-color: #34d399;">✅ Placement Secured</h1>
                <p style="font-size: 1.25rem;">Confidence Score: <strong>${prob}%</strong></p>
                <div class="salary-box">
                    <span style="color: #a0a0a0; text-transform: uppercase;">Projected Package (INR)</span><br/>
                    <span style="font-size: 2.5rem; font-weight: 800;">₹${data.salary_lower.toLocaleString()} - ₹${data.salary_upper.toLocaleString()}</span>
                </div>
            `;
        } else {
            resBox.className = 'result-box not-placed';
            resBox.innerHTML = `
                <h1 style="color: #f87171; background: none; -webkit-text-fill-color: #f87171;">⚠️ Upskilling Recommended</h1>
                <p style="font-size: 1.25rem;">Confidence Score: <strong>${prob}%</strong></p>
                <p style="margin-top: 1rem; color: #fecaca;">Consider certifications and Bengaluru startup internships to improve odds.</p>
            `;
        }
    } catch(err) {
        resBox.innerHTML = `<p style="color:red;">Backend unreachable. Ensure server.py is running.</p>`;
    }
});

async function loadJobs() {
    const list = document.getElementById('jobsList');
    try {
        const res = await fetch(`${API_URL}/api/jobs`);
        const jobs = await res.json();
        if(jobs.length === 0) throw new Error("No jobs");
        
        list.innerHTML = jobs.map(j => `
            <div class="job-card">
                <a href="${j.refs?.landing_page}" target="_blank">${j.name}</a>
                <p style="margin: 0.5rem 0 0; color: #94a3b8; font-size: 0.9rem;">🏢 ${j.company?.name}</p>
            </div>
        `).join('');
    } catch {
        list.innerHTML = '<p>Could not load live jobs. Backend running?</p>';
    }
}

async function sendChat() {
    const input = document.getElementById('chatInput');
    const api_key = document.getElementById('apiKey').value;
    const q = input.value.trim();
    if(!q) return;

    const hist = document.getElementById('chatHistory');
    hist.innerHTML += `<div class="msg user"><strong>You:</strong> ${q}</div>`;
    input.value = '';
    hist.scrollTop = hist.scrollHeight;

    try {
        const res = await fetch(`${API_URL}/api/chat`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query: q, api_key: api_key})
        });
        const data = await res.json();
        
        const prefix = data.type === 'gemini' ? '🧠 Gemini AI:' : '🤖 Smart Agent:';
        if(data.error) {
            hist.innerHTML += `<div class="msg ai" style="color:#f87171;"><strong>Error:</strong> ${data.error}</div>`;
        } else {
            hist.innerHTML += `<div class="msg ai"><strong>${prefix}</strong> ${data.response}</div>`;
        }
    } catch {
        hist.innerHTML += `<div class="msg ai" style="color:#f87171;">Connection failed.</div>`;
    }
    hist.scrollTop = hist.scrollHeight;
}
