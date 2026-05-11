import streamlit as st

from module import predict_crop

st.set_page_config(page_title="Plant Recommendation", layout="centered")

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #f5fcf9;
        background-image: radial-gradient(circle at 10% 20%, #e3faef 0%, transparent 20%), 
                          radial-gradient(circle at 90% 80%, #e3faef 0%, transparent 20%);
    }

    h1, h2, h3, p, label {
        color: #1e293b !important;
    }

    [data-testid="stForm"] {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 40px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #eef2f1;
    }

    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 10px 15px !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #94a3b8 !important;
        opacity: 1 !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 1px #10b981 !important;
    }

    .badge {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 16px;
        border-radius: 50px;
        font-size: 14px;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #bbf7d0;
        margin-bottom: 15px;
    }

    [data-testid="stFormSubmitButton"] > button {
        background-color: #046F0B !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        width: 100% !important;
        border: none !important;
        margin-top: 20px !important;
    }

    [data-testid="stFormSubmitButton"] * {
        color: #ffffff !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    
    .center-header {
        text-align: center !important;
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    .center-header h1 {
        text-align: center !important;
    }
    .center-header p {
        text-align: center !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('''
<div class="center-header" style="padding-top: 20px;">
    <span class="badge">
        Smart Recommendations
    </span>
    <h1 style="font-weight: 800; font-size: 42px; margin-top: 10px;">Get Your Crop Recommendation</h1>
    <p style="color: #64748b !important; font-size: 18px; margin-bottom: 30px;">Enter your soil and climate data to receive personalized crop recommendations from our ML model</p>
</div>
''', unsafe_allow_html=True)

with st.form("plant_recommendation_form"):
    
    # Section NPK
    st.markdown('<h3 style="font-size: 20px; font-weight: 700; margin-bottom: 10px;">Soil Nutrients (NPK)</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        n = st.text_input("Nitrogen (N)", placeholder="0-200")
    with col2:
        p = st.text_input("Phosphorus (P)", placeholder="0-200")
    with col3:
        k = st.text_input("Potassium (K)", placeholder="0-200")

    # Section Properti Tanah
    st.markdown('<h3 style="font-size: 20px; font-weight: 700; margin-top: 20px; margin-bottom: 10px;">Soil Properties</h3>', unsafe_allow_html=True)
    ph = st.text_input("Soil pH", placeholder="3-10")

    # Section Iklim
    st.markdown('<h3 style="font-size: 20px; font-weight: 700; margin-top: 20px; margin-bottom: 10px;">Climate Conditions</h3>', unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)
    with col4:
        temp = st.text_input("Temperature - °C", placeholder="-10 to 50")
    with col5:
        hum = st.text_input("Humidity - %", placeholder="0-100")
    with col6:
        rain = st.text_input("Rainfall - mm", placeholder="0-300")

    submitted = st.form_submit_button("🔍 Get Recommendation")

if submitted:
    if n == "" or p == "" or k == "" or ph == "" or temp == "" or hum == "" or rain == "":
        st.error("⚠️ Warning: Please ensure all input fields are filled before submitting!")
    else:
        try:
            n_val = float(n)
            p_val = float(p)
            k_val = float(k)
            ph_val = float(ph)
            temp_val = float(temp)
            hum_val = float(hum)
            rain_val = float(rain)
            
            errors = []
            if not (0 <= n_val <= 200): errors.append("Nitrogen (N) must be between 0 and 200.")
            if not (0 <= p_val <= 200): errors.append("Phosphorus (P) must be between 0 and 200.")
            if not (0 <= k_val <= 200): errors.append("Potassium (K) must be between 0 and 200.")
            if not (3 <= ph_val <= 10): errors.append("Soil pH must be between 3 and 10.")
            if not (-10 <= temp_val <= 50): errors.append("Temperature must be between -10 and 50 °C.")
            if not (0 <= hum_val <= 100): errors.append("Humidity must be between 0 and 100 %.")
            if not (0 <= rain_val <= 300): errors.append("Rainfall must be between 0 and 300 mm.")
            
            if len(errors) > 0:
                for error_msg in errors:
                    st.error(f"⚠️ {error_msg}")
            
            else:
                with st.spinner("Analyzing soil and climate compatibility..."):
                    import time
                    time.sleep(1.5) 
                    
                    hasil_tanaman = predict_crop(
                        n_val,
                        p_val,
                        k_val,
                        temp_val,
                        hum_val,
                        ph_val,
                        rain_val
                    )

                    icon_tanaman = "🌾" 
                    
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
                        border: 2px solid #10b981;
                        border-radius: 16px;
                        padding: 40px;
                        text-align: center;
                        margin-top: 30px;
                        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
                        animation: fadeIn 0.5s ease-in-out;
                    ">
                        <h3 style="color: #064e3b; margin-bottom: 10px; font-size: 20px; font-weight: 600;">
                            Best Recommended Crop
                        </h3>
                        <h1 style="color: #047857; font-size: 54px; margin: 0; font-weight: 900; text-transform: uppercase;">
                            {icon_tanaman} {hasil_tanaman}
                        </h1>
                        <p style="color: #065f46; margin-top: 15px; font-size: 16px;">
                            Based on the analysis of NPK levels, soil pH, and weather conditions, this land has the highest compatibility for planting <strong>{hasil_tanaman}</strong> to achieve maximum yield.
                        </p>
                    </div>
                    
                    <style>
                        @keyframes fadeIn {{
                            from {{ opacity: 0; transform: translateY(20px); }}
                            to {{ opacity: 1; transform: translateY(0); }}
                        }}
                    </style>
                    """, unsafe_allow_html=True)
                    
                    st.balloons()
                    
        except ValueError:
            st.error("⚠️ Warning: Please ensure you only enter NUMBERS into the form fields. Use a period (.) for decimals.")

st.markdown('<p style="text-align: center; margin-top: 40px; font-size: 14px; color: #94a3b8 !important;">© 2026 Kelompok 8. All rights reserved.</p>', unsafe_allow_html=True)