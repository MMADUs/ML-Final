import html
import time

import streamlit as st

from module import predict_crop, predict_crop_proba

# ──────────────────────────────────────────────────────────────
# Plant name mapping (EN → ID)
# ──────────────────────────────────────────────────────────────
PLANT_NAME_ID = {
    "rice": "Padi",
    "maize": "Jagung",
    "chickpea": "Kacang Arab",
    "kidneybeans": "Kacang Merah",
    "pigeonpeas": "Kacang Gude",
    "mothbeans": "Kacang Moth",
    "mungbean": "Kacang Hijau",
    "blackgram": "Kacang Urad",
    "lentil": "Lentil",
    "pomegranate": "Delima",
    "banana": "Pisang",
    "mango": "Mangga",
    "grapes": "Anggur",
    "watermelon": "Semangka",
    "muskmelon": "Melon",
    "orange": "Jeruk",
    "papaya": "Pepaya",
    "coconut": "Kelapa",
    "cotton": "Kapas",
    "jute": "Goni",
    "coffee": "Kopi",
    "groundnuts": "Kacang Tanah",
}


def classify_ph(ph_value: float) -> str:
    if ph_value < 6.5:
        return "acidic"
    if ph_value <= 7.5:
        return "neutral"
    return "alkaline"


def classify_temperature(temp_value: float) -> str:
    if temp_value < 20:
        return "cool"
    if temp_value <= 30:
        return "moderate"
    return "hot"


def safe_ratio(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def fmt_num(value) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def normalize_feature_dict(feat_dict, n, p, k, temp, hum, ph, rain) -> dict:
    """
    Makes the feature-engineering panel robust.
    If predict_crop_proba() does not return feat_dict or uses different keys,
    this function computes the displayed values locally.
    """
    feat_dict = feat_dict or {}

    aliases = {
        "N/P Ratio": ["N/P Ratio", "N_P_Ratio", "np_ratio", "n_p_ratio"],
        "N/K Ratio": ["N/K Ratio", "N_K_Ratio", "nk_ratio", "n_k_ratio"],
        "P/K Ratio": ["P/K Ratio", "P_K_Ratio", "pk_ratio", "p_k_ratio"],
        "NPK Total": ["NPK Total", "NPK_Total", "npk_total"],
        "Humidity/Rain Ratio": ["Humidity/Rain Ratio", "Humidity_Rain_Ratio", "humidity_rain_ratio"],
        "pH Type": ["pH Type", "PH Type", "ph_type", "pH_Type"],
        "Temperature Zone": ["Temperature Zone", "temperature_zone", "Temperature_Zone"],
    }

    def get_value(canonical_key, fallback):
        for key in aliases[canonical_key]:
            if key in feat_dict:
                return feat_dict[key]
        return fallback

    return {
        "N/P Ratio": get_value("N/P Ratio", safe_ratio(n, p)),
        "N/K Ratio": get_value("N/K Ratio", safe_ratio(n, k)),
        "P/K Ratio": get_value("P/K Ratio", safe_ratio(p, k)),
        "NPK Total": get_value("NPK Total", n + p + k),
        "Humidity/Rain Ratio": get_value("Humidity/Rain Ratio", safe_ratio(hum, rain)),
        "pH Type": str(get_value("pH Type", classify_ph(ph))).lower(),
        "Temperature Zone": str(get_value("Temperature Zone", classify_temperature(temp))).lower(),
    }


# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Crop Recommendation", layout="centered")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* Hide Streamlit's native header/top bar/deploy controls */
#MainMenu,
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.stDeployButton {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}

/* Hide "Press Enter to submit form" / input helper text */
[data-testid="InputInstructions"] {
    display: none !important;
}

.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 2rem !important;
}

body, .stApp, .block-container, input, textarea, button, label, p, h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #f0faf4;
    background-image:
        radial-gradient(ellipse at 5% 10%, #bbf7d080 0%, transparent 35%),
        radial-gradient(ellipse at 95% 85%, #a7f3d080 0%, transparent 35%);
}

/* Safer text coloring: do not target every div globally */
h1, h2, h3, p, label {
    color: #1e293b !important;
}

/* Keep Streamlit icons rendering correctly */
[data-testid="stIconMaterial"],
span[data-testid="stIconMaterial"],
.material-symbols-rounded,
.material-icons {
    font-family: 'Material Symbols Rounded' !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-size: 20px !important;
    line-height: 1 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
}

[data-testid="stForm"] {
    background-color: #ffffff;
    border-radius: 20px;
    padding: 36px 40px !important;
    box-shadow: 0 4px 30px rgba(0,0,0,0.06);
    border: 1px solid #d1fae5;
}

.stTextInput > div > div > input {
    background-color: #f8fffe !important;
    color: #1e293b !important;
    border: 1.5px solid #d1fae5 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: 15px !important;
}
.stTextInput > div > div > input::placeholder { color: #94a3b8 !important; }
.stTextInput > div > div > input:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 2px #6ee7b740 !important;
}
.stTextInput label {
    font-weight: 600 !important;
    font-size: 13px !important;
    color: #374151 !important;
}

[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #059669, #047857) !important;
    border-radius: 12px !important;
    padding: 14px 24px !important;
    width: 100% !important;
    border: none !important;
    margin-top: 24px !important;
    box-shadow: 0 4px 14px rgba(5, 150, 105, 0.35) !important;
}
[data-testid="stFormSubmitButton"] * {
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

/* Fixed supported-plant dropdown/expander styling */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1.5px solid #d1fae5 !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 22px rgba(0, 0, 0, 0.04) !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] details {
    border: none !important;
}
[data-testid="stExpander"] summary {
    background: #ffffff !important;
    padding: 14px 18px !important;
    border-radius: 16px !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary svg {
    color: #065f46 !important;
    fill: #065f46 !important;
    font-weight: 700 !important;
}
[data-testid="stExpanderDetails"] {
    background: #ffffff !important;
    padding: 4px 18px 18px 18px !important;
}

.section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #059669 !important;
    margin: 28px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #d1fae5;
}

.tips-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 14px;
    padding: 18px 22px;
    margin-top: 10px;
}
.tips-box p {
    margin: 0 0 6px 0;
    font-size: 13px;
    color: #374151 !important;
    line-height: 1.7;
}
.tips-box p:last-child { margin-bottom: 0; }
.tips-title {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #047857 !important;
    margin-bottom: 10px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.plant-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 8px;
    margin-top: 10px;
}
.plant-chip {
    background: #ffffff;
    border: 1.5px solid #d1fae5;
    border-radius: 10px;
    padding: 9px 12px;
    font-size: 13px;
    font-weight: 600;
    color: #065f46 !important;
    text-align: center;
}
.plant-chip .en {
    font-size: 11px;
    color: #6b7280 !important;
    font-weight: 400;
    margin-top: 2px;
}

.result-card {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 2px solid #10b981;
    border-radius: 20px;
    padding: 36px 40px;
    text-align: center;
    margin-top: 30px;
    box-shadow: 0 10px 40px rgba(16, 185, 129, 0.15);
    animation: fadeUp 0.5s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

.prob-section {
    background: #ffffff;
    border: 1.5px solid #d1fae5;
    border-radius: 16px;
    padding: 22px 24px;
    margin-top: 24px;
    animation: fadeUp 0.5s ease 0.1s both;
}
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 7px;
}
.prob-label {
    width: 138px;
    font-size: 12.5px;
    font-weight: 600;
    color: #065f46 !important;
    text-align: right;
    flex-shrink: 0;
    line-height: 1.3;
}
.prob-label small {
    display: block;
    font-size: 10.5px;
    font-weight: 400;
    color: #94a3b8 !important;
}
.prob-bar-bg {
    flex: 1;
    background: #e9faf3;
    border-radius: 50px;
    height: 9px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 50px;
}
.prob-pct {
    width: 44px;
    font-size: 12px;
    font-weight: 700;
    color: #047857 !important;
    text-align: right;
}

.feat-section {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 16px;
    padding: 22px 24px;
    margin-top: 24px;
    animation: fadeUp 0.5s ease 0.2s both;
}
.feat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 12px;
}
.feat-cell {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 16px;
}
.feat-cell-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #64748b !important;
    margin-bottom: 4px;
}
.feat-cell-value {
    font-size: 20px;
    font-weight: 800;
    color: #0f172a !important;
    letter-spacing: -0.02em;
}
.feat-cell.span2 { grid-column: span 2; }

.section-header {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #059669 !important;
    margin-bottom: 4px;
}

.badge {
    background-color: #dcfce7;
    color: #166534 !important;
    padding: 5px 16px;
    border-radius: 50px;
    font-size: 12px;
    font-weight: 700;
    display: inline-block;
    border: 1.5px solid #bbf7d0;
    margin-bottom: 12px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

@media (max-width: 640px) {
    [data-testid="stForm"] { padding: 24px 20px !important; }
    .plant-grid { grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); }
    .feat-grid { grid-template-columns: 1fr; }
    .feat-cell.span2 { grid-column: span 1; }
    .prob-row { align-items: flex-start; }
    .prob-label { width: 110px; }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="text-align:center; padding-top:24px; padding-bottom:4px;">
    <span class="badge">Smart Recommendations</span>
    <h1 style="font-weight:800; font-size:38px; margin:8px 0 6px; letter-spacing:-0.02em;">
        Get Your Crop Recommendation
    </h1>
    <p style="color:#64748b !important; font-size:16px; margin-bottom:8px;">
        Enter your soil and climate data to receive a personalised crop recommendation by our AI-powered system.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

with st.expander(
    f"View all crops supported by this system ({len(PLANT_NAME_ID)} crops)",
    expanded=False,
):
    chips_html = '<div class="plant-grid">'
    for en, idn in PLANT_NAME_ID.items():
        chips_html += f"""
        <div class="plant-chip">
            <div>{html.escape(idn)}</div>
            <div class="en">{html.escape(en.title())}</div>
        </div>"""
    chips_html += "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)

st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)

with st.form("plant_recommendation_form"):
    st.markdown(
        '<div class="section-label">Soil Nutrients (NPK)</div>', unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        n = st.text_input("Nitrogen (N) — mg/kg", placeholder="e.g. 90")
    with col2:
        p = st.text_input("Phosphorus (P) — mg/kg", placeholder="e.g. 40")
    with col3:
        k = st.text_input("Potassium (K) — mg/kg", placeholder="e.g. 50")

    st.markdown(
        '<div class="section-label">Soil Properties</div>', unsafe_allow_html=True
    )
    ph = st.text_input("Soil pH (3 – 10)", placeholder="e.g. 6.5")

    st.markdown(
        '<div class="section-label">Climate Conditions</div>', unsafe_allow_html=True
    )
    col4, col5, col6 = st.columns(3)
    with col4:
        temp = st.text_input("Temperature — °C", placeholder="e.g. 25")
    with col5:
        hum = st.text_input("Humidity — %", placeholder="e.g. 70")
    with col6:
        rain = st.text_input("Rainfall — mm/month", placeholder="e.g. 120")

    st.markdown(
        """
    <div class="tips-box" style="margin-top:22px">
        <p class="tips-title">How to measure these values</p>
        <p><strong>N, P, K:</strong> For accurate measurement, these usually require soil laboratory testing. Contact your local agricultural extension office for sampling guidance.</p>
        <p><strong>pH:</strong> Does not necessarily require a lab, it can be measured using a calibrated soil pH meter or a soil pH test kit available at most gardening stores.</p>
        <p><strong>Temperature, Humidity, Rainfall:</strong> These can be obtained from a weather station, weather API, or a weather app based on your current location.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    submitted = st.form_submit_button("Get Recommendation")

if submitted:
    values = [n, p, k, ph, temp, hum, rain]

    if any(str(v).strip() == "" for v in values):
        st.error("Warning: Please fill in all fields before submitting.")
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
            if not (0 <= n_val <= 200):
                errors.append("Nitrogen (N) must be between 0 and 200 mg/kg.")
            if not (0 <= p_val <= 200):
                errors.append("Phosphorus (P) must be between 0 and 200 mg/kg.")
            if not (0 <= k_val <= 200):
                errors.append("Potassium (K) must be between 0 and 200 mg/kg.")
            if not (3 <= ph_val <= 10):
                errors.append("Soil pH must be between 3 and 10.")
            if not (-10 <= temp_val <= 50):
                errors.append("Temperature must be between -10 and 50 °C.")
            if not (0 <= hum_val <= 100):
                errors.append("Humidity must be between 0 and 100%.")
            if not (0 <= rain_val <= 300):
                errors.append("Rainfall must be between 0 and 300 mm/month.")

            if errors:
                for e in errors:
                    st.error(f"Warning: {e}")
            else:
                with st.spinner("Analysing soil and climate compatibility..."):
                    time.sleep(0.4)

                    best_crop = predict_crop(
                        n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val
                    )

                    proba_result = predict_crop_proba(
                        n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val
                    )

                if isinstance(proba_result, tuple) and len(proba_result) == 2:
                    proba_dict, raw_feat_dict = proba_result
                else:
                    proba_dict, raw_feat_dict = proba_result, {}

                proba_dict = proba_dict or {}
                feat_dict = normalize_feature_dict(
                    raw_feat_dict,
                    n_val,
                    p_val,
                    k_val,
                    temp_val,
                    hum_val,
                    ph_val,
                    rain_val,
                )

                best_crop_key = str(best_crop).lower()
                id_name = PLANT_NAME_ID.get(best_crop_key, str(best_crop).title())

                # Winner card
                st.markdown(
                    f"""
                <div class="result-card">
                    <p style="color:#065f46 !important; font-size:11px; font-weight:700;
                               letter-spacing:.1em; text-transform:uppercase; margin:0 0 10px;">
                        Best Recommended Crop
                    </p>
                    <h1 style="color:#047857 !important; font-size:50px; margin:0;
                                font-weight:900; letter-spacing:-.02em;">
                        {html.escape(id_name)}
                    </h1>
                    <p style="color:#6b7280 !important; font-size:14px; margin-top:4px; font-weight:500;">
                        {html.escape(str(best_crop).title())}
                    </p>
                    <p style="color:#065f46 !important; margin-top:14px; font-size:14.5px; line-height:1.7; max-width:480px; margin-left:auto; margin-right:auto;">
                        Based on your NPK levels, soil pH, and climate conditions, this land has the
                        highest compatibility for growing <strong>{html.escape(id_name)}</strong> to achieve maximum yield.
                    </p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.balloons()

                # Feature Engineering Panel
                ph_type = str(feat_dict.get("pH Type", "-")).lower()
                temp_zone = str(feat_dict.get("Temperature Zone", "-")).lower()

                ph_colors = {
                    "acidic": ("#fef9c3", "#92400e"),
                    "neutral": ("#dcfce7", "#065f46"),
                    "alkaline": ("#dbeafe", "#1e40af"),
                }
                tz_colors = {
                    "cool": ("#dbeafe", "#1e40af"),
                    "moderate": ("#dcfce7", "#065f46"),
                    "hot": ("#fee2e2", "#991b1b"),
                }
                ph_bg, ph_fg = ph_colors.get(ph_type, ("#f1f5f9", "#475569"))
                tz_bg, tz_fg = tz_colors.get(temp_zone, ("#f1f5f9", "#475569"))

                st.markdown(
                    f"""
                <div class="feat-section">
                    <div class="section-header">Feature Engineering: derived inputs used by the model</div>
                    <div class="feat-grid">
                        <div class="feat-cell">
                            <div class="feat-cell-label">N / P Ratio</div>
                            <div class="feat-cell-value">{fmt_num(feat_dict.get('N/P Ratio'))}</div>
                        </div>
                        <div class="feat-cell">
                            <div class="feat-cell-label">N / K Ratio</div>
                            <div class="feat-cell-value">{fmt_num(feat_dict.get('N/K Ratio'))}</div>
                        </div>
                        <div class="feat-cell">
                            <div class="feat-cell-label">P / K Ratio</div>
                            <div class="feat-cell-value">{fmt_num(feat_dict.get('P/K Ratio'))}</div>
                        </div>
                        <div class="feat-cell">
                            <div class="feat-cell-label">NPK Total (mg/kg)</div>
                            <div class="feat-cell-value">{fmt_num(feat_dict.get('NPK Total'))}</div>
                        </div>
                        <div class="feat-cell span2">
                            <div class="feat-cell-label">Humidity / Rainfall Ratio</div>
                            <div class="feat-cell-value">{fmt_num(feat_dict.get('Humidity/Rain Ratio'))}</div>
                        </div>
                        <div class="feat-cell">
                            <div class="feat-cell-label">pH Classification</div>
                            <div class="feat-cell-value">
                                <span style="background:{ph_bg}; color:{ph_fg}; font-size:13px;
                                             font-weight:700; padding:3px 14px; border-radius:50px;
                                             display:inline-block; margin-top:4px;">
                                    {html.escape(ph_type.title())}
                                </span>
                            </div>
                        </div>
                        <div class="feat-cell">
                            <div class="feat-cell-label">Temperature Zone</div>
                            <div class="feat-cell-value">
                                <span style="background:{tz_bg}; color:{tz_fg}; font-size:13px;
                                             font-weight:700; padding:3px 14px; border-radius:50px;
                                             display:inline-block; margin-top:4px;">
                                    {html.escape(temp_zone.title())}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Probability Bars
                sorted_proba = sorted(
                    proba_dict.items(), key=lambda x: float(x[1]), reverse=True
                )

                bars_html = ""
                for crop_en, prob in sorted_proba:
                    crop_key = str(crop_en).lower()
                    idn_label = PLANT_NAME_ID.get(crop_key, str(crop_en).title())
                    pct = max(0.0, min(float(prob) * 100, 100.0))
                    is_best = crop_key == best_crop_key
                    fill = "#047857" if is_best else "#6ee7b7"
                    fw = "800" if is_best else "600"
                    bars_html += f"""
                    <div class="prob-row">
                        <div class="prob-label" style="font-weight:{fw};">
                            {html.escape(idn_label)}
                            <small>{html.escape(str(crop_en).title())}</small>
                        </div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill"
                                 style="width:{pct:.2f}%; background:{fill};"></div>
                        </div>
                        <div class="prob-pct">{pct:.1f}%</div>
                    </div>"""

                if bars_html:
                    st.markdown(
                        f"""
                    <div class="prob-section">
                        <div class="section-header">Score output for all crops</div>
                        <div style="margin-top:14px">{bars_html}</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No probability scores were returned by the model.")

        except ValueError:
            st.error("Warning: Please enter numbers only. Use a period (.) for decimals.")

st.markdown(
    """
<p style="text-align:center; margin-top:52px; font-size:13px; color:#94a3b8 !important;">
    © 2026 Kelompok 8. All rights reserved.
</p>
""",
    unsafe_allow_html=True,
)
