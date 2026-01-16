import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Credit Risk Analyzer", page_icon="🏦", layout="centered")

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Gagal memuat assets: {e}")
        return None, None

model, scaler = load_assets()

def preprocess_input(data_dict):
    df_input = pd.DataFrame([data_dict])
    
    df_input['person_inc_div_loan'] = df_input['person_income'] / df_input['loan_amnt']
    df_input['loan_to_emp_length_ratio'] = df_input['person_emp_length'] / df_input['loan_amnt']
    df_input['int_rate_to_loan_amt_ratio'] = df_input['loan_int_rate'] / df_input['loan_amnt']
    
    expected_columns = [
        'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
        'person_inc_div_loan', 'loan_to_emp_length_ratio', 'int_rate_to_loan_amt_ratio', 
        'person_home_ownership_OTHER', 'person_home_ownership_OWN', 'person_home_ownership_RENT', 
        'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 
        'loan_intent_PERSONAL', 'loan_intent_VENTURE', 'loan_grade_B', 'loan_grade_C', 
        'loan_grade_D', 'loan_grade_E', 'loan_grade_F', 'loan_grade_G', 
        'cb_person_default_on_file_Y', 'loan_amnt_class_medium', 
        'loan_amnt_class_large', 'loan_amnt_class_very large'
    ]
    
    df_final = pd.DataFrame(0, index=[0], columns=expected_columns)
    
    for col in expected_columns[:10]:
        df_final[col] = df_input[col].values[0]
    
    if data_dict['home_ownership'] != 'MORTGAGE':
        col = f"person_home_ownership_{data_dict['home_ownership']}"
        if col in df_final.columns: df_final[col] = 1
            
    if data_dict['loan_intent'] != 'DEBTCONSOLIDATION':
        intent_val = data_dict['loan_intent'].replace(" ", "")
        col = f"loan_intent_{intent_val}"
        if col in df_final.columns: df_final[col] = 1
            
    if data_dict['loan_grade'] != 'A':
        col = f"loan_grade_{data_dict['loan_grade']}"
        if col in df_final.columns: df_final[col] = 1
            
    if data_dict['default_on_file'] == 'Y':
        df_final['cb_person_default_on_file_Y'] = 1
        
    val = data_dict['loan_amnt']
    if 5000 < val <= 8000: df_final['loan_amnt_class_medium'] = 1
    elif 8000 < val <= 12500: df_final['loan_amnt_class_large'] = 1
    elif val > 12500: df_final['loan_amnt_class_very large'] = 1

    return df_final

st.title("🏦 Analisis Kelayakan Kredit")
st.markdown("Masukkan data nasabah di bawah ini untuk mendapatkan prediksi risiko.")

if model is None:
    st.error("Assets (model/scaler) tidak ditemukan.")
else:
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Usia Nasabah", 18, 100, 22)
            income = st.number_input("Pendapatan Tahunan (IDR)", 1000, 1000000000, 59000)
            home = st.selectbox("Status Rumah", ["RENT", "OWN", "MORTGAGE", "OTHER"])
            emp_len = st.number_input("Lama Bekerja (Tahun)", 0.0, 150.0, 12.0)
            intent = st.selectbox("Tujuan Pinjaman", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
            
        with col2:
            grade = st.selectbox("Grade Pinjaman", ["A", "B", "C", "D", "E", "F", "G"], index=3) # Default D
            amnt = st.number_input("Jumlah Pinjaman (IDR)", 1000, 1000000000, 35000)
            rate = st.number_input("Suku Bunga (%)", 0.1, 30.0, 16.02)
            default = st.radio("Pernah Default?", ["N", "Y"], index=1, horizontal=True) # Default Y
            hist_len = st.number_input("Lama Riwayat Kredit", 0, 50, 3)

    st.divider()

    if st.button("🔍 JALANKAN ANALISIS", use_container_width=True):
        if emp_len > 40:
            st.warning(f"⚠️ **Peringatan Anomali:** Lama bekerja ({emp_len} thn) tidak lazim (Data Outlier).")
        if age > 80:
            st.warning(f"⚠️ **Peringatan Anomali:** Usia ({age} thn) di luar jangkauan training.")

        raw_data = {
            'person_age': age, 'person_income': income, 'person_emp_length': emp_len,
            'loan_amnt': amnt, 'loan_int_rate': rate, 'loan_percent_income': amnt/income,
            'cb_person_cred_hist_length': hist_len, 'home_ownership': home,
            'loan_intent': intent, 'loan_grade': grade, 'default_on_file': default
        }

        with st.spinner('Menganalisis...'):
            final_input = preprocess_input(raw_data)
            
            input_scaled = scaler.transform(final_input)
            
            prob = model.predict_proba(input_scaled)[0]
            label = "BAD (Risiko Tinggi)" if prob[1] > 0.5 else "GOOD (Risiko Rendah)"
            color = "#FF4B4B" if prob[1] > 0.5 else "#28A745"
            
            st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 25px; border-radius: 12px; border-left: 10px solid {color};">
                    <h2 style="color: {color}; margin: 0;">{label}</h2>
                    <p style="font-size: 20px; margin: 10px 0 0 0;">Probabilitas Gagal Bayar: <strong>{prob[1]:.2%}</strong></p>
                    <p style="font-size: 16px; margin: 0; color: #666;">Probabilitas Aman: {prob[0]:.2%}</p>
                </div>
            """, unsafe_allow_html=True)
            
            with st.expander("🔢 Lihat Data Ter-encode"):
                st.dataframe(final_input)

st.caption("Aplikasi v1.0 | 2026")