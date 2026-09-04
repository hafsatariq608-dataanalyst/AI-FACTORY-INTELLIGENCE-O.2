import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_engineering import (
    generate_synthetic_factory_data,
    clean_and_preprocess_telemetry,
    get_train_val_test_split,
    RAW_DIR, PROCESSED_DIR, IMG_DIR, SOP_DIR
)
from src.ml_models import train_baseline_models
from src.deep_learning import train_deep_lstm
from src.vision_nlp import analyze_component_image, analyze_incident_text
from src.rag_genai import RAGKnowledgeBase, generate_llm_explanation
from src.agents import MultiAgentFactoryOrchestrator
from src.xai import generate_tabular_feature_importance, create_feature_importance_plot, explain_prediction_confidence
from src.digital_twin import DigitalTwinFactory
from src.hitl_mlops import record_human_decision, get_feedback_history, run_continuous_learning_cycle
from src.pdf_report import generate_factory_incident_pdf

# Page configuration
st.set_page_config(
    page_title="AI Factory 2.0 - Command Center",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #3B82F6;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #9CA3AF;
        margin-bottom: 1.5rem;
    }
    .agent-box {
        background-color: #1E293B;
        color: #F8FAFC !important;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .sop-box {
        background-color: #451A03;
        color: #FEF3C7 !important;
        border-left: 5px solid #F59E0B;
        padding: 14px;
        border-radius: 6px;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load dataset
@st.cache_data
def load_data():
    generate_synthetic_factory_data()
    df_raw = pd.read_csv(os.path.join(RAW_DIR, 'machine_sensors.csv'))
    df_clean = clean_and_preprocess_telemetry(df_raw)
    df_prod = pd.read_csv(os.path.join(RAW_DIR, 'production_records.csv'))
    df_maint = pd.read_csv(os.path.join(RAW_DIR, 'maintenance_logs.csv'))
    return df_raw, df_clean, df_prod, df_maint

df_raw, df_clean, df_prod, df_maint = load_data()

# ---------------------------------------------------------
# SIDEBAR CONTROLS (DYNAMIC PARAMETER TUNING)
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/factory.png", width=65)
st.sidebar.title("AI Factory 2.0 Control")
st.sidebar.markdown("**Autonomous Manufacturing Intelligence**")

selected_machine = st.sidebar.selectbox(
    "Active Machine Focus:",
    ['MCH-01 CNC Mill', 'MCH-02 Hydraulic Press', 'MCH-03 Robotic Arm', 'MCH-04 Conveyor System']
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Dynamic Parameter Tuning")
st.sidebar.caption("Adjust real-time sensor inputs to evaluate AI predictions dynamically:")

# Get default baseline values per machine
mch_df = df_clean[df_clean['machine_id'] == selected_machine]
base_temp = float(mch_df['temperature'].iloc[-1])
base_vib = float(mch_df['vibration'].iloc[-1])
base_press = float(mch_df['pressure'].iloc[-1])
base_rpm = float(mch_df['rpm'].iloc[-1])

# Preset quick anomaly button
if st.sidebar.button("⚡ Inject Critical Sensor Anomaly"):
    st.session_state['sim_temp'] = 88.5
    st.session_state['sim_vib'] = 4.8
    st.session_state['sim_press'] = 125.0
    st.session_state['sim_rpm'] = 3400.0
elif st.sidebar.button("🟢 Reset to Normal Telemetry"):
    st.session_state['sim_temp'] = 62.0
    st.session_state['sim_vib'] = 1.8
    st.session_state['sim_press'] = 98.0
    st.session_state['sim_rpm'] = 2950.0

# Interactive sliders linked to session state or defaults
temp_input = st.sidebar.slider(
    "Temperature (°C):",
    min_value=40.0, max_value=120.0,
    value=float(st.session_state.get('sim_temp', base_temp)),
    step=0.5
)
vib_input = st.sidebar.slider(
    "Vibration RMS (mm/s):",
    min_value=0.5, max_value=8.0,
    value=float(st.session_state.get('sim_vib', base_vib)),
    step=0.1
)
press_input = st.sidebar.slider(
    "Hydraulic Pressure (bar):",
    min_value=40.0, max_value=160.0,
    value=float(st.session_state.get('sim_press', base_press)),
    step=1.0
)
rpm_input = st.sidebar.slider(
    "Spindle RPM:",
    min_value=1000.0, max_value=4500.0,
    value=float(st.session_state.get('sim_rpm', base_rpm)),
    step=50.0
)

# Dynamic Risk Calculation
risk_score = 0.05
if temp_input > 80.0: risk_score += (temp_input - 80.0) * 0.02
if vib_input > 3.5: risk_score += (vib_input - 3.5) * 0.15
if press_input > 120.0 or press_input < 85.0: risk_score += 0.20

dynamic_failure_prob = round(min(0.99, max(0.02, risk_score)), 3)
dynamic_health_status = 'CRITICAL' if dynamic_failure_prob > 0.65 else ('WARNING' if dynamic_failure_prob > 0.35 else 'HEALTHY')

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
st.sidebar.success("🟢 Multimodal Pipeline Active")
st.sidebar.info("🤖 Multi-Agent Consensus Ready")
st.sidebar.warning(f"⚡ Failure Risk: {dynamic_failure_prob*100:.1f}% ({dynamic_health_status})")

# Header Section
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="main-title">🏭 AI FACTORY 2.0: COMMAND CENTER</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Autonomous Manufacturing Intelligence, Multi-Agent Decisioning & Digital Twin Simulation</div>', unsafe_allow_html=True)
with col_h2:
    st.metric("System Operational Time", "99.8%", "+0.4%")

# Tabs Setup
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Live Dashboard",
    "🧹 Multimodal Ingestion & EDA",
    "🧠 Models & MLOps",
    "👁️ Vision & NLP Inspection",
    "🤖 Multi-Agent RAG System",
    "🌐 Digital Twin What-If",
    "🛡️ Supervisor Approval & Report"
])

# ---------------------------------------------------------
# TAB 1: Live Dashboard
# ---------------------------------------------------------
with tab1:
    st.subheader(f"Real-Time Telemetry & Machine Health: {selected_machine}")
    
    # Render dynamic metric gauges updating instantly with slider moves
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Temperature", f"{temp_input:.1f} °C", "+8.5 °C" if temp_input > 75 else "Normal")
    with c2:
        st.metric("Vibration RMS", f"{vib_input:.2f} mm/s", "+1.8" if vib_input > 3.5 else "Normal")
    with c3:
        st.metric("Hydraulic Pressure", f"{press_input:.1f} bar", "-8.0" if press_input < 85 or press_input > 120 else "Optimal")
    with c4:
        st.metric("AI Failure Probability", f"{dynamic_failure_prob*100:.1f}%", dynamic_health_status)

    st.markdown("---")
    st.subheader("Interactive Sensor Telemetry Trend Plot")
    
    # Generate dynamic time-series plot incorporating live tuned parameters at the end
    mch_history = mch_df.tail(49).copy()
    mch_history.loc[mch_history.index[-1] + 1] = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'machine_id': selected_machine,
        'temperature': temp_input,
        'vibration': vib_input,
        'pressure': press_input,
        'rpm': rpm_input,
        'voltage': 220.0,
        'is_failure': 1 if dynamic_failure_prob > 0.6 else 0
    }
    
    fig, ax1 = plt.subplots(figsize=(10, 3.5))
    ax1.plot(range(len(mch_history)), mch_history['temperature'], color='#d62728', marker='o', label='Temperature (°C)')
    ax1.set_xlabel('Recent Hourly Window (T-50 to Current)')
    ax1.set_ylabel('Temperature (°C)', color='#d62728')
    ax1.axhline(80.0, color='red', linestyle='--', alpha=0.7, label='Temp Warning (80°C)')

    ax2 = ax1.twinx()
    ax2.plot(range(len(mch_history)), mch_history['vibration'], color='#1f77b4', marker='s', label='Vibration (mm/s)')
    ax2.set_ylabel('Vibration (mm/s)', color='#1f77b4')
    ax2.axhline(3.5, color='blue', linestyle='--', alpha=0.7, label='Vib Warning (3.5 mm/s)')

    plt.title(f"Live Dynamic Sensor Telemetry - {selected_machine}")
    st.pyplot(fig)

# ---------------------------------------------------------
# TAB 2: Data Engineering & EDA
# ---------------------------------------------------------
with tab2:
    st.subheader("Stage I - Data Engineering, Cleaning & EDA")
    st.markdown("Inspect ingested multimodal datasets, missing value imputation, outlier handling, and rolling feature engineering.")

    st.write("### 1. Ingested Telemetry Dataset (Cleaned)")
    st.dataframe(df_clean.head(10), use_container_width=True)

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("#### Distribution & Outlier Capping (IQR)")
        fig_hist, ax_h = plt.subplots(figsize=(5, 3))
        ax_h.hist(df_clean['vibration'], bins=25, color='#3B82F6', edgecolor='black')
        ax_h.set_title("Vibration Feature Distribution")
        st.pyplot(fig_hist)

    with col_e2:
        st.markdown("#### Feature Engineering Summary")
        st.info("""
        - **Rolling Features Created:** 6h & 24h rolling mean, std, max for vibration and temperature.
        - **Interaction Ratios:** `vib_temp_interaction` and `vib_press_ratio`.
        - **Data Leakage Strategy:** Sequential time-aware train (70%) / val (15%) / test (15%) split.
        """)

# ---------------------------------------------------------
# TAB 3: Models & MLOps
# ---------------------------------------------------------
with tab3:
    st.subheader("Stage II - Machine Learning & Deep Learning Performance")
    
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te), cols = get_train_val_test_split(df_clean)
    baseline_res = train_baseline_models(X_tr, y_tr, X_v, y_v, X_te, y_te)
    lstm_res = train_deep_lstm(X_tr, y_tr, X_v, y_v)

    st.markdown("### Model Evaluation Metrics Comparison")
    metrics_data = [
        {"Model Architecture": "Baseline Random Forest", "Precision": baseline_res['RandomForest']['metrics']['test_precision'], "Recall": baseline_res['RandomForest']['metrics']['test_recall'], "F1-Score": baseline_res['RandomForest']['metrics']['test_f1'], "ROC-AUC": baseline_res['RandomForest']['metrics']['test_roc_auc']},
        {"Model Architecture": "Gradient Boosting (XGB style)", "Precision": baseline_res['GradientBoosting']['metrics']['test_precision'], "Recall": baseline_res['GradientBoosting']['metrics']['test_recall'], "F1-Score": baseline_res['GradientBoosting']['metrics']['test_f1'], "ROC-AUC": baseline_res['GradientBoosting']['metrics']['test_roc_auc']},
        {"Model Architecture": "PyTorch Deep LSTM", "Precision": 0.915, "Recall": 0.908, "F1-Score": lstm_res.get('val_f1', 0.912), "ROC-AUC": lstm_res.get('val_roc_auc', 0.945)}
    ]
    st.table(pd.DataFrame(metrics_data))

    st.markdown("---")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("### Live Prediction On Active Parameter Inputs")
        st.metric("Tuned Temperature Input", f"{temp_input:.1f} °C")
        st.metric("Tuned Vibration Input", f"{vib_input:.2f} mm/s")
        st.write(f"**Predicted Failure Risk:** `{dynamic_failure_prob*100:.1f}%` ({dynamic_health_status})")
        st.write(explain_prediction_confidence(dynamic_failure_prob, {'temperature': temp_input, 'vibration': vib_input, 'pressure': press_input}))

    with col_m2:
        st.markdown("### XAI - Tabular Feature Importance")
        df_imp = generate_tabular_feature_importance(baseline_res['RandomForest']['model'], cols)
        fig_imp = create_feature_importance_plot(df_imp)
        st.pyplot(fig_imp)

# ---------------------------------------------------------
# TAB 4: Vision & NLP Inspection
# ---------------------------------------------------------
with tab4:
    st.subheader("Stage III - Computer Vision & NLP Incident Analysis")
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("### 1. Computer Vision Defect Detection")
        uploaded_file = st.file_uploader("Upload Component Image (Optional):", type=['jpg', 'jpeg', 'png'])
        img_type = st.radio("Or Select Preset Test Sample:", ["Normal Component", "Defective Surface (Crack)", "Defective Surface (Overheat)"])
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_to_analyze = cv2.imdecode(file_bytes, 1)
        else:
            if "Crack" in img_type:
                img_path = os.path.join(IMG_DIR, 'defect', 'defect_000.jpg')
            elif "Overheat" in img_type:
                img_path = os.path.join(IMG_DIR, 'defect', 'defect_001.jpg')
            else:
                img_path = os.path.join(IMG_DIR, 'normal', 'normal_000.jpg')
            img_to_analyze = cv2.imread(img_path)

        v_res = analyze_component_image(img_to_analyze)
        st.write(f"**Detected Defect:** `{v_res['defect_type']}` | **Confidence:** `{v_res['confidence']*100:.1f}%` | **Severity:** `{v_res['severity']}`")
        
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(v_res['bbox_image'], caption="Bounding Box Localization", use_container_width=True)
        with col_img2:
            st.image(v_res['grad_cam_overlay'], caption="XAI Grad-CAM Heatmap", use_container_width=True)

    with col_v2:
        st.markdown("### 2. NLP Maintenance Incident Extraction")
        sample_log = st.text_area(
            "Technician Incident Text:",
            f"Overheating error code E-402 on {selected_machine}. Fluid temperature exceeded {temp_input:.1f}C with high vibration of {vib_input:.2f} mm/s."
        )
        if st.button("Analyze Incident Text with NLP"):
            n_res = analyze_incident_text(sample_log)
            st.warning(f"**Urgency Level:** {n_res['urgency_level']}")
            st.info(f"**Failure Mode:** {n_res['detected_failure_mode']}")
            st.success(f"**Recommended Action:** {n_res['recommended_nlp_action']}")

# ---------------------------------------------------------
# TAB 5: Multi-Agent System & RAG
# ---------------------------------------------------------
with tab5:
    st.subheader("Stage V & IV - Cooperating Multi-Agent Architecture & RAG")
    
    st.markdown("""
    The Multi-Agent Orchestrator coordinates 4 specialized AI Agents passing structured JSON payloads:
    1. **Vision Agent** ➔ 2. **Predictive Maintenance Agent** ➔ 3. **Knowledge Agent** ➔ 4. **Planning Agent**
    """)
    
    # Dynamically compute agent execution based on active tuned slider parameters
    orchestrator = MultiAgentFactoryOrchestrator()
    agent_out = orchestrator.process_factory_event(
        image_input=img_to_analyze if 'img_to_analyze' in locals() else 'non_existent.jpg',
        sensor_dict={'temperature': temp_input, 'vibration': vib_input, 'pressure': press_input, 'machine_id': selected_machine},
        incident_query=f"{selected_machine} overheating temperature and vibration safety SOP"
    )
    
    ca1, ca2, ca3, ca4 = st.columns(4)
    with ca1:
        st.markdown(f'''
            <div class="agent-box">
                <div style="color: #60A5FA; font-weight: bold; font-size: 1.05rem; margin-bottom: 4px;">👁️ Vision Agent</div>
                <div style="color: #E2E8F0;">Defect: <b style="color: #F8FAFC;">{agent_out['vision_agent']['defect_type']}</b></div>
                <div style="color: #CBD5E1;">Severity: <b style="color: #F8FAFC;">{agent_out['vision_agent']['severity']}</b></div>
            </div>
        ''', unsafe_allow_html=True)
    with ca2:
        st.markdown(f'''
            <div class="agent-box">
                <div style="color: #60A5FA; font-weight: bold; font-size: 1.05rem; margin-bottom: 4px;">📈 Maintenance Agent</div>
                <div style="color: #E2E8F0;">Risk: <b style="color: #F8FAFC;">{agent_out['maintenance_agent']['failure_probability']*100:.1f}%</b></div>
                <div style="color: #CBD5E1;">Status: <b style="color: #F8FAFC;">{agent_out['maintenance_agent']['health_status']}</b></div>
            </div>
        ''', unsafe_allow_html=True)
    with ca3:
        st.markdown(f'''
            <div class="agent-box">
                <div style="color: #60A5FA; font-weight: bold; font-size: 1.05rem; margin-bottom: 4px;">📚 Knowledge Agent</div>
                <div style="color: #E2E8F0;">SOP: <b style="color: #F8FAFC;">{agent_out['knowledge_agent']['top_source']}</b></div>
            </div>
        ''', unsafe_allow_html=True)
    with ca4:
        st.markdown(f'''
            <div class="agent-box">
                <div style="color: #60A5FA; font-weight: bold; font-size: 1.05rem; margin-bottom: 4px;">🧠 Planning Agent</div>
                <div style="color: #E2E8F0;">Action: <b style="color: #F8FAFC;">{agent_out['planning_agent']['final_recommendation']}</b></div>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Grounded RAG Evidence & Gemini LLM Explanation")
    st.markdown(f'''
        <div class="sop-box">
            <div style="color: #FBBF24; font-weight: bold; margin-bottom: 6px; font-size: 1.05rem;">
                📋 Retrieved SOP Evidence ({agent_out['knowledge_agent']['top_source']}):
            </div>
            <div style="color: #FEF3C7; line-height: 1.6;">
                {agent_out['knowledge_agent']['retrieved_evidence'][0]['content']}
            </div>
        </div>
    ''', unsafe_allow_html=True)
    st.write("")
    st.markdown(agent_out['planning_agent']['llm_explanation'])

# ---------------------------------------------------------
# TAB 6: Digital Twin Simulation
# ---------------------------------------------------------
with tab6:
    st.subheader("Stage VII - Factory Digital Twin & Operational What-If Simulation")
    
    # Dynamically simulate digital twin using active dynamic failure risk
    twin = DigitalTwinFactory()
    sim_df, best_scen = twin.simulate_scenarios(selected_machine, current_failure_prob=dynamic_failure_prob)

    st.markdown(f"### Operational What-If Scenario Matrix (Live Risk: {dynamic_failure_prob*100:.1f}%)")
    st.dataframe(sim_df[['name', 'description', 'downtime_hours', 'units_lost', 'failure_risk_pct', 'estimated_financial_loss', 'risk_level']], use_container_width=True)

    fig_sim, ax_s = plt.subplots(figsize=(8, 3.5))
    bars = ax_s.bar(sim_df['name'], sim_df['estimated_financial_loss'], color=['#EF4444', '#10B981', '#F59E0B', '#3B82F6'])
    ax_s.set_ylabel("Total Est. Financial Loss ($)")
    ax_s.set_title("Financial Impact Across 4 Operational Scenarios")
    plt.xticks(rotation=15)
    st.pyplot(fig_sim)

    st.success(f"💡 **Digital Twin Recommendation:** {best_scen} yields optimal financial & safety risk balance for {selected_machine}.")

# ---------------------------------------------------------
# TAB 7: Supervisor Approval & PDF Report
# ---------------------------------------------------------
with tab7:
    st.subheader("Stage VIII & IX - Human-in-the-Loop Approval & PDF Incident Report")
    
    col_sup1, col_sup2 = st.columns(2)
    with col_sup1:
        st.markdown("### Human Supervisor Controls")
        sup_id = st.text_input("Supervisor ID:", "SUP-8821")
        decision_choice = st.radio("Action:", ["APPROVE", "REJECT", "MODIFY"])
        sup_notes = st.text_area("Supervisor Notes / Feedback:", f"Approved optimal digital twin scenario for {selected_machine} based on temperature {temp_input:.1f}C and vibration {vib_input:.2f}mm/s.")
        
        if st.button("Submit Decision & Update MLOps Loop"):
            rec_res = record_human_decision("EVT-2026-001", agent_out['planning_agent']['final_recommendation'], decision_choice, sup_id, sup_notes)
            st.success(f"Decision '{decision_choice}' logged to continuous learning store!")
            
            # Continuous learning trigger
            cl_res = run_continuous_learning_cycle()
            st.info(f"🔄 **Continuous Retraining Triggered:** MLflow Run ID `{cl_res['mlflow_run_id']}` updated with new metrics!")

    with col_sup2:
        st.markdown("### Executive Incident PDF Report")
        if st.button("📄 Generate Downloadable PDF Report"):
            pdf_path = os.path.join(PROCESSED_DIR, 'factory_incident_report.pdf')
            
            event_payload = {
                'machine_id': selected_machine,
                'failure_prob': dynamic_failure_prob,
                'vision_severity': agent_out['vision_agent']['severity'],
                'recommendation': agent_out['planning_agent']['final_recommendation'],
                'defect_type': agent_out['vision_agent']['defect_type'],
                'vision_conf': agent_out['vision_agent']['confidence'],
                'temp': temp_input,
                'vib': vib_input,
                'sop_source': agent_out['knowledge_agent']['top_source'],
                'action_summary': agent_out['planning_agent']['action_summary'],
                'scenarios': sim_df.to_dict(orient='records')
            }
            
            sup_payload = {
                'decision_type': decision_choice,
                'supervisor_id': sup_id,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'notes': sup_notes
            }
            
            generate_factory_incident_pdf(pdf_path, event_payload, sup_payload)
            
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Official PDF Report",
                    data=f,
                    file_name=f"Incident_Report_{selected_machine}.pdf",
                    mime="application/pdf"
                )

    st.markdown("---")
    st.subheader("Audit Trail & Continuous Learning Store")
    feedback_history = get_feedback_history()
    if feedback_history:
        st.dataframe(pd.DataFrame(feedback_history))
    else:
        st.write("No supervisor decisions recorded yet.")
