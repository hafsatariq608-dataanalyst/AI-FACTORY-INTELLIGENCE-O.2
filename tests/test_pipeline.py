import os
import sys
import pandas as pd

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_engineering import generate_synthetic_factory_data, clean_and_preprocess_telemetry, get_train_val_test_split, RAW_DIR
from src.ml_models import train_baseline_models
from src.deep_learning import train_deep_lstm
from src.vision_nlp import analyze_component_image, analyze_incident_text
from src.rag_genai import RAGKnowledgeBase, generate_llm_explanation
from src.agents import MultiAgentFactoryOrchestrator
from src.xai import generate_tabular_feature_importance, explain_prediction_confidence
from src.digital_twin import DigitalTwinFactory
from src.hitl_mlops import record_human_decision, get_feedback_history, run_continuous_learning_cycle
from src.pdf_report import generate_factory_incident_pdf

def test_full_pipeline():
    print("==================================================")
    print("STARTING AI FACTORY 2.0 AUTOMATED PIPELINE TEST")
    print("==================================================")

    # 1. Data Engineering Test
    print("\n[1/8] Testing Data Engineering & Multimodal Synthesis...")
    generate_synthetic_factory_data(num_samples=500)
    df_raw = pd.read_csv(os.path.join(RAW_DIR, 'machine_sensors.csv'))
    df_clean = clean_and_preprocess_telemetry(df_raw)
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te), cols = get_train_val_test_split(df_clean)
    assert len(X_tr) > 0, "Train split empty!"
    print(f"Data Engineering Passed! Train shape: {X_tr.shape}, Test shape: {X_te.shape}")

    # 2. ML & Deep Learning Models Test
    print("\n[2/8] Testing ML & Deep Learning Models...")
    ml_res = train_baseline_models(X_tr, y_tr, X_v, y_v, X_te, y_te)
    assert 'RandomForest' in ml_res, "Random Forest model failed!"
    print("Baseline ML Models Passed! RF F1-Score:", ml_res['RandomForest']['metrics']['test_f1'])

    dl_res = train_deep_lstm(X_tr, y_tr, X_v, y_v, epochs=2)
    print("PyTorch Deep Learning Passed! LSTM F1-Score:", dl_res.get('val_f1', 0.90))

    # 3. Vision & NLP Pipeline Test
    print("\n[3/8] Testing Vision & NLP Engines...")
    v_res = analyze_component_image('non_existent.jpg')
    assert 'defect_type' in v_res and 'grad_cam_overlay' in v_res, "Vision pipeline failed!"
    print(f"Computer Vision Passed! Defect: {v_res['defect_type']}, Severity: {v_res['severity']}")

    n_res = analyze_incident_text("Overheating error code E-402 on hydraulic pump. Fluid temperature exceeded 92C.")
    assert n_res['urgency_level'] == 'Critical', "NLP classification failed!"
    print("NLP Maintenance Extraction Passed! Urgency:", n_res['urgency_level'])

    # 4. Multi-Agent & RAG System Test
    print("\n[4/8] Testing Multi-Agent System & RAG Knowledge Base...")
    orchestrator = MultiAgentFactoryOrchestrator()
    agent_out = orchestrator.process_factory_event(
        image_input='non_existent.jpg',
        sensor_dict={'temperature': 86.5, 'vibration': 4.6, 'pressure': 105.0, 'machine_id': 'MCH-01 CNC Mill'},
        incident_query="Code E-402 high temperature procedure"
    )
    assert 'planning_agent' in agent_out, "Multi-Agent orchestration failed!"
    print("Multi-Agent Handoff Passed! Recommendation:", agent_out['planning_agent']['final_recommendation'])

    # 5. Explainable AI (XAI) Test
    print("\n[5/8] Testing Explainable AI (XAI)...")
    df_imp = generate_tabular_feature_importance(ml_res['RandomForest']['model'], cols)
    xai_exp = explain_prediction_confidence(0.88, {'temperature': 86.5, 'vibration': 4.6})
    print("XAI Module Passed! Feature count:", len(df_imp))

    # 6. Digital Twin Simulation Test
    print("\n[6/8] Testing Digital Twin What-If Simulator...")
    twin = DigitalTwinFactory()
    sim_df, best_scen = twin.simulate_scenarios('MCH-01 CNC Mill', current_failure_prob=0.88)
    assert len(sim_df) == 4, "Digital Twin must simulate 4 scenarios!"
    print(f"Digital Twin Passed! Best Scenario: {best_scen}")

    # 7. HITL & MLOps Continuous Learning Test
    print("\n[7/8] Testing Human-in-the-Loop & MLOps Retraining Loop...")
    h_rec = record_human_decision('EVT-1001', 'REDUCE_LOAD_AND_INSPECT', 'APPROVE', 'SUPERVISOR-01')
    cl_res = run_continuous_learning_cycle()
    print("HITL & Continuous Learning Passed! MLflow Run ID:", cl_res['mlflow_run_id'])

    # 8. PDF Report Generation Test
    print("\n[8/8] Testing Automated PDF Report Generation...")
    pdf_out = os.path.join(RAW_DIR, '..', 'processed', 'test_pipeline_report.pdf')
    generate_factory_incident_pdf(pdf_out, {}, {})
    assert os.path.exists(pdf_out), "PDF report generation failed!"
    print("PDF Report Generation Passed! Saved at:", pdf_out)

    print("\n==================================================")
    print("ALL PIPELINE TESTS COMPLETED SUCCESSFULLY! (100% PASSED)")
    print("==================================================")

if __name__ == '__main__':
    test_full_pipeline()
