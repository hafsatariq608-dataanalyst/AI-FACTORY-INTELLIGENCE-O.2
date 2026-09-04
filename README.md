# AI Factory 2.0 : Autonomous Manufacturing Intelligence & Digital Twin

An integrated, enterprise-grade AI Factory Intelligence Command Center synthesizing multimodal data, predictive ML/DL models, computer vision, NLP, Generative AI & RAG, cooperating AI agents, explainable AI, factory digital twin simulation, Human-in-the-Loop workflows, MLflow MLOps experiment tracking, and automated executive PDF reporting.

---

## 🏗️ End-to-End System Architecture

```
                                  +------------------------------------+
                                  |   Streamlit Web Command Center     |
                                  +------------------+-----------------+
                                                     |
                                                     v
                                  +------------------------------------+
                                  | Multimodal Data Engineering Engine |
                                  | (Tabular, Sensor TS, Image, SOPs)  |
                                  +------------------+-----------------+
                                                     |
          +------------------------------------------+------------------------------------------+
          |                                          |                                          |
          v                                          v                                          v
+-------------------+                      +-------------------+                      +-------------------+
|   Vision Agent    |                      | Maintenance Agent |                      |  Knowledge Agent  |
| (PyTorch CNN/Grad)|                      | (LSTM / ML Model) |                      | (RAG SOP Vector)  |
+---------+---------+                      +---------+---------+                      +---------+---------+
          |                                          |                                          |
          +------------------------------------------+------------------------------------------+
                                                     |
                                                     v
                                  +------------------------------------+
                                  |     Planning & Decision Agent      |
                                  |  (Multi-Agent Consensus Synthesis) |
                                  +------------------+-----------------+
                                                     |
                                                     v
                                  +------------------------------------+
                                  |  Digital Twin What-If Simulator    |
                                  |  (4 Operational Cost/Risk Scenarios|
                                  +------------------+-----------------+
                                                     |
                                                     v
                                  +------------------------------------+
                                  |  Human-in-the-Loop & MLOps Engine  |
                                  | (Approve/Reject + MLflow + PDF)    |
                                  +------------------------------------+
```

---

## 🌟 Key Features Across all 9 Stages

### 1. Data Engineering & Multimodal Ingestion (Stage I)
- Ingests **4 distinct modalities**:
  - **Tabular**: Machine batch production records and quality scores.
  - **Time-Series**: 2,500+ hourly sensor telemetry readings (Temperature, Vibration RMS, Hydraulic Pressure, RPM, Voltage).
  - **Images**: Component inspection surface images (Normal vs Defective - Cracks, Overheating, Scratches).
  - **Text & PDF**: Maintenance logs, incident text notes, and Machine SOP safety manuals (`CNC_Mill_SOP.txt`, `Hydraulic_Press_Manual.txt`).
- Automated data cleaning, median missing value imputation, IQR outlier detection/capping, and rolling 6h/24h statistical feature engineering.
- Time-aware sequential split (70% Train, 15% Validation, 15% Test) ensuring zero data leakage.

### 2. Machine Learning & Deep Learning Core (Stage II)
- **Baseline ML**: Random Forest & Gradient Boosting Classifier for telemetry failure prediction.
- **Deep Learning**: PyTorch `SensorLSTM` model for time-series sequential failure forecasting + PyTorch `DefectCNN` model for component image defect classification.
- Metrics: Precision, Recall, F1-Score, ROC-AUC calculation with explicit Error Analysis (False Negatives vs False Positives breakdown).

### 3. Computer Vision & NLP Engine (Stage III)
- **Vision Pipeline**: Defect detection, severity scoring (Minor, Moderate, Critical), bounding box localization, and **Grad-CAM heatmap overlay visualizer**.
- **NLP Pipeline**: Maintenance log classification, urgency level determination (Low, Medium, High, Critical), failure mode extraction, and action recommendations.

### 4. Generative AI & RAG Knowledge Base (Stage IV)
- TF-IDF vector index for RAG retrieval over machine SOPs and safety manuals.
- Gemini API (`google-genai` SDK) explanation engine providing grounded AI decision explanations citing exact SOP manual sections.

### 5. Multi-Agent System (Stage V)
- 4 cooperating specialized agents passing structured JSON payloads:
  1. **Vision Agent**: Analyzes component images -> returns defect type, severity, confidence score.
  2. **Predictive Maintenance Agent**: Analyzes sensor streams -> returns failure probability and health status.
  3. **Knowledge Agent**: Queries RAG knowledge base -> returns SOP steps and safety rules.
  4. **Planning Agent**: Synthesizes evidence from Vision, Maintenance, and Knowledge agents into an actionable operational recommendation.

### 6. Explainable AI & Digital Twin Simulation (Stage VI & VII)
- **XAI**: Feature Importance bar charts and human-readable confidence sentences explaining prediction drivers.
- **Digital Twin Simulation**: Software representation of factory production line evaluating **4 Operational What-If Scenarios**:
  1. *Continue Operation (Status Quo)*
  2. *Immediate Maintenance Shutdown*
  3. *Reduce Machine Load (-30% Speed)*
  4. *Reroute Production to Backup Line*
  Compares production units lost, downtime hours, failure risk %, and net financial loss ($).

### 7. Human-in-the-Loop, MLOps & PDF Reporting (Stage VIII & IX)
- **HITL Control**: Human supervisor `APPROVE`, `REJECT`, `MODIFY` actions with feedback recording.
- **MLOps**: MLflow experiment tracking (`file:./mlruns`) logging parameters, evaluation metrics, and candidate model registration.
- **Continuous Learning Loop**: Human feedback -> Data store -> Retraining queue -> MLflow metric updates.
- **Executive PDF Reporting**: ReportLab automated PDF report generation with executive summary, multi-agent findings, scenario comparison table, and supervisor signature block.

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run automated test suite verifying all 8 stages
python tests/test_pipeline.py

# 3. Launch Streamlit Web Command Center
streamlit run app.py
```
