import os
import json
import numpy as np
import pandas as pd
import cv2
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
IMG_DIR = os.path.join(RAW_DIR, 'images')
SOP_DIR = os.path.join(RAW_DIR, 'sops_and_manuals')

def ensure_directories():
    for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, IMG_DIR, SOP_DIR]:
        os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(IMG_DIR, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(IMG_DIR, 'defect'), exist_ok=True)

def generate_synthetic_factory_data(num_samples=2500, random_seed=42):
    """
    Generates realistic synthetic multimodal factory data for AI Factory 2.0.
    1. Tabular: Production records & Machine telemetry
    2. Time-series: Telemetry (Temp, Vibration, Pressure, RPM, Voltage)
    3. Images: Synthetic component images (Normal vs Defect)
    4. Text & PDF: Maintenance incident text logs and machine SOP manuals
    """
    ensure_directories()
    np.random.seed(random_seed)

    start_date = datetime(2026, 1, 1, 8, 0, 0)
    timestamps = [start_date + timedelta(hours=i) for i in range(num_samples)]
    machines = ['MCH-01 CNC Mill', 'MCH-02 Hydraulic Press', 'MCH-03 Robotic Arm', 'MCH-04 Conveyor System']

    records = []
    for i, ts in enumerate(timestamps):
        mch = machines[i % len(machines)]
        # Baseline normal values
        base_temp = 65.0 if 'CNC' in mch else (75.0 if 'Hydraulic' in mch else 55.0)
        base_vib = 1.2 if 'Robotic' in mch else 2.5
        base_press = 100.0 if 'Hydraulic' in mch else 30.0
        base_rpm = 3000 if 'CNC' in mch else 1500

        # Add noise & periodic trends
        temp = base_temp + np.sin(i / 20) * 5 + np.random.normal(0, 2.5)
        vib = base_vib + np.cos(i / 15) * 0.8 + np.random.normal(0, 0.4)
        press = base_press + np.random.normal(0, 3.0)
        rpm = base_rpm + np.random.normal(0, 50)
        voltage = 220.0 + np.random.normal(0, 4.0)

        # Inject periodic failure symptoms (anomalies)
        is_failure = 0
        failure_type = 'None'
        if i % 180 in range(165, 180): # Anomaly window
            temp += np.random.uniform(15, 30)
            vib += np.random.uniform(2.0, 4.5)
            press += np.random.uniform(10, 25)
            if temp > 88.0 or vib > 5.0:
                is_failure = 1
                failure_type = 'Overheating/Vibration'

        records.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'machine_id': mch,
            'temperature': round(temp, 2),
            'vibration': round(vib, 2),
            'pressure': round(press, 2),
            'rpm': round(rpm, 1),
            'voltage': round(voltage, 2),
            'is_failure': is_failure,
            'failure_type': failure_type
        })

    df_telemetry = pd.DataFrame(records)
    
    # Introduce realistic missing values and outliers for cleaning demo
    missing_idx = np.random.choice(df_telemetry.index, size=25, replace=False)
    df_telemetry.loc[missing_idx, 'vibration'] = np.nan
    outlier_idx = np.random.choice(df_telemetry.index, size=15, replace=False)
    df_telemetry.loc[outlier_idx, 'temperature'] = 999.0 # Synthetic sensor error outlier

    telemetry_path = os.path.join(RAW_DIR, 'machine_sensors.csv')
    df_telemetry.to_csv(telemetry_path, index=False)

    # 2. Production Records Data
    prod_records = []
    for i in range(500):
        mch = machines[i % len(machines)]
        units = np.random.randint(80, 150)
        defects = int(units * np.random.beta(0.5, 10))
        quality = round(max(0, 100 - (defects / units * 100) + np.random.normal(0, 2)), 1)
        defect_type = 'None' if defects == 0 else np.random.choice(['Crack', 'Scratch', 'Overheat', 'Misalignment'], p=[0.4, 0.3, 0.2, 0.1])

        prod_records.append({
            'batch_id': f'BAT-{2026000 + i}',
            'line_id': f'LINE-0{(i%3)+1}',
            'machine_id': mch,
            'timestamp': (start_date + timedelta(hours=i*5)).strftime('%Y-%m-%d %H:%M:%S'),
            'units_produced': units,
            'defect_count': defects,
            'quality_score': quality,
            'defect_type': defect_type
        })
    df_prod = pd.DataFrame(prod_records)
    df_prod.to_csv(os.path.join(RAW_DIR, 'production_records.csv'), index=False)

    # 3. Maintenance Logs Data (Text Modality)
    incident_templates = [
        ("High vibration detected in spindle bearing. Unusual grinding noise heard during high RPM cycle.", "High", "Bearing Replacement Needed"),
        ("Overheating error code E-402 on hydraulic pump. Fluid temperature exceeded 92C.", "Critical", "Hydraulic Coolant Flush"),
        ("Routine inspection complete. Minor belt slippage noted on conveyor drive motor.", "Low", "Adjusted Belt Tension"),
        ("Pressure drop in main actuator line causing alignment errors on component batch.", "Medium", "Replaced Pressure Valve Seal"),
        ("Robotic arm joint 3 jittering during pickup motion. Calibration needed.", "Medium", "Recalibrated Motor Controller"),
        ("Emergency stop tripped due to thermal overload sensor triggering.", "Critical", "Replaced Thermal Fuse & Sensor")
    ]

    maint_records = []
    for i in range(120):
        tmpl = incident_templates[i % len(incident_templates)]
        mch = machines[i % len(machines)]
        maint_records.append({
            'log_id': f'LOG-{1000 + i}',
            'timestamp': (start_date + timedelta(days=i*2)).strftime('%Y-%m-%d %H:%M:%S'),
            'machine_id': mch,
            'incident_text': tmpl[0],
            'urgency_level': tmpl[1],
            'action_taken': tmpl[2],
            'downtime_hours': round(np.random.uniform(0.5, 8.0), 1)
        })
    df_maint = pd.DataFrame(maint_records)
    df_maint.to_csv(os.path.join(RAW_DIR, 'maintenance_logs.csv'), index=False)

    # 4. Machine SOPs & Safety Guides (Text/PDF Modality)
    sops = {
        'CNC_Mill_SOP.txt': """STANDARD OPERATING PROCEDURE: CNC MILL (MCH-01)
1. Overview: High-precision milling center operated under 3-phase AC power.
2. Temperature Thresholds: Normal operational range is 55C - 75C. Warning threshold at 82C. Critical emergency shutdown at 90C.
3. Vibration Safety Limits: Normal RMS vibration is 1.0 - 2.5 mm/s. Alert level at 4.0 mm/s. Spindle failure risk high above 5.5 mm/s.
4. Maintenance Actions for Code E-402:
   - Immediately reduce spindle speed by 30%.
   - Inspect lubricant reservoir level.
   - If vibration exceeds 5.0 mm/s, halt operation and trigger Maintenance Request Tier-2.
5. Safety Protocol: Wear safety goggles, ear protection, and heat-resistant gloves during tool change.""",

        'Hydraulic_Press_Manual.txt': """OPERATIONAL MANUAL: HYDRAULIC PRESS SYSTEM (MCH-02)
1. Operating Pressure: Target pressure is 95 - 110 bar. Maximum allowable pressure limit is 135 bar.
2. Common Failure Modes:
   - Valve Leakage: Indicated by rapid pressure drops under load (<80 bar).
   - Thermal Degradation: Fluid temperature rising above 85C indicates heat exchanger clogging.
3. Emergency Procedures: Press Emergency Stop Red Switch (E-STOP-02). Isolate hydraulic line valve V-12.
4. Inspection Schedule: Check hydraulic seal integrity every 200 operational hours."""
    }

    for sop_name, text in sops.items():
        with open(os.path.join(SOP_DIR, sop_name), 'w', encoding='utf-8') as f:
            f.write(text)

    # 5. Synthetic Inspection Images (Image Modality)
    # Generate synthetic 128x128 images with texture and defect shapes
    for img_idx in range(60):
        # Normal image: smooth grey metal surface with subtle grain
        img_norm = np.random.normal(160, 10, (128, 128, 3)).astype(np.uint8)
        cv2.imwrite(os.path.join(IMG_DIR, 'normal', f'normal_{img_idx:03d}.jpg'), img_norm)

        # Defect image: metal surface with dark scratch / red overheat region / dark crack line
        img_def = np.random.normal(160, 10, (128, 128, 3)).astype(np.uint8)
        defect_kind = img_idx % 3
        if defect_kind == 0: # Crack
            cv2.line(img_def, (20, 30), (100, 110), (20, 20, 20), 3)
        elif defect_kind == 1: # Overheat spot
            cv2.circle(img_def, (64, 64), 30, (30, 40, 200), -1)
        else: # Scratch
            cv2.line(img_def, (10, 80), (110, 20), (50, 50, 50), 2)
        cv2.imwrite(os.path.join(IMG_DIR, 'defect', f'defect_{img_idx:03d}.jpg'), img_def)

    return df_telemetry, df_prod, df_maint

def clean_and_preprocess_telemetry(df):
    """
    Cleans raw telemetry data:
    1. Handles missing values (Median Imputation)
    2. Outlier Detection & Capping using IQR bounds
    3. Feature Engineering: Rolling stats (mean, std, min, max), interaction ratios, lag features
    """
    df = df.copy()

    # Sort chronologically for time-series integrity
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Handle missing values
    num_cols = ['temperature', 'vibration', 'pressure', 'rpm', 'voltage']
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Outlier Detection & IQR Capping
    for col in ['temperature', 'vibration', 'pressure']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 2.5 * iqr
        upper_bound = q3 + 2.5 * iqr
        df[col] = np.clip(df[col], lower_bound, upper_bound)

    # Feature Engineering
    # Rolling window features per machine
    df_featured = []
    for mch, group in df.groupby('machine_id'):
        group = group.copy()
        for window in [6, 24]:
            group[f'temp_roll_mean_{window}h'] = group['temperature'].rolling(window, min_periods=1).mean()
            group[f'temp_roll_std_{window}h'] = group['temperature'].rolling(window, min_periods=1).std().fillna(0)
            group[f'vib_roll_mean_{window}h'] = group['vibration'].rolling(window, min_periods=1).mean()
            group[f'vib_roll_max_{window}h'] = group['vibration'].rolling(window, min_periods=1).max()
            group[f'press_roll_mean_{window}h'] = group['pressure'].rolling(window, min_periods=1).mean()

        # Interaction features
        group['vib_temp_interaction'] = group['vibration'] * group['temperature']
        group['vib_press_ratio'] = group['vibration'] / (group['pressure'] + 1e-5)
        group['rpm_vib_ratio'] = group['rpm'] / (group['vibration'] + 1e-5)

        # Lag features
        group['temp_lag_1'] = group['temperature'].ffill().bfill()
        group['vib_lag_1'] = group['vibration'].ffill().bfill()
        df_featured.append(group)

    df_cleaned = pd.concat(df_featured, axis=0).sort_values('timestamp').reset_index(drop=True)
    return df_cleaned

def get_train_val_test_split(df, target_col='is_failure', train_ratio=0.7, val_ratio=0.15):
    """
    Time-aware sequential train / validation / test split to prevent data leakage in time-series forecasting.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    feature_cols = [c for c in df.columns if c not in ['timestamp', 'machine_id', 'is_failure', 'failure_type']]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols

if __name__ == '__main__':
    print("Generating synthetic datasets...")
    generate_synthetic_factory_data()
    df_raw = pd.read_csv(os.path.join(RAW_DIR, 'machine_sensors.csv'))
    df_clean = clean_and_preprocess_telemetry(df_raw)
    df_clean.to_csv(os.path.join(PROCESSED_DIR, 'machine_sensors_clean.csv'), index=False)
    print(f"Dataset generated & processed successfully! Clean shape: {df_clean.shape}")
