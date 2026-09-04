import os
import json
from src.vision_nlp import analyze_component_image, analyze_incident_text
from src.rag_genai import RAGKnowledgeBase, generate_llm_explanation

class VisionAgent:
    """
    Agent 1: Vision Agent
    Responsibility: Computer Vision inspection of manufactured components.
    Output: Structured JSON containing defect type, severity, and confidence score.
    """
    def run(self, image_path_or_array):
        res = analyze_component_image(image_path_or_array)
        return {
            'agent_name': 'Vision Agent',
            'status': 'SUCCESS',
            'defect_type': res['defect_type'],
            'confidence': res['confidence'],
            'severity': res['severity'],
            'edge_density': res['edge_density'],
            'grad_cam_overlay': res['grad_cam_overlay'],
            'bbox_image': res['bbox_image']
        }

class PredictiveMaintenanceAgent:
    """
    Agent 2: Predictive Maintenance Agent
    Responsibility: Analyze machine sensor telemetry & time-series trends.
    Output: Structured failure probability, health status, and anomaly flags.
    """
    def run(self, sensor_data_dict):
        temp = sensor_data_dict.get('temperature', 60.0)
        vib = sensor_data_dict.get('vibration', 1.8)
        press = sensor_data_dict.get('pressure', 30.0)
        machine_id = sensor_data_dict.get('machine_id', 'MCH-01 CNC Mill')

        # Calibrated machine-specific failure risk formula
        temp_risk = max(0.0, (temp - 78.0) / 25.0) if temp > 78.0 else 0.0
        vib_risk = max(0.0, (vib - 3.2) / 3.0) if vib > 3.2 else 0.0

        if 'Hydraulic' in machine_id:
            press_risk = 0.30 if (press < 70.0 or press > 135.0) else 0.0
        else:
            press_risk = 0.30 if press > 65.0 else 0.0

        total_risk = 0.05 + (temp_risk * 0.45) + (vib_risk * 0.50) + press_risk
        failure_prob = round(min(0.98, max(0.04, total_risk)), 3)
        
        health_status = 'CRITICAL' if failure_prob > 0.60 else ('WARNING' if failure_prob > 0.30 else 'HEALTHY')

        return {
            'agent_name': 'Predictive Maintenance Agent',
            'status': 'SUCCESS',
            'machine_id': machine_id,
            'failure_probability': failure_prob,
            'health_status': health_status,
            'telemetry_snapshot': {'temperature': temp, 'vibration': vib, 'pressure': press}
        }

class KnowledgeAgent:
    """
    Agent 3: Knowledge Agent
    Responsibility: Retrieve exact SOP rules, safety guidelines, and procedures using RAG.
    Output: Relevant document chunk content, source citation, and rule match.
    """
    def __init__(self, rag_kb=None):
        self.rag = rag_kb if rag_kb is not None else RAGKnowledgeBase()

    def run(self, query):
        evidence_list = self.rag.retrieve_evidence(query, top_k=2)
        return {
            'agent_name': 'Knowledge Agent',
            'status': 'SUCCESS',
            'retrieved_evidence': evidence_list,
            'top_source': evidence_list[0]['source'] if evidence_list else 'Default SOP'
        }

class PlanningDecisionAgent:
    """
    Agent 4: Planning / Decision Agent
    Responsibility: Coordinates output from Vision, Maintenance, and Knowledge agents.
    Synthesizes multi-agent payloads into a unified operational decision recommendation.
    """
    def run(self, vision_output, maint_output, knowledge_output):
        v_severity = vision_output.get('severity', 'None')
        m_prob = maint_output.get('failure_probability', 0.05)
        machine_id = maint_output.get('machine_id', 'MCH-01 CNC Mill')
        top_sop = knowledge_output.get('top_source', 'SOP Manual')

        # Joint decision matrix
        if m_prob > 0.60 or v_severity == 'Critical':
            recommendation = "EMERGENCY_SHUTDOWN_AND_MAINTENANCE"
            priority = "URGENT"
            action_summary = f"Immediately halt {machine_id}. Execute Tier-2 bearing & thermal fuse replacement per {top_sop}."
        elif m_prob > 0.30 or v_severity == 'Moderate':
            recommendation = "REDUCE_LOAD_AND_INSPECT"
            priority = "HIGH"
            action_summary = f"Reduce operating RPM on {machine_id} by 30%. Schedule technician inspection within 2 hours per {top_sop}."
        else:
            recommendation = "CONTINUE_NORMAL_OPERATION"
            priority = "LOW"
            action_summary = f"{machine_id} operating within standard safety envelope. Proceed with current shift production schedule."

        synthesized_payload = {
            'agent_name': 'Planning & Decision Agent',
            'status': 'SUCCESS',
            'final_recommendation': recommendation,
            'priority': priority,
            'action_summary': action_summary,
            'contributing_agents': [
                vision_output['agent_name'],
                maint_output['agent_name'],
                knowledge_output['agent_name']
            ]
        }

        # Generate GenAI explanation over multi-agent consensus
        explanation = generate_llm_explanation(
            prediction_summary={
                'machine_id': machine_id,
                'failure_probability': m_prob,
                'defect_type': vision_output.get('defect_type', 'None'),
                'telemetry': maint_output.get('telemetry_snapshot', {})
            },
            retrieved_evidence=knowledge_output.get('retrieved_evidence', [])
        )
        synthesized_payload['llm_explanation'] = explanation

        return synthesized_payload

class MultiAgentFactoryOrchestrator:
    """
    Multi-Agent Orchestrator executing sequential agent handoffs:
    Vision Agent -> Predictive Maintenance Agent -> Knowledge Agent -> Planning Agent
    """
    def __init__(self):
        self.vision_agent = VisionAgent()
        self.maint_agent = PredictiveMaintenanceAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.planning_agent = PlanningDecisionAgent()

    def process_factory_event(self, image_input, sensor_dict, incident_query):
        # Step 1: Vision Agent
        vision_res = self.vision_agent.run(image_input)
        # Step 2: Predictive Maintenance Agent
        maint_res = self.maint_agent.run(sensor_dict)
        # Step 3: Knowledge Agent
        knowledge_res = self.knowledge_agent.run(incident_query)
        # Step 4: Planning & Decision Agent
        planning_res = self.planning_agent.run(vision_res, maint_res, knowledge_res)

        return {
            'vision_agent': vision_res,
            'maintenance_agent': maint_res,
            'knowledge_agent': knowledge_res,
            'planning_agent': planning_res
        }

if __name__ == '__main__':
    orchestrator = MultiAgentFactoryOrchestrator()
    results = orchestrator.process_factory_event(
        image_input='non_existent.jpg',
        sensor_dict={'temperature': 60.0, 'vibration': 1.8, 'pressure': 30.0, 'machine_id': 'MCH-01 CNC Mill'},
        incident_query="Code E-402 high temperature procedure"
    )
    print("Multi-Agent Handoff Completed!")
    print("Final Planning Recommendation:", results['planning_agent']['final_recommendation'])
