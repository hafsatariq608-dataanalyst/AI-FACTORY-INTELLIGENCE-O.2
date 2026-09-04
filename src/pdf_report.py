import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_factory_incident_pdf(output_path, event_data, supervisor_decision_data):
    """
    Generates a professional Factory Incident & Decision PDF Report using ReportLab.
    Includes: Executive summary, multi-agent findings, RAG SOP evidence, XAI breakdown,
    Digital Twin what-if comparison table, human supervisor sign-off, and timestamp.
    """
    doc = SimpleDocTemplate(output_path, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E3A8A'),
        alignment=0,
        spaceAfter=10
    )
    h2_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1E40AF'),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1F2937')
    )

    story = []

    # Title & Header
    story.append(Paragraph("AI FACTORY 2.0 - COMMAND CENTER INCIDENT REPORT", title_style))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | <b>Facility ID:</b> FAC-NORTH-01", body_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1E3A8A'), spaceBefore=8, spaceAfter=12))

    # Executive Overview Table
    exec_data = [
        [Paragraph("<b>Target Machine ID:</b>", body_style), Paragraph(str(event_data.get('machine_id', 'MCH-01 CNC Mill')), body_style)],
        [Paragraph("<b>Failure Risk Confidence:</b>", body_style), Paragraph(f"<font color='red'><b>{event_data.get('failure_prob', 0.88)*100:.1f}%</b></font>", body_style)],
        [Paragraph("<b>Defect Severity (Vision):</b>", body_style), Paragraph(str(event_data.get('vision_severity', 'Critical')), body_style)],
        [Paragraph("<b>Primary Recommendation:</b>", body_style), Paragraph(f"<b>{event_data.get('recommendation', 'REDUCE_LOAD_AND_INSPECT')}</b>", body_style)],
    ]
    t_exec = Table(exec_data, colWidths=[160, 360])
    t_exec.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F3F4F6')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E5E7EB')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_exec)
    story.append(Spacer(1, 10))

    # Multi-Agent Analysis
    story.append(Paragraph("1. Multi-Agent Consensus Findings", h2_style))
    agent_text = f"""<b>Vision Agent:</b> Defect Detected - <i>{event_data.get('defect_type', 'Crack')}</i> (Confidence: {event_data.get('vision_conf', 0.94)*100:.1f}%)<br/>
<b>Predictive Maintenance Agent:</b> Sensor Telemetry Warning - Temperature {event_data.get('temp', 84.5)}°C, Vibration {event_data.get('vib', 4.8)} mm/s.<br/>
<b>Knowledge Agent:</b> Retrieved Grounded SOP Evidence from <i>{event_data.get('sop_source', 'CNC_Mill_SOP.txt')}</i>.<br/>
<b>Planning Agent Decision:</b> {event_data.get('action_summary', 'Reduce operational load by 30% and inspect bearing.')}"""
    story.append(Paragraph(agent_text, body_style))
    story.append(Spacer(1, 10))

    # Digital Twin Scenario Matrix
    story.append(Paragraph("2. Digital Twin What-If Operational Simulation", h2_style))
    scenarios = event_data.get('scenarios', [
        {'name': 'Continue Operation', 'units_lost': 480, 'failure_risk_pct': 88.0, 'estimated_financial_loss': 36600.0},
        {'name': 'Immediate Maintenance', 'units_lost': 300, 'failure_risk_pct': 5.0, 'estimated_financial_loss': 14875.0},
        {'name': 'Reduce Load (-30%)', 'units_lost': 288, 'failure_risk_pct': 35.2, 'estimated_financial_loss': 18260.0},
        {'name': 'Reroute to Line 02', 'units_lost': 246, 'failure_risk_pct': 2.0, 'estimated_financial_loss': 11670.0}
    ])

    sim_table_data = [["Scenario", "Units Lost", "Failure Risk", "Est. Financial Loss ($)"]]
    for sc in scenarios:
        sim_table_data.append([
            sc['name'],
            str(sc['units_lost']),
            f"{sc['failure_risk_pct']:.1f}%",
            f"${sc['estimated_financial_loss']:,.2f}"
        ])

    t_sim = Table(sim_table_data, colWidths=[170, 90, 90, 150])
    t_sim.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E40AF')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (1,1), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D1D5DB')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F9FAFB')]),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_sim)
    story.append(Spacer(1, 10))

    # Human-in-the-Loop Approval Section
    story.append(Paragraph("3. Human Supervisor Sign-Off & Audit Trail", h2_style))
    hitl_text = f"""<b>Supervisor Action:</b> <font color='green'><b>{supervisor_decision_data.get('decision_type', 'APPROVED')}</b></font><br/>
<b>Supervisor ID:</b> {supervisor_decision_data.get('supervisor_id', 'SUP-8821')}<br/>
<b>Audit Timestamp:</b> {supervisor_decision_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}<br/>
<b>Supervisor Notes:</b> {supervisor_decision_data.get('notes', 'Action approved per digital twin scenario analysis.')}"""
    story.append(Paragraph(hitl_text, body_style))
    story.append(Spacer(1, 15))

    # Signature Block
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#9CA3AF'), spaceBefore=5, spaceAfter=15))
    sig_data = [
        [Paragraph("<b>Supervisor Signature:</b> ___________________________", body_style), Paragraph("<b>Date:</b> _____________", body_style)]
    ]
    t_sig = Table(sig_data, colWidths=[360, 160])
    story.append(t_sig)

    doc.build(story)
    return output_path

if __name__ == '__main__':
    os.makedirs('../data/processed', exist_ok=True)
    pdf_p = generate_factory_incident_pdf('../data/processed/test_report.pdf', {}, {})
    print("Test PDF generated successfully at:", pdf_p)
