import os
import pandas as pd
import numpy as np

class DigitalTwinFactory:
    """
    Software Digital Twin representing factory production line state.
    Simulates 4 operational what-if scenarios when a machine anomaly is detected.
    Calculates expected downtime, production loss, failure risk, and financial impact.
    """
    def __init__(self, hourly_production_rate=120, unit_revenue=45.0, maintenance_cost_per_hr=250.0, catastrophic_failure_cost=15000.0):
        self.hourly_rate = hourly_production_rate
        self.unit_revenue = unit_revenue
        self.maint_cost_per_hr = maintenance_cost_per_hr
        self.catastrophic_cost = catastrophic_failure_cost

    def simulate_scenarios(self, machine_id, current_failure_prob, shift_hours=8):
        """
        Simulates 4 Operational What-If Scenarios over an 8-hour production shift:
        Scenario 1: Continue Operation (Status Quo)
        Scenario 2: Immediate Maintenance Shutdown
        Scenario 3: Reduce Machine Load (-30% Speed)
        Scenario 4: Reroute Production to Backup Line
        """
        results = []

        # Scenario 1: Continue Operation (Status Quo)
        # Risk of breakdown midway through shift
        expected_breakdown_prob = current_failure_prob
        expected_downtime_s1 = shift_hours * expected_breakdown_prob * 0.7
        prod_loss_s1 = expected_downtime_s1 * self.hourly_rate
        risk_cost_s1 = expected_breakdown_prob * self.catastrophic_cost
        total_cost_s1 = (prod_loss_s1 * self.unit_revenue) + risk_cost_s1

        results.append({
            'scenario_id': 'SCENARIO_1',
            'name': 'Continue Operation (Status Quo)',
            'description': 'Keep machine running at full speed without intervention.',
            'downtime_hours': round(expected_downtime_s1, 1),
            'units_lost': int(prod_loss_s1),
            'failure_risk_pct': round(expected_breakdown_prob * 100, 1),
            'estimated_financial_loss': round(total_cost_s1, 2),
            'risk_level': 'CRITICAL' if expected_breakdown_prob > 0.7 else 'HIGH'
        })

        # Scenario 2: Immediate Shutdown & Maintenance
        # Planned maintenance takes 2.5 hours, then machine resumes at 100% capacity
        downtime_s2 = 2.5
        prod_loss_s2 = downtime_s2 * self.hourly_rate
        maint_cost_s2 = downtime_s2 * self.maint_cost_per_hr
        residual_risk_s2 = 0.05
        total_cost_s2 = (prod_loss_s2 * self.unit_revenue) + maint_cost_s2 + (residual_risk_s2 * self.catastrophic_cost)

        results.append({
            'scenario_id': 'SCENARIO_2',
            'name': 'Immediate Maintenance Shutdown',
            'description': 'Halt machine for 2.5h planned repair, then resume at full capacity.',
            'downtime_hours': round(downtime_s2, 1),
            'units_lost': int(prod_loss_s2),
            'failure_risk_pct': round(residual_risk_s2 * 100, 1),
            'estimated_financial_loss': round(total_cost_s2, 2),
            'risk_level': 'LOW'
        })

        # Scenario 3: Reduce Load (-30% Speed)
        # Slow down machine to alleviate thermal & mechanical stress. Reduces failure risk by 60%.
        reduced_rate = self.hourly_rate * 0.70
        downtime_s3 = 0.0
        prod_loss_s3 = (self.hourly_rate - reduced_rate) * shift_hours
        reduced_risk_s3 = current_failure_prob * 0.40
        total_cost_s3 = (prod_loss_s3 * self.unit_revenue) + (reduced_risk_s3 * self.catastrophic_cost)

        results.append({
            'scenario_id': 'SCENARIO_3',
            'name': 'Reduce Load (-30% Speed)',
            'description': 'Operate at 70% capacity to lower stress and delay breakdown until scheduled shift end.',
            'downtime_hours': 0.0,
            'units_lost': int(prod_loss_s3),
            'failure_risk_pct': round(reduced_risk_s3 * 100, 1),
            'estimated_financial_loss': round(total_cost_s3, 2),
            'risk_level': 'MODERATE'
        })

        # Scenario 4: Reroute Production to Backup Line
        # Switch batch to Line-02 (85% throughput efficiency). 1 hour setup switchover time.
        downtime_s4 = 1.0
        rerouted_rate = self.hourly_rate * 0.85
        prod_loss_s4 = (downtime_s4 * self.hourly_rate) + ((self.hourly_rate - rerouted_rate) * (shift_hours - downtime_s4))
        residual_risk_s4 = 0.02
        total_cost_s4 = (prod_loss_s4 * self.unit_revenue) + (residual_risk_s4 * self.catastrophic_cost) + 300.0 # Setup cost

        results.append({
            'scenario_id': 'SCENARIO_4',
            'name': 'Reroute to Backup Line',
            'description': 'Transfer batch to secondary production line (Line 02) with 1h setup.',
            'downtime_hours': round(downtime_s4, 1),
            'units_lost': int(prod_loss_s4),
            'failure_risk_pct': round(residual_risk_s4 * 100, 1),
            'estimated_financial_loss': round(total_cost_s4, 2),
            'risk_level': 'LOW'
        })

        df_sim = pd.DataFrame(results)
        
        # Identify optimal recommendation (minimum financial loss & manageable risk)
        best_row = df_sim.sort_values('estimated_financial_loss').iloc[0]
        optimal_scenario = best_row['name']

        return df_sim, optimal_scenario

if __name__ == '__main__':
    twin = DigitalTwinFactory()
    df_res, best = twin.simulate_scenarios('MCH-01 CNC Mill', current_failure_prob=0.88)
    print("Digital Twin Simulation Results:\n", df_res[['name', 'units_lost', 'failure_risk_pct', 'estimated_financial_loss']])
    print(f"\nOptimal Twin Supported Recommendation: {best}")
