"""
Analysis and Visualization for Proven Fact-Based Algorithm Results
Generates tables and plots matching the paper's format
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List


class ProvenFactAnalyzer:
    """Analyze and visualize proven fact simulation results"""
    
    def __init__(self, results_file: str):
        """Load results from JSON file"""
        with open(results_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.sessions = self.data['sessions']
        self.metrics = self.data['metrics']
        self.final_audit = self.data.get('final_audit', {})
        
    def generate_session_table(self) -> pd.DataFrame:
        """Generate table similar to paper's format"""
        table_data = []
        
        for session in self.sessions:
            session_num = session['session_number']
            
            # Count sentences
            total_sentences = 0
            for q in session.get('student_questions', []):
                total_sentences += len([s for s in q.split('.') if s.strip()])
            for exp in session.get('professor_explanations', []):
                total_sentences += len([s for s in exp.split('.') if s.strip()])
            
            hallucinations = session.get('hallucinations_detected', 0)
            hall_rate = session.get('hallucination_rate', 0)
            
            table_data.append({
                'Round': session_num,
                'Total Sentences': total_sentences,
                'Hallucination Sentences': hallucinations,
                'Hallucination Rate (%)': f"{hall_rate * 100:.2f}%"
            })
        
        df = pd.DataFrame(table_data)
        return df
    
    def generate_summary_statistics(self) -> Dict:
        """Generate comprehensive summary"""
        hallucination_rates = [s.get('hallucination_rate', 0) for s in self.sessions]
        
        return {
            'Total Sessions': self.metrics['total_sessions'],
            'Total Sentences': self.metrics['total_sentences'],
            'Total Hallucinations': self.metrics['total_hallucinations'],
            'Final Hallucination Rate': f"{self.metrics['final_hallucination_rate']:.4%}",
            'Data Quality': self.final_audit.get('data_quality', 'N/A'),
            'Mean Session Hallucination Rate': f"{sum(hallucination_rates)/len(hallucination_rates):.4%}" if hallucination_rates else "0.00%",
            'Max Session Hallucination Rate': f"{max(hallucination_rates):.4%}" if hallucination_rates else "0.00%",
            'Sessions with 0% Hallucinations': sum(1 for r in hallucination_rates if r == 0),
            'Referee Resets': self.metrics['referee_resets'],
            'Execution Time': f"{self.metrics['execution_time']:.2f}s"
        }
    
    def analyze_evidence_stages(self) -> pd.DataFrame:
        """Analyze performance by evidence stage"""
        stage_data = {}
        
        for session in self.sessions:
            stage = session.get('stage', 1)
            if stage not in stage_data:
                stage_data[stage] = {
                    'sessions': 0,
                    'total_hallucinations': 0,
                    'total_sentences': 0
                }
            
            stage_data[stage]['sessions'] += 1
            stage_data[stage]['total_hallucinations'] += session.get('hallucinations_detected', 0)
            
            # Count sentences
            sentences = 0
            for q in session.get('student_questions', []):
                sentences += len([s for s in q.split('.') if s.strip()])
            for exp in session.get('professor_explanations', []):
                sentences += len([s for s in exp.split('.') if s.strip()])
            stage_data[stage]['total_sentences'] += sentences
        
        # Create DataFrame
        rows = []
        for stage, data in sorted(stage_data.items()):
            hall_rate = data['total_hallucinations'] / data['total_sentences'] if data['total_sentences'] > 0 else 0
            rows.append({
                'Stage': stage,
                'Sessions': data['sessions'],
                'Evidence Items': len(self.sessions[0].get('available_evidence', [])),  # Approximate
                'Total Sentences': data['total_sentences'],
                'Hallucinations': data['total_hallucinations'],
                'Hallucination Rate': f"{hall_rate:.4%}"
            })
        
        return pd.DataFrame(rows)
    
    def plot_hallucination_trend(self, save_path: str = 'hallucination_trend.png'):
        """Plot hallucination rate across sessions"""
        sessions = [s['session_number'] for s in self.sessions]
        rates = [s.get('hallucination_rate', 0) * 100 for s in self.sessions]
        
        plt.figure(figsize=(12, 6))
        plt.plot(sessions, rates, marker='o', linewidth=2, markersize=8, color='steelblue')
        plt.axhline(y=1.0, color='r', linestyle='--', linewidth=1.5, label='1% threshold', alpha=0.7)
        plt.axhline(y=0.0, color='g', linestyle='--', linewidth=1.5, label='0% (ideal)', alpha=0.7)
        plt.xlabel('Session Number', fontsize=13, fontweight='bold')
        plt.ylabel('Hallucination Rate (%)', fontsize=13, fontweight='bold')
        plt.title('Hallucination Rate Across Learning Sessions', fontsize=15, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle=':')
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trend plot saved to: {save_path}")
        plt.close()
    
    def plot_stage_comparison(self, save_path: str = 'stage_comparison.png'):
        """Compare hallucination rates by evidence stage"""
        df = self.analyze_evidence_stages()
        
        if len(df) == 0:
            print("No stage data available")
            return
        
        # Extract numeric hallucination rates
        stages = df['Stage'].tolist()
        rates = [float(r.strip('%')) for r in df['Hallucination Rate'].tolist()]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(stages, rates, color='teal', alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (stage, rate) in enumerate(zip(stages, rates)):
            plt.text(stage, rate + 0.1, f'{rate:.2f}%', ha='center', fontsize=10, fontweight='bold')
        
        plt.xlabel('Evidence Stage', fontsize=13, fontweight='bold')
        plt.ylabel('Hallucination Rate (%)', fontsize=13, fontweight='bold')
        plt.title('Hallucination Rate by Evidence Stage', fontsize=15, fontweight='bold')
        plt.grid(True, axis='y', alpha=0.3, linestyle=':')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Stage comparison saved to: {save_path}")
        plt.close()
    
    def extract_reasoning_paths(self) -> List[Dict]:
        """Extract A-B-C reasoning paths from transcript"""
        reasoning_paths = []
        
        for session in self.sessions:
            session_num = session['session_number']
            
            for i, explanation in enumerate(session.get('professor_explanations', [])):
                # Look for causal patterns
                if 'because' in explanation.lower() or 'therefore' in explanation.lower():
                    reasoning_paths.append({
                        'session': session_num,
                        'professor': i,
                        'explanation': explanation[:200] + "..." if len(explanation) > 200 else explanation,
                        'pattern': 'A is B because C'
                    })
        
        return reasoning_paths
    
    def analyze_residual_hallucinations(self) -> pd.DataFrame:
        """Analyze residual hallucinations from final audit"""
        residual = self.final_audit.get('residual_hallucinations', [])
        
        if not residual:
            return pd.DataFrame({
                'Type': ['None'],
                'Count': [0],
                'Severity': ['N/A']
            })
        
        # Categorize by type
        type_counts = {}
        for hall in residual:
            h_type = hall.get('type', 'unknown')
            severity = hall.get('severity', 'unknown')
            
            key = f"{h_type} ({severity})"
            type_counts[key] = type_counts.get(key, 0) + 1
        
        df = pd.DataFrame([
            {'Type': k, 'Count': v}
            for k, v in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        
        return df
    
    def generate_latex_table(self, output_file: str = 'results_table.tex'):
        """Generate LaTeX table for paper"""
        df = self.generate_session_table()
        
        latex_str = df.to_latex(
            index=False,
            column_format='cccc',
            caption='Session-by-Session Hallucination Analysis',
            label='tab:sessions',
            escape=False
        )
        
        with open(output_file, 'w') as f:
            f.write(latex_str)
        
        print(f"LaTeX table saved to: {output_file}")
    
    def generate_full_report(self, output_dir: str = './analysis_output'):
        """Generate comprehensive analysis report"""
        Path(output_dir).mkdir(exist_ok=True)
        
        print("\n" + "="*70)
        print("ANALYSIS REPORT")
        print("="*70)
        
        # Summary statistics
        summary = self.generate_summary_statistics()
        print("\nSUMMARY STATISTICS:")
        print("-" * 70)
        for key, value in summary.items():
            print(f"{key:.<50} {value}")
        
        with open(f"{output_dir}/summary.txt", 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        # Session table
        df_sessions = self.generate_session_table()
        df_sessions.to_csv(f"{output_dir}/session_table.csv", index=False)
        print(f"\n✓ Session table saved to: {output_dir}/session_table.csv")
        
        # Stage analysis
        df_stages = self.analyze_evidence_stages()
        if len(df_stages) > 0:
            df_stages.to_csv(f"{output_dir}/stage_analysis.csv", index=False)
            print(f"✓ Stage analysis saved to: {output_dir}/stage_analysis.csv")
        
        # Residual hallucinations
        df_residual = self.analyze_residual_hallucinations()
        df_residual.to_csv(f"{output_dir}/residual_hallucinations.csv", index=False)
        print(f"✓ Residual hallucinations saved to: {output_dir}/residual_hallucinations.csv")
        
        # Reasoning paths
        reasoning_paths = self.extract_reasoning_paths()
        with open(f"{output_dir}/reasoning_paths.json", 'w') as f:
            json.dump(reasoning_paths, f, indent=2, ensure_ascii=False)
        print(f"✓ Reasoning paths saved to: {output_dir}/reasoning_paths.json")
        print(f"  Total A-B-C patterns found: {len(reasoning_paths)}")
        
        # Visualizations
        self.plot_hallucination_trend(f"{output_dir}/hallucination_trend.png")
        self.plot_stage_comparison(f"{output_dir}/stage_comparison.png")
        
        # LaTeX table
        self.generate_latex_table(f"{output_dir}/results_table.tex")
        
        # Final audit details
        if self.final_audit:
            print("\nFINAL AUDIT RESULTS:")
            print("-" * 70)
            print(f"Data Quality: {self.final_audit.get('data_quality', 'N/A')}")
            print(f"Residual Hallucinations: {len(self.final_audit.get('residual_hallucinations', []))}")
            
            if self.final_audit.get('recommendations'):
                print("\nRecommendations:")
                for rec in self.final_audit['recommendations']:
                    print(f"  • {rec}")
        
        print(f"\n{'='*70}")
        print(f"Full report generated in: {output_dir}/")
        print(f"{'='*70}\n")


def compare_multiple_simulations(result_files: List[str], output_file: str = 'comparison.png'):
    """Compare results from multiple simulations"""
    plt.figure(figsize=(14, 7))
    
    for result_file in result_files:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        sessions = [s['session_number'] for s in data['sessions']]
        rates = [s.get('hallucination_rate', 0) * 100 for s in data['sessions']]
        
        label = Path(result_file).stem.replace('_', ' ').title()
        plt.plot(sessions, rates, marker='o', label=label, linewidth=2, markersize=5, alpha=0.8)
    
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=1.5, label='1% threshold', alpha=0.5)
    plt.axhline(y=0.0, color='g', linestyle='--', linewidth=1.5, label='0% (ideal)', alpha=0.5)
    plt.xlabel('Session Number', fontsize=13, fontweight='bold')
    plt.ylabel('Hallucination Rate (%)', fontsize=13, fontweight='bold')
    plt.title('Comparison of Multiple Simulations', fontsize=15, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")
    plt.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_proven_fact.py <results_file.json>")
        print("Example: python analyze_proven_fact.py earth_rotation_results.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    analyzer = ProvenFactAnalyzer(results_file)
    analyzer.generate_full_report()
