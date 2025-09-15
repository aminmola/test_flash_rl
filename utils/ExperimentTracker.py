import json
import csv
import pickle
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import torch
import logging
from typing import Dict, List, Any, Optional, Union

class ExperimentTracker:
    """Comprehensive experiment tracking for 100-round federated learning experiments"""

    def __init__(self, experiment_name: str, output_dir: str = "experiments", max_rounds: int = 100):
        """
        Initialize the experiment tracker

        Parameters:
        - experiment_name: Name of the experiment
        - output_dir: Directory to save results
        - max_rounds: Maximum number of rounds to track
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.max_rounds = max_rounds
        self.start_time = None
        self.end_time = None

        # Create output directory
        self.experiment_dir = os.path.join(output_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Initialize tracking dictionaries
        self.metrics = {
            'rounds': [],
            'accuracy': [],
            'loss': [],
            'f1_score': [],
            'recall': [],
            'precision': [],
            'training_time': [],
            'communication_time': [],
            'total_time': [],
            'selected_clients': [],
            'client_contributions': [],
            'model_weights_norm': [],
            'gradient_norms': [],
            'convergence_metrics': [],
            'resource_usage': []
        }

        # Algorithm-specific metrics
        self.rl_metrics = {
            'q_values': [],
            'epsilon': [],
            'reward': [],
            'loss_dql': [],
            'exploration_rate': [],
            'policy_entropy': []
        }

        # Client-specific tracking
        self.client_metrics = defaultdict(lambda: {
            'selection_frequency': 0,
            'contributions': [],
            'local_accuracy': [],
            'data_size': 0,
            'computation_time': [],
            'communication_cost': []
        })

        # Round-by-round detailed logs
        self.round_logs = []

        # Setup logging
        self.setup_logging()

        # Best model tracking
        self.best_metrics = {
            'best_accuracy': 0,
            'best_f1_score': 0,
            'best_round': 0,
            'best_model_path': None
        }

        print(f"Experiment tracker initialized: {self.experiment_dir}")

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.experiment_dir, "experiment.log")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(self.experiment_name)

    def start_experiment(self, config: Dict[str, Any]):
        """
        Start tracking an experiment

        Parameters:
        - config: Experiment configuration
        """
        self.start_time = time.time()
        self.config = config

        # Save configuration
        config_file = os.path.join(self.experiment_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        self.logger.info(f"Experiment started: {self.experiment_name}")
        self.logger.info(f"Configuration: {config}")

    def log_round(self, round_num: int, metrics: Dict[str, Any],
                  selected_clients: List[int], model_weights: Optional[Dict] = None):
        """
        Log metrics for a single round

        Parameters:
        - round_num: Current round number
        - metrics: Dictionary of metrics for this round
        - selected_clients: List of selected client indices
        - model_weights: Model weights (optional)
        """
        round_start = time.time()

        # Core metrics
        self.metrics['rounds'].append(round_num)
        self.metrics['accuracy'].append(metrics.get('accuracy', 0))
        self.metrics['loss'].append(metrics.get('loss', 0))
        self.metrics['f1_score'].append(metrics.get('f1_score', 0))
        self.metrics['recall'].append(metrics.get('recall', 0))
        self.metrics['precision'].append(metrics.get('precision', 0))

        # Time metrics
        self.metrics['training_time'].append(metrics.get('training_time', 0))
        self.metrics['communication_time'].append(metrics.get('communication_time', 0))
        self.metrics['total_time'].append(time.time() - self.start_time if self.start_time else 0)

        # Client selection
        self.metrics['selected_clients'].append(selected_clients.copy())

        # Model metrics
        if model_weights:
            weights_norm = self._calculate_weights_norm(model_weights)
            self.metrics['model_weights_norm'].append(weights_norm)

        # RL specific metrics
        if 'rl_metrics' in metrics:
            rl_data = metrics['rl_metrics']
            self.rl_metrics['q_values'].append(rl_data.get('q_values', []))
            self.rl_metrics['epsilon'].append(rl_data.get('epsilon', 0))
            self.rl_metrics['reward'].append(rl_data.get('reward', []))
            self.rl_metrics['loss_dql'].append(rl_data.get('loss_dql', 0))

        # Update client-specific metrics
        for client_id in selected_clients:
            self.client_metrics[client_id]['selection_frequency'] += 1

        # Check for best model
        current_accuracy = metrics.get('accuracy', 0)
        if current_accuracy > self.best_metrics['best_accuracy']:
            self.best_metrics['best_accuracy'] = current_accuracy
            self.best_metrics['best_round'] = round_num

            if model_weights:
                best_model_path = os.path.join(self.experiment_dir, f"best_model_round_{round_num}.pt")
                torch.save(model_weights, best_model_path)
                self.best_metrics['best_model_path'] = best_model_path

        # Detailed round log
        round_log = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'selected_clients': selected_clients,
            'processing_time': time.time() - round_start
        }
        self.round_logs.append(round_log)

        # Log to file every 10 rounds
        if round_num % 10 == 0:
            self.save_intermediate_results()

        self.logger.info(f"Round {round_num}: Accuracy={current_accuracy:.4f}, "
                        f"Loss={metrics.get('loss', 0):.4f}, "
                        f"Selected clients={len(selected_clients)}")

    def log_client_contribution(self, client_id: int, contribution_metrics: Dict[str, Any]):
        """
        Log individual client contributions

        Parameters:
        - client_id: Client identifier
        - contribution_metrics: Dictionary of contribution metrics
        """
        self.client_metrics[client_id]['contributions'].append(contribution_metrics)
        self.client_metrics[client_id]['local_accuracy'].append(
            contribution_metrics.get('local_accuracy', 0)
        )
        self.client_metrics[client_id]['data_size'] = contribution_metrics.get('data_size', 0)
        self.client_metrics[client_id]['computation_time'].append(
            contribution_metrics.get('computation_time', 0)
        )
        self.client_metrics[client_id]['communication_cost'].append(
            contribution_metrics.get('communication_cost', 0)
        )

    def log_convergence_metrics(self, round_num: int):
        """
        Calculate and log convergence metrics

        Parameters:
        - round_num: Current round number
        """
        if len(self.metrics['accuracy']) < 2:
            return

        # Calculate convergence metrics
        recent_accuracy = self.metrics['accuracy'][-10:] if len(self.metrics['accuracy']) >= 10 else self.metrics['accuracy']
        accuracy_variance = np.var(recent_accuracy)
        accuracy_trend = np.mean(np.diff(recent_accuracy)) if len(recent_accuracy) > 1 else 0

        convergence_metrics = {
            'round': round_num,
            'accuracy_variance': accuracy_variance,
            'accuracy_trend': accuracy_trend,
            'convergence_score': self._calculate_convergence_score(recent_accuracy)
        }

        self.metrics['convergence_metrics'].append(convergence_metrics)

    def _calculate_convergence_score(self, recent_values: List[float], window_size: int = 5) -> float:
        """Calculate convergence score based on recent values stability"""
        if len(recent_values) < window_size:
            return 0.0

        recent_window = recent_values[-window_size:]
        variance = np.var(recent_window)
        mean_value = np.mean(recent_window)

        # Convergence score: lower variance + higher mean = better convergence
        if mean_value > 0:
            convergence_score = mean_value / (1 + variance)
        else:
            convergence_score = 0.0

        return convergence_score

    def _calculate_weights_norm(self, model_weights: Dict) -> float:
        """Calculate L2 norm of model weights"""
        total_norm = 0.0
        for param_tensor in model_weights.values():
            if torch.is_tensor(param_tensor):
                total_norm += torch.norm(param_tensor).item() ** 2
            else:
                total_norm += np.linalg.norm(param_tensor) ** 2
        return total_norm ** 0.5

    def end_experiment(self):
        """End the experiment and save final results"""
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time if self.start_time else 0

        self.logger.info(f"Experiment completed. Total duration: {total_duration:.2f} seconds")

        # Save all results
        self.save_final_results()

        # Generate final report
        self.generate_final_report()

    def save_intermediate_results(self):
        """Save intermediate results during experiment"""
        # Save metrics as JSON
        metrics_file = os.path.join(self.experiment_dir, "metrics_intermediate.json")
        with open(metrics_file, 'w') as f:
            json.dump(self._serialize_metrics(), f, indent=2, default=str)

        # Save as pickle for complete data
        pickle_file = os.path.join(self.experiment_dir, "metrics_intermediate.pkl")
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                'metrics': self.metrics,
                'rl_metrics': self.rl_metrics,
                'client_metrics': dict(self.client_metrics),
                'round_logs': self.round_logs
            }, f)

    def save_final_results(self):
        """Save final experiment results"""
        # Metrics as JSON
        metrics_file = os.path.join(self.experiment_dir, "final_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self._serialize_metrics(), f, indent=2, default=str)

        # Complete data as pickle
        pickle_file = os.path.join(self.experiment_dir, "complete_results.pkl")
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                'experiment_name': self.experiment_name,
                'config': self.config,
                'metrics': self.metrics,
                'rl_metrics': self.rl_metrics,
                'client_metrics': dict(self.client_metrics),
                'round_logs': self.round_logs,
                'best_metrics': self.best_metrics,
                'start_time': self.start_time,
                'end_time': self.end_time
            }, f)

        # CSV export for easy analysis
        self.export_to_csv()

    def _serialize_metrics(self) -> Dict:
        """Serialize metrics for JSON export"""
        serialized = {}
        for key, value in self.metrics.items():
            if isinstance(value, list):
                serialized[key] = [self._convert_to_serializable(item) for item in value]
            else:
                serialized[key] = self._convert_to_serializable(value)

        # Add RL metrics
        serialized['rl_metrics'] = {}
        for key, value in self.rl_metrics.items():
            serialized['rl_metrics'][key] = [self._convert_to_serializable(item) for item in value]

        # Add best metrics
        serialized['best_metrics'] = self.best_metrics

        return serialized

    def _convert_to_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        if torch.is_tensor(obj):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj

    def export_to_csv(self):
        """Export metrics to CSV files"""
        # Main metrics
        main_metrics_df = pd.DataFrame({
            'round': self.metrics['rounds'],
            'accuracy': self.metrics['accuracy'],
            'loss': self.metrics['loss'],
            'f1_score': self.metrics['f1_score'],
            'recall': self.metrics['recall'],
            'precision': self.metrics['precision'],
            'training_time': self.metrics['training_time'],
            'communication_time': self.metrics['communication_time'],
            'total_time': self.metrics['total_time']
        })

        csv_file = os.path.join(self.experiment_dir, "main_metrics.csv")
        main_metrics_df.to_csv(csv_file, index=False)

        # Client selection frequency
        client_selection_data = []
        for client_id, data in self.client_metrics.items():
            client_selection_data.append({
                'client_id': client_id,
                'selection_frequency': data['selection_frequency'],
                'avg_contribution': np.mean(data.get('contributions', [0])),
                'data_size': data['data_size']
            })

        if client_selection_data:
            client_df = pd.DataFrame(client_selection_data)
            client_csv = os.path.join(self.experiment_dir, "client_metrics.csv")
            client_df.to_csv(client_csv, index=False)

    def generate_plots(self, save_plots: bool = True) -> Dict[str, plt.Figure]:
        """
        Generate comprehensive plots for the experiment

        Parameters:
        - save_plots: Whether to save plots to files

        Returns:
        - Dictionary of generated figures
        """
        figures = {}

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Training Progress Plot
        fig1, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Accuracy over rounds
        axes[0, 0].plot(self.metrics['rounds'], self.metrics['accuracy'], 'b-', linewidth=2)
        axes[0, 0].set_title('Test Accuracy Over Rounds', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, alpha=0.3)

        # Loss over rounds
        axes[0, 1].plot(self.metrics['rounds'], self.metrics['loss'], 'r-', linewidth=2)
        axes[0, 1].set_title('Test Loss Over Rounds', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)

        # F1 Score over rounds (if available)
        if any(self.metrics['f1_score']):
            axes[1, 0].plot(self.metrics['rounds'], self.metrics['f1_score'], 'g-', linewidth=2)
            axes[1, 0].set_title('F1 Score Over Rounds', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].grid(True, alpha=0.3)

        # Training time over rounds
        axes[1, 1].plot(self.metrics['rounds'], self.metrics['training_time'], 'm-', linewidth=2)
        axes[1, 1].set_title('Training Time per Round', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        figures['training_progress'] = fig1

        # 2. Client Selection Analysis
        if self.client_metrics:
            fig2, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Client selection frequency
            client_ids = list(self.client_metrics.keys())
            selection_freq = [self.client_metrics[cid]['selection_frequency'] for cid in client_ids]

            axes[0].bar(range(len(client_ids)), selection_freq)
            axes[0].set_title('Client Selection Frequency', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Client ID')
            axes[0].set_ylabel('Times Selected')
            axes[0].set_xticks(range(0, len(client_ids), max(1, len(client_ids)//10)))

            # Selection frequency distribution
            axes[1].hist(selection_freq, bins=min(20, len(client_ids)), alpha=0.7, edgecolor='black')
            axes[1].set_title('Distribution of Selection Frequencies', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Selection Frequency')
            axes[1].set_ylabel('Number of Clients')

            plt.tight_layout()
            figures['client_analysis'] = fig2

        # 3. RL Metrics (if available)
        if any(self.rl_metrics.get('epsilon', [])):
            fig3, axes = plt.subplots(2, 2, figsize=(16, 12))

            # Epsilon decay
            rounds_rl = list(range(len(self.rl_metrics['epsilon'])))
            axes[0, 0].plot(rounds_rl, self.rl_metrics['epsilon'], 'b-', linewidth=2)
            axes[0, 0].set_title('Epsilon Decay (Exploration Rate)', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Round')
            axes[0, 0].set_ylabel('Epsilon')
            axes[0, 0].grid(True, alpha=0.3)

            # DQL Loss
            if self.rl_metrics.get('loss_dql'):
                axes[0, 1].plot(rounds_rl, self.rl_metrics['loss_dql'], 'r-', linewidth=2)
                axes[0, 1].set_title('DQL Training Loss', fontsize=14, fontweight='bold')
                axes[0, 1].set_xlabel('Round')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].grid(True, alpha=0.3)

            # Reward evolution
            if self.rl_metrics.get('reward'):
                reward_means = [np.mean(r) if isinstance(r, list) and r else 0
                              for r in self.rl_metrics['reward']]
                axes[1, 0].plot(rounds_rl, reward_means, 'g-', linewidth=2)
                axes[1, 0].set_title('Average Reward per Round', fontsize=14, fontweight='bold')
                axes[1, 0].set_xlabel('Round')
                axes[1, 0].set_ylabel('Average Reward')
                axes[1, 0].grid(True, alpha=0.3)

            # Q-values distribution (last round)
            if self.rl_metrics.get('q_values') and self.rl_metrics['q_values'][-1]:
                last_q_values = self.rl_metrics['q_values'][-1]
                if isinstance(last_q_values, list) and last_q_values:
                    axes[1, 1].hist(last_q_values, bins=20, alpha=0.7, edgecolor='black')
                    axes[1, 1].set_title('Q-Values Distribution (Final Round)', fontsize=14, fontweight='bold')
                    axes[1, 1].set_xlabel('Q-Value')
                    axes[1, 1].set_ylabel('Frequency')

            plt.tight_layout()
            figures['rl_metrics'] = fig3

        # Save plots if requested
        if save_plots:
            for name, fig in figures.items():
                plot_path = os.path.join(self.experiment_dir, f"{name}.png")
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved: {plot_path}")

        return figures

    def generate_final_report(self):
        """Generate a comprehensive final report"""
        report_file = os.path.join(self.experiment_dir, "final_report.md")

        with open(report_file, 'w') as f:
            f.write(f"# Experiment Report: {self.experiment_name}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Experiment Summary
            f.write("## Experiment Summary\n\n")
            f.write(f"- **Duration:** {self.end_time - self.start_time:.2f} seconds\n")
            f.write(f"- **Total Rounds:** {len(self.metrics['rounds'])}\n")
            f.write(f"- **Best Accuracy:** {self.best_metrics['best_accuracy']:.4f} (Round {self.best_metrics['best_round']})\n")

            if self.metrics['f1_score'] and any(self.metrics['f1_score']):
                best_f1 = max(self.metrics['f1_score'])
                f.write(f"- **Best F1 Score:** {best_f1:.4f}\n")

            # Configuration
            f.write("\n## Configuration\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.config, indent=2, default=str))
            f.write("\n```\n\n")

            # Performance Analysis
            f.write("## Performance Analysis\n\n")

            if len(self.metrics['accuracy']) > 1:
                final_accuracy = self.metrics['accuracy'][-1]
                initial_accuracy = self.metrics['accuracy'][0]
                improvement = final_accuracy - initial_accuracy

                f.write(f"- **Initial Accuracy:** {initial_accuracy:.4f}\n")
                f.write(f"- **Final Accuracy:** {final_accuracy:.4f}\n")
                f.write(f"- **Total Improvement:** {improvement:.4f}\n")

                # Convergence analysis
                if len(self.metrics['accuracy']) >= 20:
                    last_20_acc = self.metrics['accuracy'][-20:]
                    convergence_variance = np.var(last_20_acc)
                    f.write(f"- **Convergence Variance (last 20 rounds):** {convergence_variance:.6f}\n")

            # Client Analysis
            if self.client_metrics:
                f.write("\n## Client Selection Analysis\n\n")
                selection_frequencies = [data['selection_frequency'] for data in self.client_metrics.values()]
                f.write(f"- **Total Clients:** {len(self.client_metrics)}\n")
                f.write(f"- **Average Selections per Client:** {np.mean(selection_frequencies):.2f}\n")
                f.write(f"- **Selection Std Deviation:** {np.std(selection_frequencies):.2f}\n")
                f.write(f"- **Most Selected Client:** {max(selection_frequencies)} times\n")
                f.write(f"- **Least Selected Client:** {min(selection_frequencies)} times\n")

            # Resource Usage
            f.write("\n## Resource Usage\n\n")
            if self.metrics['training_time']:
                total_training_time = sum(self.metrics['training_time'])
                avg_training_time = np.mean(self.metrics['training_time'])
                f.write(f"- **Total Training Time:** {total_training_time:.2f} seconds\n")
                f.write(f"- **Average Training Time per Round:** {avg_training_time:.2f} seconds\n")

            if self.metrics['communication_time']:
                total_comm_time = sum(self.metrics['communication_time'])
                avg_comm_time = np.mean(self.metrics['communication_time'])
                f.write(f"- **Total Communication Time:** {total_comm_time:.2f} seconds\n")
                f.write(f"- **Average Communication Time per Round:** {avg_comm_time:.2f} seconds\n")

            # Files Generated
            f.write("\n## Generated Files\n\n")
            f.write("- `config.json` - Experiment configuration\n")
            f.write("- `final_metrics.json` - Final metrics in JSON format\n")
            f.write("- `complete_results.pkl` - Complete results in pickle format\n")
            f.write("- `main_metrics.csv` - Main metrics in CSV format\n")
            f.write("- `client_metrics.csv` - Client-specific metrics\n")
            f.write("- `*.png` - Generated plots\n")
            f.write("- `experiment.log` - Detailed experiment logs\n")

        print(f"Final report generated: {report_file}")

    def compare_with_baselines(self, baseline_results: Dict[str, Dict]):
        """
        Compare current experiment with baseline results

        Parameters:
        - baseline_results: Dictionary of baseline results
        """
        comparison_file = os.path.join(self.experiment_dir, "baseline_comparison.json")

        comparison = {
            'current_experiment': {
                'name': self.experiment_name,
                'final_accuracy': self.metrics['accuracy'][-1] if self.metrics['accuracy'] else 0,
                'best_accuracy': self.best_metrics['best_accuracy'],
                'convergence_round': self.best_metrics['best_round'],
                'total_rounds': len(self.metrics['rounds'])
            },
            'baselines': baseline_results,
            'improvements': {}
        }

        # Calculate improvements
        current_acc = self.best_metrics['best_accuracy']
        for baseline_name, baseline_data in baseline_results.items():
            baseline_acc = baseline_data.get('best_accuracy', 0)
            improvement = current_acc - baseline_acc
            comparison['improvements'][baseline_name] = {
                'accuracy_improvement': improvement,
                'relative_improvement': (improvement / baseline_acc * 100) if baseline_acc > 0 else 0
            }

        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)

        print(f"Baseline comparison saved: {comparison_file}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the experiment"""
        if not self.metrics['accuracy']:
            return {}

        return {
            'experiment_name': self.experiment_name,
            'total_rounds': len(self.metrics['rounds']),
            'final_accuracy': self.metrics['accuracy'][-1],
            'best_accuracy': self.best_metrics['best_accuracy'],
            'best_round': self.best_metrics['best_round'],
            'accuracy_improvement': self.metrics['accuracy'][-1] - self.metrics['accuracy'][0],
            'average_training_time': np.mean(self.metrics['training_time']) if self.metrics['training_time'] else 0,
            'total_experiment_time': (self.end_time - self.start_time) if self.end_time and self.start_time else 0,
            'num_clients_used': len(self.client_metrics),
            'selection_fairness': np.std([data['selection_frequency'] for data in self.client_metrics.values()]) if self.client_metrics else 0
        }