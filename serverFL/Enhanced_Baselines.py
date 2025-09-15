import copy
import timeit
import clientFL.Client as Client
from tqdm import tqdm
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.nn import functional as F
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

class Enhanced_Baselines:
    """Enhanced implementation of FedAvg and FedProx with comprehensive tracking"""

    def __init__(self, num_clients, global_model, dict_clients, loss_fct, B,
                 dataset_test, learning_rate, momentum, clients_info, algorithm="FedAvg"):
        """
        Initialize the Enhanced Baselines server.

        Parameters:
        - algorithm: "FedAvg" or "FedProx"
        """
        self.algorithm = algorithm
        self.N = num_clients
        self.model = global_model
        self.list_clients = []
        self.B = B
        self.dataset_test = dataset_test
        self.testdataloader = DataLoader(self.dataset_test, batch_size=self.B)
        self.dict_clients = dict_clients
        self.loss_function = copy.deepcopy(loss_fct)
        self.number_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Enhanced tracking
        self.clients_info = clients_info
        self.training_history = {
            'rounds': [],
            'accuracy': [],
            'loss': [],
            'f1_score': [],
            'recall': [],
            'precision': [],
            'selected_clients': [],
            'round_times': [],
            'communication_costs': [],
            'client_contributions': []
        }

        self.create_clients(learning_rate, momentum)

    def create_clients(self, learning_rate, momentum):
        """Create client objects with enhanced tracking capabilities"""
        for cpt, client_name in enumerate(self.dict_clients.keys()):
            client = Client.Client(
                self.clients_info[cpt][0],
                self.dict_clients[client_name],
                copy.deepcopy(self.model),
                self.clients_info[cpt][2],
                self.clients_info[cpt][3],
                self.clients_info[cpt][4],
                copy.deepcopy(self.loss_function),
                self.B,
                learning_rate,
                momentum
            )
            self.list_clients.append(client)

    def weight_scaling_factor(self, client, active_clients):
        """Calculate client weight scaling factor"""
        global_count = sum([client_obj.get_size() for client_obj in active_clients])
        local_count = client.get_size()
        return local_count / global_count

    def scale_model_weights(self, weight, scalar):
        """Scale model weights by scalar factor"""
        w_scaled = copy.deepcopy(weight)
        for k in weight.keys():
            w_scaled[k] = scalar * w_scaled[k]
        return w_scaled

    def sum_scaled_weights(self, scaled_weight_list):
        """Aggregate scaled weights"""
        w_avg = copy.deepcopy(scaled_weight_list[0])
        for k in w_avg.keys():
            tmp = torch.zeros_like(scaled_weight_list[0][k], dtype=torch.float32)
            for i in range(len(scaled_weight_list)):
                tmp += scaled_weight_list[i][k]
            w_avg[k].copy_(tmp)
        return w_avg

    def select_clients_iid(self, comm_round, C):
        """IID client selection (random sampling)"""
        client_indices = np.arange(0, len(self.list_clients))
        m = int(max(C * self.N, 1))
        selected_indices = random.sample(list(client_indices), k=m)
        return [self.list_clients[i] for i in selected_indices], selected_indices

    def select_clients_non_iid(self, comm_round, C, strategy="data_size"):
        """Non-IID client selection strategies"""
        m = int(max(C * self.N, 1))

        if strategy == "data_size":
            # Select clients with most data
            client_sizes = [(i, client.get_size()) for i, client in enumerate(self.list_clients)]
            client_sizes.sort(key=lambda x: x[1], reverse=True)
            selected_indices = [x[0] for x in client_sizes[:m]]

        elif strategy == "diverse":
            # Select diverse clients based on data distribution
            selected_indices = self._select_diverse_clients(m)

        elif strategy == "performance_based":
            # Select based on historical performance
            selected_indices = self._select_performance_based_clients(m, comm_round)

        else:  # default to random
            client_indices = np.arange(0, len(self.list_clients))
            selected_indices = random.sample(list(client_indices), k=m)

        return [self.list_clients[i] for i in selected_indices], selected_indices

    def _select_diverse_clients(self, m):
        """Select clients with diverse data distributions"""
        # Simple diversity metric based on data size variation
        client_sizes = [client.get_size() for client in self.list_clients]
        mean_size = np.mean(client_sizes)

        # Select clients with sizes both above and below mean
        above_mean = [i for i, size in enumerate(client_sizes) if size >= mean_size]
        below_mean = [i for i, size in enumerate(client_sizes) if size < mean_size]

        selected = []
        for i in range(m):
            if i % 2 == 0 and above_mean:
                selected.append(above_mean.pop(random.randint(0, len(above_mean)-1)))
            elif below_mean:
                selected.append(below_mean.pop(random.randint(0, len(below_mean)-1)))
            elif above_mean:
                selected.append(above_mean.pop(random.randint(0, len(above_mean)-1)))

        return selected[:m]

    def _select_performance_based_clients(self, m, comm_round):
        """Select clients based on historical performance"""
        if comm_round < 5:  # Not enough history, use random
            return random.sample(range(len(self.list_clients)), m)

        # Simple performance metric (could be enhanced)
        client_performance = [random.random() for _ in self.list_clients]  # Placeholder
        client_indices = list(range(len(self.list_clients)))

        # Sort by performance and select top performers with some randomness
        sorted_clients = sorted(zip(client_indices, client_performance),
                              key=lambda x: x[1], reverse=True)

        # Select top 70% performers randomly
        top_clients = [x[0] for x in sorted_clients[:int(0.7 * len(sorted_clients))]]
        return random.sample(top_clients, min(m, len(top_clients)))

    def calculate_communication_cost(self, active_clients):
        """Calculate communication cost for the round"""
        total_params = self.number_parameters
        total_cost = 0

        for client in active_clients:
            # Simplified communication cost model
            bandwidth = random.choice(client.bandwidth)
            cost = (total_params * 64) / (1000000 * bandwidth)  # in seconds
            total_cost += cost

        return total_cost

    def track_client_contributions(self, active_clients, performance_gain):
        """Track individual client contributions"""
        contributions = []
        for client in active_clients:
            contribution = {
                'client_id': client.client_name,
                'data_size': client.get_size(),
                'performance_gain': performance_gain / len(active_clients),  # Simplified
                'round_time': random.uniform(0.1, 2.0)  # Placeholder
            }
            contributions.append(contribution)
        return contributions

    def global_train(self, comms_round, C, E, mu=0, distribution="iid",
                    client_selection="random", verbose_test=1, verbos=0,
                    type_data="others", save_results=True):
        """
        Enhanced global training with comprehensive tracking

        Parameters:
        - distribution: "iid" or "non_iid"
        - client_selection: "random", "data_size", "diverse", "performance_based"
        """

        if type_data == "Fall":
            return self.global_train_fall(comms_round, C, E, mu, distribution,
                                        client_selection, verbose_test, verbos, save_results)
        else:
            return self.global_train_others(comms_round, C, E, mu, distribution,
                                          client_selection, verbose_test, verbos, save_results)

    def global_train_others(self, comms_round, C, E, mu, distribution,
                          client_selection, verbose_test, verbos, save_results):
        """Enhanced training for non-fall detection tasks"""

        print(f"Starting {self.algorithm} training with {distribution} distribution")
        print(f"Client selection strategy: {client_selection}")

        best_model_weights = {}
        best_accuracy = 0
        previous_accuracy = 0

        self.model.train()

        for comm_round in tqdm(range(comms_round), desc=f"{self.algorithm} Training"):
            round_start_time = timeit.default_timer()

            # Client selection based on distribution type
            if distribution == "iid":
                active_clients, selected_indices = self.select_clients_iid(comm_round, C)
            else:
                active_clients, selected_indices = self.select_clients_non_iid(
                    comm_round, C, client_selection)

            # Track selected clients
            self.training_history['selected_clients'].append(selected_indices)

            if verbos:
                print(f"Round {comm_round + 1}: Selected clients {selected_indices}")

            # Store global weights
            global_weights = self.model.state_dict()
            scaled_local_weight_list = []

            # Train selected clients
            for client in active_clients:
                if self.algorithm == "FedProx":
                    client_w = client.train(copy.deepcopy(global_weights), E, mu, type_data, verbos)
                else:  # FedAvg
                    client_w = client.train(copy.deepcopy(global_weights), E, 0, type_data, verbos)

                # Scale weights
                client_scaling_factor = self.weight_scaling_factor(client, active_clients)
                client_scaled_weight = self.scale_model_weights(client_w, client_scaling_factor)
                scaled_local_weight_list.append(client_scaled_weight)

            # Aggregate weights
            average_weights = self.sum_scaled_weights(scaled_local_weight_list)
            self.model.load_state_dict(average_weights)

            # Evaluate
            accuracy, loss = self.test_others()

            # Calculate metrics
            round_time = timeit.default_timer() - round_start_time
            comm_cost = self.calculate_communication_cost(active_clients)
            performance_gain = max(0, accuracy - previous_accuracy)
            contributions = self.track_client_contributions(active_clients, performance_gain)

            # Update tracking
            self.training_history['rounds'].append(comm_round + 1)
            self.training_history['accuracy'].append(accuracy.item() if torch.is_tensor(accuracy) else accuracy)
            self.training_history['loss'].append(loss)
            self.training_history['round_times'].append(round_time)
            self.training_history['communication_costs'].append(comm_cost)
            self.training_history['client_contributions'].append(contributions)

            # Update best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_weights = copy.deepcopy(average_weights)

            if verbose_test:
                acc_val = accuracy.item() if torch.is_tensor(accuracy) else accuracy
                print(f"Round {comm_round + 1}: Accuracy = {acc_val*100:.2f}%, Loss = {loss:.4f}")

            previous_accuracy = accuracy

        # Cleanup
        for client in self.list_clients:
            client.delete_model()

        # Prepare results
        results = {
            "algorithm": self.algorithm,
            "distribution": distribution,
            "client_selection": client_selection,
            "Best_model_weights": best_model_weights,
            "training_history": self.training_history,
            "final_accuracy": best_accuracy,
            "total_rounds": comms_round
        }

        if save_results:
            self.save_results(results)

        return results

    def global_train_fall(self, comms_round, C, E, mu, distribution,
                         client_selection, verbose_test, verbos, save_results):
        """Enhanced training for fall detection tasks"""

        print(f"Starting {self.algorithm} training for fall detection with {distribution} distribution")

        best_model_weights = {}
        best_f1score = 0
        previous_f1 = 0

        # Add fall-specific metrics to tracking
        self.training_history.update({
            'f1_score': [],
            'recall': [],
            'precision': []
        })

        self.model.train()

        for comm_round in tqdm(range(comms_round), desc=f"{self.algorithm} Fall Detection Training"):
            round_start_time = timeit.default_timer()

            # Client selection
            if distribution == "iid":
                active_clients, selected_indices = self.select_clients_iid(comm_round, C)
            else:
                active_clients, selected_indices = self.select_clients_non_iid(
                    comm_round, C, client_selection)

            self.training_history['selected_clients'].append(selected_indices)

            # Store global weights
            global_weights = self.model.state_dict()
            scaled_local_weight_list = []

            # Train clients
            for client in active_clients:
                if self.algorithm == "FedProx":
                    client_w = client.train(copy.deepcopy(global_weights), E, mu, "Fall", verbos)
                else:
                    client_w = client.train(copy.deepcopy(global_weights), E, 0, "Fall", verbos)

                client_scaling_factor = self.weight_scaling_factor(client, active_clients)
                client_scaled_weight = self.scale_model_weights(client_w, client_scaling_factor)
                scaled_local_weight_list.append(client_scaled_weight)

            # Aggregate
            average_weights = self.sum_scaled_weights(scaled_local_weight_list)
            self.model.load_state_dict(average_weights)

            # Evaluate
            accuracy, f1score, recall, precision, loss = self.test_falldetection()

            # Calculate metrics
            round_time = timeit.default_timer() - round_start_time
            comm_cost = self.calculate_communication_cost(active_clients)
            performance_gain = max(0, f1score - previous_f1)
            contributions = self.track_client_contributions(active_clients, performance_gain)

            # Update tracking
            self.training_history['rounds'].append(comm_round + 1)
            self.training_history['accuracy'].append(accuracy)
            self.training_history['f1_score'].append(f1score)
            self.training_history['recall'].append(recall)
            self.training_history['precision'].append(precision)
            self.training_history['loss'].append(loss)
            self.training_history['round_times'].append(round_time)
            self.training_history['communication_costs'].append(comm_cost)
            self.training_history['client_contributions'].append(contributions)

            # Update best model
            if f1score > best_f1score:
                best_f1score = f1score
                best_model_weights = copy.deepcopy(average_weights)

            if verbose_test:
                print(f"Round {comm_round + 1}: Acc = {accuracy*100:.2f}%, F1 = {f1score*100:.2f}%, Loss = {loss:.4f}")

            previous_f1 = f1score

        # Cleanup
        for client in self.list_clients:
            client.delete_model()

        # Prepare results
        results = {
            "algorithm": self.algorithm,
            "distribution": distribution,
            "client_selection": client_selection,
            "Best_model_weights": best_model_weights,
            "training_history": self.training_history,
            "final_f1score": best_f1score,
            "total_rounds": comms_round
        }

        if save_results:
            self.save_results(results)

        return results

    def test_others(self):
        """Test for non-fall detection tasks"""
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.testdataloader:
                log_probs = self.model(data)
                test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
                pred = log_probs.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.testdataloader.dataset)
        accuracy = correct / len(self.testdataloader.dataset)

        return accuracy, test_loss

    def test_falldetection(self):
        """Test for fall detection tasks"""
        self.model.eval()
        epoch_loss = 0
        correct, total = 0, 0
        targets, preds = [], []

        for X_batch, y_batch in self.testdataloader:
            with torch.no_grad():
                out = self.model(X_batch.float())
                loss = self.loss_function(out, y_batch)

                pred = F.log_softmax(out, dim=1).argmax(dim=1)
                total += y_batch.size(0)
                correct += (pred == y_batch).sum().item()

                targets.extend(y_batch.tolist())
                preds.extend(pred.tolist())

                epoch_loss += loss.item()

        accuracy = correct / total
        f1score = f1_score(targets, preds, zero_division=1)
        recall = recall_score(targets, preds, zero_division=1)
        precision = precision_score(targets, preds, zero_division=1)

        return accuracy, f1score, recall, precision, epoch_loss

    def save_results(self, results):
        """Save training results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/{self.algorithm}_{results['distribution']}_{timestamp}.json"

        os.makedirs("results", exist_ok=True)

        # Convert numpy/tensor types to JSON serializable
        serializable_results = copy.deepcopy(results)
        serializable_results.pop('Best_model_weights', None)  # Remove weights for JSON

        def convert_to_serializable(obj):
            if torch.is_tensor(obj):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(x) for x in obj]
            return obj

        serializable_results = convert_to_serializable(serializable_results)

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {filename}")

    def plot_training_curves(self, save_path=None):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        axes[0, 0].plot(self.training_history['rounds'], self.training_history['accuracy'])
        axes[0, 0].set_title(f'{self.algorithm} - Accuracy over Rounds')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True)

        # Loss
        axes[0, 1].plot(self.training_history['rounds'], self.training_history['loss'])
        axes[0, 1].set_title(f'{self.algorithm} - Loss over Rounds')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)

        # Communication Cost
        axes[1, 0].plot(self.training_history['rounds'], self.training_history['communication_costs'])
        axes[1, 0].set_title(f'{self.algorithm} - Communication Cost over Rounds')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Cost (seconds)')
        axes[1, 0].grid(True)

        # Round Times
        axes[1, 1].plot(self.training_history['rounds'], self.training_history['round_times'])
        axes[1, 1].set_title(f'{self.algorithm} - Round Time over Rounds')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")

        plt.show()

        return fig