import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import random
from typing import Dict, List, Tuple, Union
import copy

class Enhanced_Data_Distribution:
    """Enhanced data distribution utilities for IID and Non-IID scenarios"""

    def __init__(self, dataset, num_clients, seed=42):
        """
        Initialize the Enhanced Data Distribution

        Parameters:
        - dataset: The original dataset
        - num_clients: Number of clients
        - seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.seed = seed

        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Extract labels and organize data
        self.labels = self._extract_labels()
        self.num_classes = len(np.unique(self.labels))
        self.class_indices = self._organize_by_class()

        print(f"Dataset: {len(dataset)} samples, {self.num_classes} classes, {num_clients} clients")

    def _extract_labels(self):
        """Extract labels from dataset"""
        if hasattr(self.dataset, 'targets'):
            return np.array(self.dataset.targets)
        elif hasattr(self.dataset, 'labels'):
            return np.array(self.dataset.labels)
        else:
            # Try to extract from dataset
            labels = []
            for i in range(len(self.dataset)):
                _, label = self.dataset[i]
                labels.append(label)
            return np.array(labels)

    def _organize_by_class(self):
        """Organize sample indices by class"""
        class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            class_indices[label].append(idx)
        return dict(class_indices)

    def create_iid_distribution(self, train_ratio=0.8, min_samples_per_client=10):
        """
        Create IID data distribution across clients

        Parameters:
        - train_ratio: Ratio of training data
        - min_samples_per_client: Minimum samples per client

        Returns:
        - Dictionary with client data distributions
        """
        print("Creating IID distribution...")

        # Split into train and test
        total_samples = len(self.dataset)
        train_size = int(train_ratio * total_samples)

        all_indices = list(range(total_samples))
        np.random.shuffle(all_indices)

        train_indices = all_indices[:train_size]
        test_indices = all_indices[train_size:]

        # Distribute training data evenly across clients
        samples_per_client = max(len(train_indices) // self.num_clients, min_samples_per_client)

        client_distributions = {}

        for client_id in range(self.num_clients):
            start_idx = client_id * samples_per_client
            end_idx = min((client_id + 1) * samples_per_client, len(train_indices))

            if start_idx < len(train_indices):
                client_train_indices = train_indices[start_idx:end_idx]

                # Add some test samples for local validation
                test_samples_per_client = len(test_indices) // self.num_clients
                test_start = client_id * test_samples_per_client
                test_end = min((client_id + 1) * test_samples_per_client, len(test_indices))
                client_test_indices = test_indices[test_start:test_end]

                client_distributions[f"client_{client_id}"] = {
                    'train': client_train_indices,
                    'test': client_test_indices,
                    'total_samples': len(client_train_indices) + len(client_test_indices)
                }

        # Statistics
        self._print_distribution_stats(client_distributions, "IID")

        return client_distributions

    def create_non_iid_dirichlet(self, alpha=0.5, train_ratio=0.8, min_samples_per_client=10):
        """
        Create Non-IID distribution using Dirichlet distribution

        Parameters:
        - alpha: Concentration parameter (lower = more non-IID)
        - train_ratio: Ratio of training data
        - min_samples_per_client: Minimum samples per client

        Returns:
        - Dictionary with client data distributions
        """
        print(f"Creating Non-IID Dirichlet distribution (alpha={alpha})...")

        client_distributions = {}

        # Split each class using Dirichlet distribution
        for class_label, indices in self.class_indices.items():
            # Split into train/test
            train_indices, test_indices = train_test_split(
                indices, train_size=train_ratio, random_state=self.seed + class_label
            )

            # Generate Dirichlet proportions for training data
            proportions = np.random.dirichlet([alpha] * self.num_clients)

            # Distribute training samples
            start_idx = 0
            for client_id in range(self.num_clients):
                num_samples = int(proportions[client_id] * len(train_indices))
                end_idx = start_idx + num_samples

                client_key = f"client_{client_id}"
                if client_key not in client_distributions:
                    client_distributions[client_key] = {'train': [], 'test': []}

                if start_idx < len(train_indices):
                    client_train_samples = train_indices[start_idx:min(end_idx, len(train_indices))]
                    client_distributions[client_key]['train'].extend(client_train_samples)

                start_idx = end_idx

            # Distribute test samples evenly
            test_per_client = len(test_indices) // self.num_clients
            for client_id in range(self.num_clients):
                start_test = client_id * test_per_client
                end_test = min((client_id + 1) * test_per_client, len(test_indices))

                client_key = f"client_{client_id}"
                if start_test < len(test_indices):
                    client_test_samples = test_indices[start_test:end_test]
                    client_distributions[client_key]['test'].extend(client_test_samples)

        # Ensure minimum samples per client
        self._ensure_minimum_samples(client_distributions, min_samples_per_client)

        # Add total samples count
        for client_key in client_distributions:
            train_count = len(client_distributions[client_key]['train'])
            test_count = len(client_distributions[client_key]['test'])
            client_distributions[client_key]['total_samples'] = train_count + test_count

        # Statistics
        self._print_distribution_stats(client_distributions, f"Non-IID Dirichlet (α={alpha})")

        return client_distributions

    def create_non_iid_pathological(self, classes_per_client=2, train_ratio=0.8):
        """
        Create pathological Non-IID distribution (each client has only few classes)

        Parameters:
        - classes_per_client: Number of classes per client
        - train_ratio: Ratio of training data

        Returns:
        - Dictionary with client data distributions
        """
        print(f"Creating Pathological Non-IID distribution ({classes_per_client} classes per client)...")

        client_distributions = {}

        # Assign classes to clients
        all_classes = list(range(self.num_classes))
        classes_per_client = min(classes_per_client, self.num_classes)

        client_class_assignment = {}
        for client_id in range(self.num_clients):
            # Rotate through classes to ensure coverage
            start_class = (client_id * classes_per_client) % self.num_classes
            assigned_classes = []
            for i in range(classes_per_client):
                class_idx = (start_class + i) % self.num_classes
                assigned_classes.append(all_classes[class_idx])

            client_class_assignment[client_id] = assigned_classes

        # Distribute data based on class assignment
        for client_id in range(self.num_clients):
            client_key = f"client_{client_id}"
            client_distributions[client_key] = {'train': [], 'test': []}

            assigned_classes = client_class_assignment[client_id]

            for class_label in assigned_classes:
                indices = self.class_indices[class_label]

                # Split train/test for this class
                train_indices, test_indices = train_test_split(
                    indices, train_size=train_ratio, random_state=self.seed + class_label
                )

                # Distribute among clients with this class
                clients_with_class = [cid for cid, classes in client_class_assignment.items()
                                    if class_label in classes]
                samples_per_client = len(train_indices) // len(clients_with_class)

                client_idx_in_group = clients_with_class.index(client_id)
                start_idx = client_idx_in_group * samples_per_client
                end_idx = min((client_idx_in_group + 1) * samples_per_client, len(train_indices))

                if start_idx < len(train_indices):
                    client_train_samples = train_indices[start_idx:end_idx]
                    client_distributions[client_key]['train'].extend(client_train_samples)

                # Test samples
                test_samples_per_client = len(test_indices) // len(clients_with_class)
                test_start = client_idx_in_group * test_samples_per_client
                test_end = min((client_idx_in_group + 1) * test_samples_per_client, len(test_indices))

                if test_start < len(test_indices):
                    client_test_samples = test_indices[test_start:test_end]
                    client_distributions[client_key]['test'].extend(client_test_samples)

        # Add total samples count
        for client_key in client_distributions:
            train_count = len(client_distributions[client_key]['train'])
            test_count = len(client_distributions[client_key]['test'])
            client_distributions[client_key]['total_samples'] = train_count + test_count

        # Statistics
        self._print_distribution_stats(client_distributions, f"Pathological Non-IID ({classes_per_client} classes/client)")

        return client_distributions

    def create_quantity_skewed_distribution(self, skew_factor=2.0, train_ratio=0.8):
        """
        Create quantity-skewed distribution where clients have different amounts of data

        Parameters:
        - skew_factor: Factor controlling skewness
        - train_ratio: Ratio of training data

        Returns:
        - Dictionary with client data distributions
        """
        print(f"Creating Quantity-skewed distribution (skew factor={skew_factor})...")

        # Generate skewed sample sizes
        base_size = len(self.dataset) // self.num_clients
        sample_sizes = []

        for i in range(self.num_clients):
            # Create power-law distribution
            skew = np.power(i + 1, skew_factor)
            size = int(base_size * skew / np.mean([np.power(j + 1, skew_factor) for j in range(self.num_clients)]))
            sample_sizes.append(max(size, 50))  # Minimum 50 samples

        # Normalize to total dataset size
        total_allocated = sum(sample_sizes)
        if total_allocated > len(self.dataset):
            scale_factor = len(self.dataset) / total_allocated
            sample_sizes = [int(size * scale_factor) for size in sample_sizes]

        # Randomly distribute samples
        all_indices = list(range(len(self.dataset)))
        np.random.shuffle(all_indices)

        client_distributions = {}
        start_idx = 0

        for client_id in range(self.num_clients):
            end_idx = min(start_idx + sample_sizes[client_id], len(all_indices))
            client_indices = all_indices[start_idx:end_idx]

            # Split train/test
            train_size = int(len(client_indices) * train_ratio)
            train_indices = client_indices[:train_size]
            test_indices = client_indices[train_size:]

            client_key = f"client_{client_id}"
            client_distributions[client_key] = {
                'train': train_indices,
                'test': test_indices,
                'total_samples': len(client_indices)
            }

            start_idx = end_idx

        # Statistics
        self._print_distribution_stats(client_distributions, f"Quantity-skewed (factor={skew_factor})")

        return client_distributions

    def _ensure_minimum_samples(self, client_distributions, min_samples):
        """Ensure each client has minimum number of samples"""
        for client_key in client_distributions:
            train_count = len(client_distributions[client_key]['train'])
            test_count = len(client_distributions[client_key]['test'])
            total = train_count + test_count

            if total < min_samples:
                # Need to add more samples
                needed = min_samples - total

                # Find clients with excess samples
                for other_client in client_distributions:
                    if other_client != client_key:
                        other_total = (len(client_distributions[other_client]['train']) +
                                     len(client_distributions[other_client]['test']))

                        if other_total > min_samples + needed:
                            # Transfer some samples
                            transfer_from_train = min(needed // 2,
                                                    len(client_distributions[other_client]['train']) - min_samples//2)
                            transfer_from_test = min(needed - transfer_from_train,
                                                   len(client_distributions[other_client]['test']) - min_samples//2)

                            # Transfer train samples
                            if transfer_from_train > 0:
                                transferred = client_distributions[other_client]['train'][:transfer_from_train]
                                client_distributions[client_key]['train'].extend(transferred)
                                client_distributions[other_client]['train'] = client_distributions[other_client]['train'][transfer_from_train:]

                            # Transfer test samples
                            if transfer_from_test > 0:
                                transferred = client_distributions[other_client]['test'][:transfer_from_test]
                                client_distributions[client_key]['test'].extend(transferred)
                                client_distributions[other_client]['test'] = client_distributions[other_client]['test'][transfer_from_test:]

                            break

    def _print_distribution_stats(self, client_distributions, distribution_type):
        """Print statistics about the distribution"""
        print(f"\n{distribution_type} Distribution Statistics:")
        print("-" * 50)

        total_samples = 0
        sample_counts = []
        class_distributions = defaultdict(lambda: defaultdict(int))

        for client_key, data in client_distributions.items():
            train_count = len(data['train'])
            test_count = len(data['test'])
            total_client_samples = train_count + test_count

            sample_counts.append(total_client_samples)
            total_samples += total_client_samples

            # Count classes for this client
            for idx in data['train'] + data['test']:
                label = self.labels[idx]
                class_distributions[client_key][label] += 1

        # Overall statistics
        print(f"Total samples distributed: {total_samples}")
        print(f"Average samples per client: {np.mean(sample_counts):.1f}")
        print(f"Std dev of samples per client: {np.std(sample_counts):.1f}")
        print(f"Min samples per client: {min(sample_counts)}")
        print(f"Max samples per client: {max(sample_counts)}")

        # Class distribution analysis
        class_entropy = self._calculate_class_entropy(class_distributions)
        print(f"Average class entropy per client: {class_entropy:.3f}")
        print(f"Max possible entropy: {np.log(self.num_classes):.3f}")
        print(f"Normalized entropy: {class_entropy / np.log(self.num_classes):.3f}")

    def _calculate_class_entropy(self, class_distributions):
        """Calculate average entropy of class distributions across clients"""
        entropies = []

        for client_key, class_counts in class_distributions.items():
            total_samples = sum(class_counts.values())
            if total_samples == 0:
                continue

            probabilities = [count / total_samples for count in class_counts.values()]
            entropy = -sum(p * np.log(p) for p in probabilities if p > 0)
            entropies.append(entropy)

        return np.mean(entropies) if entropies else 0

    def visualize_distribution(self, client_distributions, save_path=None, max_clients_to_show=20):
        """
        Visualize the data distribution across clients

        Parameters:
        - client_distributions: Client distributions dictionary
        - save_path: Path to save the plot
        - max_clients_to_show: Maximum number of clients to show in visualization
        """
        # Prepare data for visualization
        clients_to_show = min(len(client_distributions), max_clients_to_show)
        client_keys = list(client_distributions.keys())[:clients_to_show]

        # Create class distribution matrix
        class_matrix = np.zeros((clients_to_show, self.num_classes))

        for i, client_key in enumerate(client_keys):
            data = client_distributions[client_key]
            all_indices = data['train'] + data['test']

            for idx in all_indices:
                label = self.labels[idx]
                class_matrix[i, label] += 1

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Heatmap of class distribution
        sns.heatmap(class_matrix,
                   xticklabels=[f"Class {i}" for i in range(self.num_classes)],
                   yticklabels=[f"Client {i}" for i in range(clients_to_show)],
                   annot=True, fmt='.0f', cmap='YlOrRd', ax=ax1)
        ax1.set_title('Class Distribution Across Clients')
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Clients')

        # Sample count distribution
        sample_counts = [client_distributions[key]['total_samples'] for key in client_keys]
        ax2.bar(range(clients_to_show), sample_counts)
        ax2.set_title('Total Samples per Client')
        ax2.set_xlabel('Client ID')
        ax2.set_ylabel('Number of Samples')
        ax2.set_xticks(range(clients_to_show))
        ax2.set_xticklabels([f"C{i}" for i in range(clients_to_show)], rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution visualization saved to {save_path}")

        plt.show()

    def create_federated_datasets(self, client_distributions):
        """
        Create federated datasets for each client

        Parameters:
        - client_distributions: Client distributions dictionary

        Returns:
        - Dictionary of client datasets
        """
        federated_datasets = {}

        for client_key, data in client_distributions.items():
            train_subset = Subset(self.dataset, data['train'])
            test_subset = Subset(self.dataset, data['test'])

            federated_datasets[client_key] = {
                'train': train_subset,
                'test': test_subset,
                'train_indices': data['train'],
                'test_indices': data['test']
            }

        return federated_datasets

    def get_global_test_dataset(self, test_ratio=0.2):
        """
        Create a global test dataset

        Parameters:
        - test_ratio: Ratio of data to use for global testing

        Returns:
        - Global test dataset
        """
        total_samples = len(self.dataset)
        test_size = int(test_ratio * total_samples)

        all_indices = list(range(total_samples))
        np.random.shuffle(all_indices)

        test_indices = all_indices[:test_size]
        global_test_dataset = Subset(self.dataset, test_indices)

        return global_test_dataset, test_indices