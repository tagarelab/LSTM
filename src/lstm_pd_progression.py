#!/usr/bin/env python3

"""
LSTM-Powered Progression Index for Parkinson's Disease Monitoring

This script implements a Long Short-Term Memory (LSTM) neural network to 
create a progression index for Parkinson's disease (PD) monitoring using
wearable sensor data. The model transforms daily ambulatory activity 
patterns into a meaningful disease progression metric. 

Author: Felice Dong 
Advisor: Dr. Hemant Tagare 
Department: Statistics and Data Science, Yale University
Date: June 5, 2025

Usage: 
python lstm_pd_progression.py [--data_path DATA_PATH] [--output_dir OUTPUT_DIR] 
                              [--num_splits NUM_SPLITS] [--seed SEED] 

Dependencies: 
- numpy
- pandas
- torch
- matplotlib
- seaborn
- scipy
- sklearn
"""

# Import necessary libraries
import argparse
import logging 
import os 
import sys 
from typing import List, Tuple, Dict, Any 
import warnings 
import json

import pandas as pd
import numpy as np
import seaborn as sns
import torch 
from torch import nn, optim
import matplotlib.pyplot as plt
from scipy import stats

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.FileHandler('lstm_pd_progression.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for model parameters and data processing settings."""
    
    # Data processing parameters
    MIN_CONSECUTIVE_WEEKS = 26
    MIN_DURATION_DAYS = 30
    MAX_INTERPOLATION_GAP = 2
    
    # Model parameters
    INPUT_SIZE = 7
    HIDDEN_SIZE = 20
    OUTPUT_SIZE = 7
    NUM_LAYERS = 2
    
    # Training parameters
    MAX_EPOCHS = 20000
    BATCH_SIZE = 15
    LEARNING_RATE = 0.0003
    NOISE_SIGMA = 0.02
    SWAP_FRACTION = 0.2
    CONVERGENCE_WINDOW = 5
    CONVERGENCE_THRESHOLD = 0.0001
    
    # Evaluation parameters
    DEFAULT_NUM_SPLITS = 20
    TEST_SUBJECTS_PER_COHORT = 2


def setup_reproducibility(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across numpy, torch, and CUDA.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


class DataProcessor:
    """Handles data loading, preprocessing, and feature engineering."""
    
    def __init__(self, config: Config):
        self.config = config
        
    def load_and_filter_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data and apply initial filtering criteria.
        
        Args:
            data_path: Path to the CSV data file
            
        Returns:
            Filtered DataFrame
        """
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        
        # Drop unnecessary columns
        if 'week_id' in data.columns:
            data = data.drop(columns=['week_id'])
            
        # Filter for subjects with sufficient data
        user_counts = data.groupby('subject').size()
        data = data[data['subject'].isin(
            user_counts[user_counts >= self.config.MIN_CONSECUTIVE_WEEKS].index
        )]
        
        # Keep only PD and HC cohorts
        data = data[data['cohort'].isin(['PD', 'Control'])]
        
        logger.info(f"Data shape after filtering: {data.shape}")
        cohort_counts = data.groupby('cohort')['subject'].nunique()
        logger.info(f"Cohort distribution: {cohort_counts.to_dict()}")
        
        return data
        
    def create_continuous_weeks(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create continuous week numbering for each subject.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with continuous week numbers
        """
        logger.info("Creating continuous week numbering")
        
        for subject, subject_data in data.groupby('subject'):
            first_year = subject_data['year'].min()
            subject_weeks = (subject_data['year'] - first_year) * 52 + subject_data['week_num']
            data.loc[subject_data.index, 'week_num'] = subject_weeks
            data.loc[subject_data.index, 'week_num'] -= subject_weeks.min() - 1
            
        return data.drop(columns=['year'])
    
    def get_consecutive_weeks(self, subject_data: pd.DataFrame) -> int:
        """
        Calculate maximum consecutive weeks with complete data for a subject.
        
        Args:
            subject_data: DataFrame for single subject
            
        Returns:
            Maximum number of consecutive complete weeks
        """
        day_cols = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        mask = subject_data[day_cols].notna().all(axis=1)
        
        consecutive_weeks = []
        current_streak = 0
        
        for i in range(len(mask)):
            if mask.iloc[i]:
                current_streak += 1
            else:
                if current_streak > 0:
                    consecutive_weeks.append(current_streak)
                current_streak = 0
                
        if current_streak > 0:
            consecutive_weeks.append(current_streak)
            
        return max(consecutive_weeks) if consecutive_weeks else 0
    
    def interpolate_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate missing values for small gaps in the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with interpolated values
        """
        logger.info("Interpolating missing data")
        result = data.copy()
        day_cols = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        for subject in data['subject'].unique():
            subject_mask = data['subject'] == subject
            subject_data = data[subject_mask].copy().sort_values('week_num')
            
            all_weeks = range(subject_data['week_num'].min(), 
                            subject_data['week_num'].max() + 1)
            
            # Create flat array for interpolation
            flat_values = []
            flat_indices = []
            
            for week in all_weeks:
                week_data = subject_data[subject_data['week_num'] == week]
                
                if len(week_data) > 0:
                    row = week_data.iloc[0]
                    for day in day_cols:
                        flat_values.append(row[day])
                        flat_indices.append((row.name, day))
                else:
                    for _ in day_cols:
                        flat_values.append(None)
                        flat_indices.append(None)
            
            # Interpolate small gaps
            flat_array = np.array(flat_values, dtype=float)
            
            for i in range(len(flat_array)):
                if np.isnan(flat_array[i]):
                    # Find gap boundaries
                    start = i
                    while start > 0 and np.isnan(flat_array[start-1]):
                        start -= 1
                    
                    end = i
                    while end < len(flat_array)-1 and np.isnan(flat_array[end+1]):
                        end += 1
                    
                    gap_size = end - start + 1
                    
                    # Interpolate if gap is small enough
                    if (gap_size <= self.config.MAX_INTERPOLATION_GAP and 
                        start > 0 and end < len(flat_array)-1):
                        
                        left_val = flat_array[start-1]
                        right_val = flat_array[end+1]
                        
                        if not np.isnan(left_val) and not np.isnan(right_val):
                            for j in range(start, end+1):
                                position = j - start + 1
                                flat_array[j] = (left_val + 
                                               (right_val - left_val) * 
                                               position / (gap_size + 1))
            
            # Map back to dataframe
            for i, idx_day in enumerate(flat_indices):
                if idx_day is not None:
                    idx, day = idx_day
                    result.loc[idx, day] = flat_array[i]
        
        return result
    
    def get_longest_consecutive_streak(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract longest consecutive streak for each subject.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with longest consecutive streaks
        """
        logger.info("Extracting longest consecutive streaks")
        results = []
        day_cols = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        for subject in data['subject'].unique():
            subject_data = data[data['subject'] == subject].copy().sort_values('week_num')
            mask = subject_data[day_cols].notna().all(axis=1)
            complete_weeks = subject_data.loc[mask, 'week_num'].values
            
            if len(complete_weeks) == 0:
                continue
            
            # Find consecutive sequences
            consecutive_streaks = []
            current_streak = []
            
            for i in range(len(complete_weeks)):
                if i == 0 or complete_weeks[i] == complete_weeks[i-1] + 1:
                    current_streak.append(complete_weeks[i])
                else:
                    if current_streak:
                        consecutive_streaks.append(current_streak)
                    current_streak = [complete_weeks[i]]
            
            if current_streak:
                consecutive_streaks.append(current_streak)
            
            # Get longest streak
            if consecutive_streaks:
                longest_streak = max(consecutive_streaks, key=len)
                
                if len(longest_streak) >= self.config.MIN_CONSECUTIVE_WEEKS:
                    target_weeks = longest_streak[:self.config.MIN_CONSECUTIVE_WEEKS]
                    streak_data = subject_data[subject_data['week_num'].isin(target_weeks)]
                    results.append(streak_data)
        
        if results:
            final_data = pd.concat(results)
            logger.info(f"Final dataset: {len(final_data['subject'].unique())} subjects")
            return final_data
        else:
            return pd.DataFrame()
    
    def apply_transformations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature transformations to create progression index.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Applying feature transformations")
        data_transformed = data.copy()
        
        # Align weeks to start from 0 for each subject
        for subject, subject_data in data_transformed.groupby('subject'):
            first_week = subject_data['week_num'].min()
            data_transformed.loc[subject_data.index, 'week_num'] = (
                subject_data['week_num'] - first_week
            )
        
        # Calculate global HC mean
        control_data = data[data['cohort'] == 'Control']
        day_columns = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        control_sum = control_data[day_columns].sum().sum()
        control_count = control_data[day_columns].notna().sum().sum()
        global_hc_mean = control_sum / control_count
        
        logger.info(f"Global HC mean: {global_hc_mean:.2f}")
        
        # Transform features: negative values + HC mean
        for day in day_columns:
            data_transformed[day] = -data_transformed[day] + global_hc_mean
        
        return data_transformed


class LSTMModel(nn.Module):
    """LSTM model for progression index estimation."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        self.lin = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the LSTM model.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, input_size)
            
        Returns:
            Tuple of (output, weights)
        """
        lstm_out, (hn, cn) = self.lstm(x)
        weight = self.lin(lstm_out)
        
        # Normalize weights
        epsilon = 1e-6
        weight_denom = torch.sum(torch.abs(weight), axis=2) + epsilon
        output = torch.sum(torch.mul(weight, x), axis=2) / weight_denom
        
        return output, weight


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def prepare_data(self, df: pd.DataFrame, sort_features: bool = True) -> Tuple[List, List, List, List]:
        """
        Convert DataFrame to sequences for model input.
        
        Args:
            df: Input DataFrame
            sort_features: Whether to sort features by activity level
            
        Returns:
            Tuple of (sequences, labels, time_points, subject_ids)
        """
        subjects = df['subject'].unique()
        all_sequences, all_labels, all_time_points, subject_ids = [], [], [], []
        
        for subject in subjects:
            subject_data = df[df['subject'] == subject].sort_values('week_num')
            features = subject_data[['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']].values
            
            if sort_features:
                # Sort features in descending order
                features = np.sort(features, axis=1)[:, ::-1]
            
            cohort_label = 1 if subject_data['cohort'].iloc[0] == 'PD' else 0
            
            all_sequences.append(features)
            all_labels.append(cohort_label)
            all_time_points.append(len(features))
            subject_ids.append(subject)
        
        return all_sequences, all_labels, all_time_points, subject_ids
    
    def create_dataloader(self, sequences: List, labels: List, time_points: List) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create padded sequences and convert to tensors.
        
        Args:
            sequences: List of sequence arrays
            labels: List of labels
            time_points: List of time points
            
        Returns:
            Tuple of (X, lengths, y, time_weights)
        """
        # Sort by sequence length (for consistency)
        sorted_idx = sorted(range(len(sequences)), key=lambda i: len(sequences[i]), reverse=True)
        sequences = [sequences[i] for i in sorted_idx]
        labels = [labels[i] for i in sorted_idx]
        time_points = [time_points[i] for i in sorted_idx]
        
        # Pad sequences
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = []
        
        for seq in sequences:
            padded_seq = np.zeros((max_len, seq.shape[1]))
            padded_seq[:len(seq)] = seq
            padded_sequences.append(padded_seq)
        
        # Convert to tensors
        X = torch.FloatTensor(padded_sequences).transpose(0, 1)  # (num_weeks, num_subjects, num_features)
        y = torch.FloatTensor(labels)
        lengths = torch.FloatTensor(time_points)
        time_weights = torch.FloatTensor(list(range(1, max_len + 1)))
        
        return X, lengths, y, time_weights
    
    def swap_data(self, x_data: torch.Tensor, swap_frac: float = 0.3) -> torch.Tensor:
        """Swap adjacent time points for data augmentation."""
        nt = np.arange(x_data.size(0))
        n_swap = int(np.floor(swap_frac * nt[-1]))
        
        for _ in range(n_swap):
            i = np.random.randint(1, len(nt))
            nt[i], nt[i-1] = nt[i-1], nt[i]
        
        return x_data[nt, :, :]
    
    def get_batch(self, x_data: torch.Tensor, lengths_data: torch.Tensor, 
                  y_data: torch.Tensor, batch_size: int = 10, 
                  sigma: float = 0.0, swap_frac: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get random batch with data augmentation."""
        n_sequences = x_data.size(1)
        batch_indices = torch.randperm(n_sequences)[:batch_size]
        
        x_batch = x_data[:, batch_indices, :]
        lengths_batch = lengths_data[batch_indices]
        y_batch = y_data[batch_indices]
        
        # Add noise and swap
        if sigma > 0:
            x_batch = x_batch + sigma * torch.normal(0, 1.0, size=x_batch.size())
        if swap_frac > 0:
            x_batch = self.swap_data(x_batch, swap_frac=swap_frac)
        
        return x_batch, lengths_batch, y_batch, batch_indices
    
    def loss_fn(self, output: torch.Tensor, weight: torch.Tensor, 
                y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Custom loss function for progression index learning."""
        y1_index = torch.nonzero(y > 0.5).flatten()  # PD indices
        y0_index = torch.nonzero(y <= 0.5).flatten()  # HC indices
        
        epsilon = 1e-6
        
        # PD term (time-weighted)
        y1_sum = torch.sum(output[:, y1_index], axis=1) / (torch.numel(y1_index) + epsilon)
        y1_sum = y1_sum * (1 + t)
        y1_sum = torch.sum(y1_sum)
        
        # HC term (squared)
        y0_sum = torch.sum((output[:, y0_index]) ** 2) / (torch.numel(y0_index) + epsilon)
        
        # Regularization
        w1_std = torch.std(weight[:, y1_index, :].flatten())
        w0_std = torch.std(weight[:, y0_index, :].flatten())
        
        loss = -y1_sum + y0_sum + 20.0 * (w1_std + w0_std)
        return loss


class ModelEvaluator:
    """Handles model evaluation and cross-validation."""
    
    def __init__(self, config: Config, trainer: ModelTrainer):
        self.config = config
        self.trainer = trainer
    
    def evaluate_model(self, sequences: List, labels: List, time_points: List, 
                      subject_ids: List, num_outer_loops: int = 20, 
                      sort_features: bool = True) -> Dict[str, Any]:
        """
        Evaluate model using cross-validation.
        
        Args:
            sequences: List of sequences
            labels: List of labels  
            time_points: List of time points
            subject_ids: List of subject IDs
            num_outer_loops: Number of CV splits
            sort_features: Whether to sort features
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Starting evaluation with {num_outer_loops} splits")
        
        labels_np = np.array(labels)
        pd_indices = np.where(labels_np == 1)[0]
        hc_indices = np.where(labels_np == 0)[0]
        
        logger.info(f"PD subjects: {len(pd_indices)}, HC subjects: {len(hc_indices)}")
        
        all_pd_outputs, all_hc_outputs, all_weights, all_pvalues = [], [], [], []
        
        for outer_loop in range(num_outer_loops):
            logger.info(f"Split {outer_loop + 1}/{num_outer_loops}")
            
            # Random test split
            test_pd_indices = np.random.choice(pd_indices, size=self.config.TEST_SUBJECTS_PER_COHORT, replace=False)
            test_hc_indices = np.random.choice(hc_indices, size=self.config.TEST_SUBJECTS_PER_COHORT, replace=False)
            test_indices = np.concatenate([test_pd_indices, test_hc_indices])
            train_indices = np.array([i for i in range(len(sequences)) if i not in test_indices])
            
            # Create datasets
            train_sequences = [sequences[i] for i in train_indices]
            train_labels = [labels[i] for i in train_indices]
            train_time_points = [time_points[i] for i in train_indices]
            
            test_sequences = [sequences[i] for i in test_indices]
            test_labels = [labels[i] for i in test_indices]
            test_time_points = [time_points[i] for i in test_indices]
            
            # Convert to tensors
            X_train, lengths_train, y_train, time_weights = self.trainer.create_dataloader(
                train_sequences, train_labels, train_time_points)
            X_test, lengths_test, y_test, _ = self.trainer.create_dataloader(
                test_sequences, test_labels, test_time_points)
            
            # Train model
            model = LSTMModel(self.config.INPUT_SIZE, self.config.HIDDEN_SIZE,
                            self.config.OUTPUT_SIZE, self.config.NUM_LAYERS)
            model = self._train_model(model, X_train, y_train, time_weights)
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                test_output, test_weight = model(X_test)
                
                test_output_np = test_output.cpu().numpy()
                test_weight_np = test_weight.cpu().numpy()
                test_labels_np = y_test.cpu().numpy()
                
                test_pd_indices = np.where(test_labels_np == 1)[0]
                test_hc_indices = np.where(test_labels_np == 0)[0]
                
                pd_outputs = test_output_np[:, test_pd_indices]
                hc_outputs = test_output_np[:, test_hc_indices]
                
                all_pd_outputs.append(pd_outputs)
                all_hc_outputs.append(hc_outputs)
                all_weights.append(test_weight_np)
                
                # Calculate p-values
                pvalues = []
                for t in range(test_output_np.shape[0]):
                    _, p_value = stats.ttest_ind(
                        pd_outputs[t, :], hc_outputs[t, :], equal_var=False)
                    pvalues.append(p_value)
                all_pvalues.append(pvalues)
        
        # Aggregate results
        max_weeks = max(arr.shape[0] for arr in all_pd_outputs)
        all_pvalues_padded = []
        for pvals in all_pvalues:
            padded = np.pad(pvals, (0, max_weeks - len(pvals)), 
                          mode='constant', constant_values=np.nan)
            all_pvalues_padded.append(padded)
        
        avg_pvalues = np.nanmean(all_pvalues_padded, axis=0)
        
        # Find significant windows
        sig_windows = []
        in_window = False
        start_week = 0
        
        for week, p in enumerate(avg_pvalues):
            if p < 0.05 and not in_window:
                in_window = True
                start_week = week
            elif p >= 0.05 and in_window:
                in_window = False
                sig_windows.append((start_week, week - 1))
        
        if in_window:
            sig_windows.append((start_week, len(avg_pvalues) - 1))
        
        logger.info(f"Significant windows (p < 0.05): {sig_windows}")
        
        return {
            'pd_outputs': all_pd_outputs,
            'hc_outputs': all_hc_outputs,
            'weights': all_weights,
            'pvalues': all_pvalues,
            'avg_pvalues': avg_pvalues,
            'sig_windows': sig_windows
        }
    
    def _train_model(self, model: LSTMModel, X_train: torch.Tensor, 
                    y_train: torch.Tensor, time_weights: torch.Tensor) -> LSTMModel:
        """Train a single model instance."""
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        losses = []
        
        model.train()
        
        for epoch in range(self.config.MAX_EPOCHS):
            # Get balanced batch
            train_pd_indices = np.where(y_train.numpy() == 1)[0]
            train_hc_indices = np.where(y_train.numpy() == 0)[0]
            
            batch_pd_size = min(self.config.BATCH_SIZE, len(train_pd_indices))
            batch_hc_size = min(self.config.BATCH_SIZE, len(train_hc_indices))
            
            pd_batch_indices = torch.tensor(np.random.choice(
                train_pd_indices, size=batch_pd_size, replace=False))
            hc_batch_indices = torch.tensor(np.random.choice(
                train_hc_indices, size=batch_hc_size, replace=False))
            
            batch_indices = torch.cat([pd_batch_indices, hc_batch_indices])
            x_batch = X_train[:, batch_indices, :]
            y_batch = y_train[batch_indices]
            
            # Data augmentation
            x_batch = x_batch + self.config.NOISE_SIGMA * torch.normal(0, 1.0, size=x_batch.size())
            if self.config.SWAP_FRACTION > 0:
                x_batch = self.trainer.swap_data(x_batch, swap_frac=self.config.SWAP_FRACTION)
            
            # Training step
            optimizer.zero_grad()
            output, weight = model(x_batch)
            loss = self.trainer.loss_fn(output, weight, y_batch, time_weights)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
            # Check convergence
            if epoch > self.config.CONVERGENCE_WINDOW and epoch % 10 == 0:
                recent_losses = losses[-self.config.CONVERGENCE_WINDOW:]
                avg_loss = np.mean(recent_losses)
                prev_avg_loss = np.mean(losses[-2 * self.config.CONVERGENCE_WINDOW:-self.config.CONVERGENCE_WINDOW])
                percent_change = abs((avg_loss - prev_avg_loss) / prev_avg_loss)
                
                if percent_change < self.config.CONVERGENCE_THRESHOLD:
                    logger.debug(f"Converged at epoch {epoch + 1}")
                    break
        
        return model


class Visualizer:
    """Handles result visualization and plotting."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_pvalues(self, avg_pvalues: np.ndarray, title_suffix: str = "") -> None:
        """Plot average p-values over time."""
        plt.figure(figsize=(12, 4))
        plt.plot(avg_pvalues, 'k-', linewidth=2)
        plt.axhline(y=0.05, color='r', linestyle='--', label='p=0.05')
        plt.xlabel('Weeks')
        plt.ylabel('Average p-value')
        plt.title(f'Average P-value between PD and HC{title_suffix}')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'pvalues{title_suffix.lower().replace(" ", "_")}.png'), dpi=300)
        plt.show()
    
    def plot_progression_indices(self, all_pd_outputs: List, all_hc_outputs: List, 
                               title_suffix: str = "") -> None:
        """Plot individual progression indices."""
        plt.figure(figsize=(12, 6))
        
        # Plot first line with labels
        if len(all_pd_outputs) > 0 and all_pd_outputs[0].shape[1] > 0:
            plt.plot(all_pd_outputs[0][:, 0], 'r-', alpha=0.5, label='PD')
        if len(all_hc_outputs) > 0 and all_hc_outputs[0].shape[1] > 0:
            plt.plot(all_hc_outputs[0][:, 0], 'b-', alpha=0.5, label='HC')
        
        # Plot remaining lines
        for i in range(len(all_pd_outputs)):
            for j in range(all_pd_outputs[i].shape[1]):
                if i == 0 and j == 0:
                    continue
                plt.plot(all_pd_outputs[i][:, j], 'r-', alpha=0.5)
        
        for i in range(len(all_hc_outputs)):
            for j in range(all_hc_outputs[i].shape[1]):
                if i == 0 and j == 0:
                    continue
                plt.plot(all_hc_outputs[i][:, j], 'b-', alpha=0.5)
        
        plt.legend()
        plt.xlabel('Weeks')
        plt.ylabel('Progression Index')
        plt.title(f'Progression Index for All Test Subjects{title_suffix}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'individual_indices{title_suffix.lower().replace(" ", "_")}.png'), dpi=300)
        plt.show()
    
    def plot_mean_progression_indices(self, all_pd_outputs: List, all_hc_outputs: List,
                                    title_suffix: str = "") -> None:
        """Plot mean progression indices with standard deviation bands."""
        # Reshape and calculate statistics
        pd_outputs_flat = np.array(all_pd_outputs).reshape(-1, all_pd_outputs[0].shape[0]).T
        hc_outputs_flat = np.array(all_hc_outputs).reshape(-1, all_hc_outputs[0].shape[0]).T
        
        mean_pd = np.mean(pd_outputs_flat, axis=1)
        mean_hc = np.mean(hc_outputs_flat, axis=1)
        std_pd = np.std(pd_outputs_flat, axis=1)
        std_hc = np.std(hc_outputs_flat, axis=1)
        
        plt.figure(figsize=(12, 6))
        plt.plot(mean_pd, color='red', label='PD')
        plt.fill_between(range(len(mean_pd)), 
                        mean_pd - std_pd, mean_pd + std_pd, 
                        color='red', alpha=0.1)
        plt.plot(mean_hc, color='blue', label='HC')
        plt.fill_between(range(len(mean_hc)), 
                        mean_hc - std_hc, mean_hc + std_hc, 
                        color='blue', alpha=0.1)
        plt.xlabel('Weeks')
        plt.ylabel('Progression Index')
        plt.title(f'Mean Progression Index for PD and HC Subjects{title_suffix}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'mean_indices{title_suffix.lower().replace(" ", "_")}.png'), dpi=300)
        plt.show()
    
    def plot_feature_weights(self, all_weights: List, features: List[str], 
                           title_suffix: str = "") -> None:
        """Plot feature weights over time."""
        avg_weights = np.mean(all_weights, axis=(0, 2))  # Average across splits and subjects
        
        plt.figure(figsize=(12, 6))
        
        # Plot weights for specific weeks
        for i in range(5, min(30, avg_weights.shape[0]), 5):
            tmp = avg_weights[i, :]
            tmp = tmp / np.sum(np.abs(tmp))  # Normalize
            plt.plot(tmp, color=plt.cm.Blues(i / 30), label=f'Week {i+1}')
        
        plt.xlabel(f'Features ({", ".join(features)})')
        plt.ylabel('Normalized Weight')
        plt.title(f'Average Learned Weights Across Splits{title_suffix}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'feature_weights{title_suffix.lower().replace(" ", "_")}.png'), dpi=300)
        plt.show()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='LSTM-based Parkinson\'s Disease Progression Index')
    parser.add_argument('--data_path', type=str, default='Data_preimp.csv',
                       help='Path to input data CSV file')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save results and plots')
    parser.add_argument('--num_splits', type=int, default=Config.DEFAULT_NUM_SPLITS,
                       help='Number of cross-validation splits')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Setup
    setup_reproducibility(args.seed)
    config = Config()
    
    # Initialize components
    data_processor = DataProcessor(config)
    trainer = ModelTrainer(config)
    evaluator = ModelEvaluator(config, trainer)
    visualizer = Visualizer(args.output_dir)
    
    try:
        # Data processing pipeline
        logger.info("Starting data processing pipeline")
        
        # Load and filter data
        data = data_processor.load_and_filter_data(args.data_path)
        
        # Create continuous weeks
        data = data_processor.create_continuous_weeks(data)
        
        # Add missing days column for tracking
        data['missing_days'] = data[['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']].isna().sum(axis=1)
        
        # Interpolate missing data
        data_interpolated = data_processor.interpolate_missing_data(data)
        
        # Get longest consecutive streaks
        data_consecutive = data_processor.get_longest_consecutive_streak(data_interpolated)
        
        if data_consecutive.empty:
            logger.error("No subjects with sufficient consecutive data found")
            return
        
        # Apply transformations
        data_final = data_processor.apply_transformations(data_consecutive)
        
        # Log final data statistics
        cohort_counts = data_final.groupby('cohort')['subject'].nunique()
        logger.info(f"Final cohort distribution: {cohort_counts.to_dict()}")

        # Run both analyses
        for sort_features in [False, True]:
            feature_type = "sorted" if sort_features else "weekday"
            output_subdir = os.path.join(args.output_dir, feature_type)

            logger.info(f"Running analysis with {feature_type} features")

            # Create output directory for this feature type
            os.makedirs(output_subdir, exist_ok=True)
            visualizer_sub = Visualizer(output_subdir)

            # Prepare data for modeling
            logger.info("Preparing data for modeling")
            sequences, labels, time_points, subject_ids = trainer.prepare_data(
                data_final, sort_features=sort_features)
            
            # Model evaluation
            logger.info("Starting model evaluation")
            results = evaluator.evaluate_model(
                sequences, labels, time_points, subject_ids, 
                num_outer_loops=args.num_splits, 
                sort_features=sort_features)
            
            # Generate visualizations
            logger.info("Generating visualizations")
            feature_suffix = " (Sorted Features)" if sort_features else " (Weekday Features)"
            visualizer_sub.plot_pvalues(results['avg_pvalues'], feature_suffix)
            visualizer_sub.plot_progression_indices(
                results['pd_outputs'], results['hc_outputs'], feature_suffix)
            visualizer_sub.plot_mean_progression_indices(
                results['pd_outputs'], results['hc_outputs'], feature_suffix)
            
            # Feature names for plotting
            if sort_features:
                feature_names = ['Most Active', '2nd Most', '3rd Most', '4th Most', 
                               '5th Most', '6th Most', 'Least Active']
            else:
                feature_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

            visualizer_sub.plot_feature_weights(results['weights'], feature_names, feature_suffix)

            # Save results summary
            results_summary = {
                'num_subjects': len(subject_ids),
                'num_pd': sum(labels),
                'num_hc': len(labels) - sum(labels),
                'num_splits': args.num_splits,
                'significant_windows': results['sig_windows'],
                'min_pvalue': float(np.min(results['avg_pvalues'])),
                'feature_type': feature_type
            }

            with open(os.path.join(output_subdir, 'results_summary.json'), 'w') as f:
                json.dump(results_summary, f, indent=2)

        # Final logging
        logger.info("All analyses completed successfully")
        logger.info(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()