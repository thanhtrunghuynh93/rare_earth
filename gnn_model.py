import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================================
# PREPARE DATA FROM PREPROCESSING OUTPUT
# ============================================================================

def prepare_data_for_gnn(X, y, edge_index, split_data):
    """
    Prepare PyTorch Geometric Data object from preprocessing outputs.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Full feature matrix
    y : pd.Series or np.ndarray
        Full labels
    edge_index : np.ndarray
        Graph edge indices
    split_data : dict
        Dictionary containing train/test split information from create_spatial_split()
    
    Returns:
    --------
    data : torch_geometric.data.Data
        PyTorch Geometric Data object ready for training
    """
    
    print("Preparing data for GNN...")
    
    # Convert to numpy if needed
    if hasattr(X, 'values'):
        X_np = X.values
    else:
        X_np = X
    
    if hasattr(y, 'values'):
        y_np = y.values
    else:
        y_np = y
    
    # Convert to PyTorch tensors
    x = torch.FloatTensor(X_np)
    y_tensor = torch.LongTensor(y_np)
    edge_index_tensor = torch.LongTensor(edge_index)
    
    # Get masks from split_data
    train_mask = split_data['train_mask']
    test_mask = split_data['test_mask']
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=x,
        edge_index=edge_index_tensor,
        y=y_tensor,
        train_mask=train_mask,
        test_mask=test_mask
    )
    
    print(f"✓ Created PyG Data object")
    print(f"  - Nodes: {data.num_nodes}")
    print(f"  - Edges: {data.num_edges}")
    print(f"  - Features: {data.num_node_features}")
    print(f"  - Train nodes: {train_mask.sum().item()}")
    print(f"  - Test nodes: {test_mask.sum().item()}")
    
    return data


# ============================================================================
# USAGE: Call this function with your preprocessing outputs
# ============================================================================
# 
# Example:
# from preprocessing import create_spatial_split  # Your preprocessing script
# 
# # Run preprocessing
# split_data = create_spatial_split(df, X, y, edge_index, test_region='South and Central Asia')
# 
# # Prepare data for GNN
# data = prepare_data_for_gnn(X, y, edge_index, split_data)
# 
# # Now you can train the model
# model = GCN(...)
# model, history = train_model(model, data, ...)
# ============================================================================

# ============================================================================
# GNN MODEL ARCHITECTURES
# ============================================================================

class GCN(nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


class GAT(nn.Module):
    """Graph Attention Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, 
                 heads=4, dropout=0.5):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        
        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                     heads=heads, dropout=dropout))
        
        # Output layer
        self.convs.append(GATConv(hidden_channels * heads, out_channels, 
                                 heads=1, concat=False, dropout=dropout))
    
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


class GraphSAGE(nn.Module):
    """GraphSAGE Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


# ============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================

def train_epoch(model, data, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Calculate training metrics
    pred = out[data.train_mask].argmax(dim=1)
    correct = (pred == data.y[data.train_mask]).sum()
    acc = int(correct) / int(data.train_mask.sum())
    
    return loss.item(), acc


@torch.no_grad()
def evaluate(model, data, mask, device):
    """Evaluate model on given mask"""
    model.eval()
    out = model(data.x, data.edge_index)
    
    # Predictions
    pred = out[mask].argmax(dim=1)
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred.cpu().numpy()
    y_prob = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def train_model(model, data, epochs=200, lr=0.01, weight_decay=5e-4, 
                patience=20, device='cpu', verbose=True):
    """
    Train GNN model with early stopping
    
    Parameters:
    -----------
    model : nn.Module
        GNN model to train
    data : Data
        PyTorch Geometric Data object
    epochs : int
        Maximum number of epochs
    lr : float
        Learning rate
    weight_decay : float
        L2 regularization
    patience : int
        Early stopping patience
    device : str
        Device to train on ('cpu' or 'cuda')
    verbose : bool
        Whether to print progress
    """
    
    # Move to device
    model = model.to(device)
    data = data.to(device)
    
    # Calculate class weights for imbalanced data
    y_train = data.y[data.train_mask].cpu().numpy()
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor([1.0 / count for count in class_counts]).to(device)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'test_f1': [],
        'test_auc': []
    }
    
    best_test_f1 = 0
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    if verbose:
        print(f"\n{'='*70}")
        print("TRAINING GNN MODEL")
        print(f"{'='*70}\n")
        iterator = tqdm(range(epochs), desc="Training")
    else:
        iterator = range(epochs)
    
    for epoch in iterator:
        # Train
        train_loss, train_acc = train_epoch(model, data, optimizer, criterion, device)
        
        # Evaluate
        test_metrics = evaluate(model, data, data.test_mask, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_metrics['accuracy'])
        history['test_f1'].append(test_metrics['f1'])
        history['test_auc'].append(test_metrics['auc'])
        
        # Early stopping based on test F1
        if test_metrics['f1'] > best_test_f1:
            best_test_f1 = test_metrics['f1']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            iterator.set_postfix({
                'Loss': f"{train_loss:.4f}",
                'Train Acc': f"{train_acc:.4f}",
                'Test F1': f"{test_metrics['f1']:.4f}",
                'Test AUC': f"{test_metrics['auc']:.4f}"
            })
        
        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def print_evaluation_report(model, data, device='cpu'):
    """Print comprehensive evaluation report"""
    
    print(f"\n{'='*70}")
    print("FINAL EVALUATION RESULTS")
    print(f"{'='*70}\n")
    
    # Evaluate on train and test
    train_metrics = evaluate(model, data, data.train_mask, device)
    test_metrics = evaluate(model, data, data.test_mask, device)
    
    # Print metrics
    print("TRAIN SET:")
    print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall:    {train_metrics['recall']:.4f}")
    print(f"  F1 Score:  {train_metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {train_metrics['auc']:.4f}")
    
    print("\nTEST SET:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {test_metrics['auc']:.4f}")
    
    # Confusion matrix
    print("\nCONFUSION MATRIX (Test Set):")
    cm = confusion_matrix(test_metrics['y_true'], test_metrics['y_pred'])
    print(cm)
    print(f"\nTN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    # Classification report
    print("\nCLASSIFICATION REPORT (Test Set):")
    print(classification_report(test_metrics['y_true'], test_metrics['y_pred'], 
                                target_names=['No REE', 'Has REE']))
    
    return train_metrics, test_metrics


def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'])
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['test_acc'], label='Test')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(history['test_f1'])
    axes[1, 0].set_title('Test F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 1].plot(history['test_auc'])
    axes[1, 1].set_title('Test AUC-ROC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved training history plot to {save_path}")
    plt.close()


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def train_gnn_model(X, y, edge_index, split_data, 
                    model_type='GCN',
                    hidden_channels=64,
                    num_layers=3,
                    dropout=0.5,
                    lr=0.01,
                    weight_decay=5e-4,
                    epochs=200,
                    patience=20,
                    device=None,
                    verbose=True):
    """
    Complete training pipeline for GNN model.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Full feature matrix
    y : pd.Series or np.ndarray
        Full labels
    edge_index : np.ndarray
        Graph edge indices
    split_data : dict
        Dictionary from create_spatial_split() containing train/test masks
    model_type : str
        'GCN', 'GAT', or 'GraphSAGE'
    hidden_channels : int
        Number of hidden units
    num_layers : int
        Number of GNN layers
    dropout : float
        Dropout rate
    lr : float
        Learning rate
    weight_decay : float
        L2 regularization
    epochs : int
        Maximum training epochs
    patience : int
        Early stopping patience
    device : str
        'cpu' or 'cuda' (auto-detected if None)
    verbose : bool
        Print progress
    
    Returns:
    --------
    model : trained GNN model
    history : training history dict
    metrics : dict with train/test metrics
    """
    
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print(f"\nUsing device: {device}")
        print(f"\n{'='*70}")
        print(f"MODEL CONFIGURATION")
        print(f"{'='*70}")
        print(f"Architecture: {model_type}")
        print(f"Hidden channels: {hidden_channels}")
        print(f"Number of layers: {num_layers}")
        print(f"Dropout: {dropout}")
        print(f"Learning rate: {lr}")
        print(f"Weight decay: {weight_decay}")
        print(f"Max epochs: {epochs}")
        print(f"Early stopping patience: {patience}")
    
    # Prepare data
    data = prepare_data_for_gnn(X, y, edge_index, split_data)
    
    # Initialize model
    in_channels = data.num_node_features
    out_channels = 2  # Binary classification
    
    if model_type == 'GCN':
        model = GCN(in_channels, hidden_channels, out_channels, num_layers, dropout)
    elif model_type == 'GAT':
        model = GAT(in_channels, hidden_channels, out_channels, num_layers, heads=4, dropout=dropout)
    elif model_type == 'GraphSAGE':
        model = GraphSAGE(in_channels, hidden_channels, out_channels, num_layers, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'GCN', 'GAT', or 'GraphSAGE'")
    
    if verbose:
        print(f"\n✓ Initialized {model_type} model")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    model, history = train_model(
        model, data,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        device=device,
        verbose=verbose
    )
    
    # Evaluate model
    train_metrics, test_metrics = print_evaluation_report(model, data, device)
    
    # Plot training history
    plot_training_history(history)
    
    # Prepare return dict
    metrics = {
        'train': train_metrics,
        'test': test_metrics
    }
    
    return model, history, metrics


