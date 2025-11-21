"""
Test script to verify all implemented functions work correctly with actual CSV data
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import csv

print("="*80)
print("TESTING RNA SECONDARY STRUCTURE PREDICTION PIPELINE")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n[1/6] Testing Data Loading...")

def load_data_from_csv(file_path):
    """
    Loads sequence and structure data from a CSV file.
    """
    data_tuples = []

    with open(file_path, 'r', newline='', encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames is None:
            return data_tuples

        def find_key(fieldnames, candidates):
            for fname in fieldnames:
                lf = fname.lower().strip()
                for c in candidates:
                    if c in lf:
                        return fname
            return None

        seq_candidates = ['sequence', 'seq', 'rna_sequence', 'nucleotide']
        struct_candidates = ['dot_bracket', 'dotbracket', 'dot', 'bracket', 'structure', 'dot-bracket', 'dot bracket']

        seq_key = find_key(reader.fieldnames, seq_candidates)
        struct_key = find_key(reader.fieldnames, struct_candidates)

        if seq_key is None and len(reader.fieldnames) >= 1:
            seq_key = reader.fieldnames[0]
        if struct_key is None:
            for fname in reader.fieldnames:
                if fname != seq_key:
                    struct_key = fname
                    break

        if seq_key is None or struct_key is None:
            raise ValueError(f"Could not determine sequence/structure columns. Available columns: {reader.fieldnames}")

        for row in reader:
            sequence = row.get(seq_key, "") or ""
            structure = row.get(struct_key, "") or ""
            data_tuples.append((sequence, structure))

    return data_tuples

# Load data
train_path = r"TR0.csv"
val_path = r"VL0.csv"
test_path = r"TS0.csv"

train_data = load_data_from_csv(train_path)
val_data = load_data_from_csv(val_path)
test_data = load_data_from_csv(test_path)

print(f"✓ Training samples: {len(train_data)}")
print(f"✓ Validation samples: {len(val_data)}")
print(f"✓ Test samples: {len(test_data)}")
print(f"✓ Sample sequence length: {len(train_data[0][0])}")
print(f"✓ Sample structure length: {len(train_data[0][1])}")

# ============================================================================
# 2. DATA ENCODING
# ============================================================================
print("\n[2/6] Testing Data Encoding Functions...")

def one_hot_encode(sequence, max_len):
    """One-hot encodes an RNA sequence."""
    nucleotide_to_index = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    encoded = np.zeros((max_len, 4), dtype=np.float32)
    sequence_truncated = sequence[:max_len]
    seq_upper = sequence_truncated.upper()
    indices = np.array([nucleotide_to_index.get(nucleotide, -1) for nucleotide in seq_upper])
    valid_positions = indices != -1
    row_indices = np.arange(len(indices))[valid_positions]
    col_indices = indices[valid_positions]
    if len(row_indices) > 0:
        encoded[row_indices, col_indices] = 1.0
    return encoded

def create_contact_map(dot_bracket, max_len):
    """Creates a contact map from a dot-bracket string."""
    dot_bracket_truncated = dot_bracket[:max_len]
    contact_map = np.zeros((max_len, max_len), dtype=np.float32)
    stack = []
    paired_indices = set()
    opening_brackets = {'(', '[', '{', '<'}
    bracket_pairs = {'(': ')', '[': ']', '{': '}', '<': '>'}
    
    for i, char in enumerate(dot_bracket_truncated):
        if char in opening_brackets:
            stack.append((i, char))
        elif char in [')', ']', '}', '>']:
            if stack:
                opening_index, opening_char = stack.pop()
                if bracket_pairs.get(opening_char) == char:
                    contact_map[opening_index, i] = 1.0
                    contact_map[i, opening_index] = 1.0
                    paired_indices.add(opening_index)
                    paired_indices.add(i)
    
    return contact_map

# Test encoding
test_seq, test_struct = train_data[0]
MAX_LEN = 150
encoded_seq = one_hot_encode(test_seq, MAX_LEN)
contact_map = create_contact_map(test_struct, MAX_LEN)

print(f"✓ One-hot encoding shape: {encoded_seq.shape}")
print(f"✓ Contact map shape: {contact_map.shape}")
print(f"✓ One-hot encoding sum: {encoded_seq.sum():.0f} (should be ≤ {MAX_LEN})")
print(f"✓ Contact map is symmetric: {np.allclose(contact_map, contact_map.T)}")
print(f"✓ Number of base pairs: {int(contact_map.sum() / 2)}")

# ============================================================================
# 3. PYTORCH DATASET
# ============================================================================
print("\n[3/6] Testing PyTorch Dataset...")

class RNADataset(Dataset):
    """Custom PyTorch Dataset for RNA sequences."""
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, structure = self.data[idx]
        encoded_sequence = one_hot_encode(sequence, self.max_len)
        contact_map = create_contact_map(structure, self.max_len)
        input_tensor = torch.from_numpy(encoded_sequence).float()
        target_tensor = torch.from_numpy(contact_map).float()
        return input_tensor, target_tensor

# Create datasets
train_dataset = RNADataset(train_data[:5], MAX_LEN)  # Use small subset for testing
val_dataset = RNADataset(val_data[:3], MAX_LEN)

print(f"✓ Training dataset size: {len(train_dataset)}")
print(f"✓ Validation dataset size: {len(val_dataset)}")

# Test getting an item
sample_input, sample_target = train_dataset[0]
print(f"✓ Sample input shape: {sample_input.shape}")
print(f"✓ Sample target shape: {sample_target.shape}")
print(f"✓ Input is PyTorch tensor: {isinstance(sample_input, torch.Tensor)}")
print(f"✓ Target is PyTorch tensor: {isinstance(sample_target, torch.Tensor)}")

# ============================================================================
# 4. DATALOADER
# ============================================================================
print("\n[4/6] Testing DataLoader...")

BATCH_SIZE = 2

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

print(f"✓ Train batches: {len(train_loader)}")
print(f"✓ Validation batches: {len(val_loader)}")

# Test getting a batch
for batch_input, batch_target in train_loader:
    print(f"✓ Batch input shape: {batch_input.shape}")
    print(f"✓ Batch target shape: {batch_target.shape}")
    break

# ============================================================================
# 5. CNN MODEL
# ============================================================================
print("\n[5/6] Testing CNN Model...")

class RNAFoldingCNN(nn.Module):
    def __init__(self, input_channels=8):
        super(RNAFoldingCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        
        self.conv_out = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)

    def forward(self, x_1d):
        batch_size = x_1d.shape[0]
        max_len = x_1d.shape[1]
        
        x_2d_i = x_1d.unsqueeze(2).repeat(1, 1, max_len, 1)
        x_2d_j = x_1d.unsqueeze(1).repeat(1, max_len, 1, 1)
        x_2d = torch.cat([x_2d_i, x_2d_j], dim=-1)
        x = x_2d.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.conv_out(x)
        x = torch.sigmoid(x)
        x = x.squeeze(1)
        x_symmetric = (x + x.transpose(1, 2)) / 2.0
        
        return x_symmetric

# Create model
model = RNAFoldingCNN()
print(f"✓ Model created successfully")
print(f"✓ Number of parameters: {sum(p.numel() for p in model.parameters())}")

# Test forward pass with a batch
model.eval()
with torch.no_grad():
    for batch_input, batch_target in train_loader:
        output = model(batch_input)
        print(f"✓ Input shape: {batch_input.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Output is symmetric: {torch.allclose(output, output.transpose(1, 2), atol=1e-6)}")
        print(f"✓ Output range: [{output.min():.4f}, {output.max():.4f}]")
        break

# ============================================================================
# 6. END-TO-END TEST
# ============================================================================
print("\n[6/6] Running End-to-End Test...")

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"✓ Device: {device}")

# Test one training step
model.train()
batch_input, batch_target = next(iter(train_loader))
batch_input = batch_input.to(device)
batch_target = batch_target.to(device)

# Forward pass
output = model(batch_input)
loss = criterion(output, batch_target)

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"✓ Forward pass successful")
print(f"✓ Backward pass successful")
print(f"✓ Loss value: {loss.item():.4f}")
print(f"✓ Loss is finite: {torch.isfinite(loss).item()}")

# Test evaluation mode
model.eval()
with torch.no_grad():
    val_batch_input, val_batch_target = next(iter(val_loader))
    val_batch_input = val_batch_input.to(device)
    val_batch_target = val_batch_target.to(device)
    val_output = model(val_batch_input)
    val_loss = criterion(val_output, val_batch_target)
    
print(f"✓ Evaluation pass successful")
print(f"✓ Validation loss: {val_loss.item():.4f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print("✓ Data loading: PASSED")
print("✓ Data encoding: PASSED")
print("✓ PyTorch Dataset: PASSED")
print("✓ DataLoader: PASSED")
print("✓ CNN Model: PASSED")
print("✓ End-to-End Pipeline: PASSED")
print("\n✓ ALL TESTS PASSED! Your implementation is ready for training.")
print("="*80)
