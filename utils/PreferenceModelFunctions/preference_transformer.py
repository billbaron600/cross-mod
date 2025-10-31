import pickle
import torch
import os
from torch.utils.data import Dataset, DataLoader,random_split
import torch.nn as nn
import torch.optim as optim

class DONTUSEPreferenceTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=128, num_heads=4, num_layers=2):
        super(PreferenceTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(model_dim, 1)  # Output preference score

    def forward(self, trajectory):
        # trajectory shape: (batch, seq_len, input_dim)
        x = self.input_proj(trajectory)
        transformer_output = self.transformer_encoder(x)
        
        # Pooling over sequence length (use mean pooling)
        pooled_output = transformer_output.mean(dim=1)
        
        score = self.output_layer(pooled_output)
        return score.squeeze(-1)

class PreferenceTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=128, num_heads=4, num_layers=2, device=None,dtype=torch.float32):
        super(PreferenceTransformer, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.input_proj = nn.Linear(input_dim, model_dim,dtype=dtype).to(self.device)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True,dtype=dtype).to(self.device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(self.device)
        
        self.output_layer = nn.Linear(model_dim, 1,dtype=dtype).to(self.device)  # Output preference score

    def forward(self, trajectory):
        # Ensure trajectory is on the correct device
        trajectory = trajectory.to(self.device)

        # trajectory shape: (batch, seq_len, input_dim)
        x = self.input_proj(trajectory)
        transformer_output = self.transformer_encoder(x)
        
        # Pooling over sequence length (use mean pooling)
        pooled_output = transformer_output.mean(dim=1)
        
        score = self.output_layer(pooled_output)
        return score.squeeze(-1)


class PairwisePreferenceDataset(Dataset):
    def __init__(self, pairwise_data):
        self.pairwise_data = pairwise_data

    def __len__(self):
        return len(self.pairwise_data)

    def __getitem__(self, idx):
        preferred = self.pairwise_data[idx]['preferred']
        non_preferred = self.pairwise_data[idx]['non_preferred']
        return preferred, non_preferred


def preference_loss(preferred_scores, non_preferred_scores):
    # Bradley-Terry pairwise logistic loss
    return -torch.mean(torch.log(torch.sigmoid(preferred_scores - non_preferred_scores)))


def train_preference_transformer(pairwise_data, model=None,input_dim=None, epochs=10, batch_size=32, learning_rate=1e-4, validation_percent=None):
    if input_dim is None:
        input_dim = pairwise_data[0]['preferred'].shape[-1]
    
    dataset = PairwisePreferenceDataset(pairwise_data)

    # Partition the dataset into training and validation if needed
    if validation_percent is not None and 0 < validation_percent < 1:
        val_size = int(len(dataset) * validation_percent)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_dataset = dataset
        val_loader = None  # No validation set

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if model==None:
        model = PreferenceTransformer(input_dim)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_history = []
    val_loss_history = [] if validation_percent is not None else None  # Store validation loss if applicable

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for preferred, non_preferred in train_loader:
            optimizer.zero_grad()

            preferred_scores = model(preferred)
            non_preferred_scores = model(non_preferred)

            loss = preference_loss(preferred_scores, non_preferred_scores)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation phase
        if val_loader:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for preferred, non_preferred in val_loader:
                    preferred_scores = model(preferred)
                    non_preferred_scores = model(non_preferred)
                    val_loss = preference_loss(preferred_scores, non_preferred_scores)
                    total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            val_loss_history.append(avg_val_loss)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

    # Attach loss history to the model object
    model.train_loss_history = train_loss_history
    model.val_loss_history = val_loss_history if val_loader else None

    return model, train_loss_history, val_loss_history

# Save the entire model to a file
def save_model(model, filepath="preference_transformer.pth"):
    filepath = filepath + "/preference_model.pth"
    torch.save(model, filepath)
    print(f"Model saved to {filepath}")
    return filepath

# Load the entire model from a file
def load_model(filepath="preference_transformer.pth"):
    model = torch.load(filepath)
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {filepath}")
    return model

# Function to run inference on a given input
def run_inference(model, input_tensor):
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        output = model(input_tensor)
    return output