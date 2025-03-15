import torch
import numpy as np
import random
import string
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

# Define the SequencePredictor class directly to avoid importing from enhanced_sequence_model
class SequencePredictor(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SequencePredictor, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, sequence):
        # sequence shape: (batch_size, sequence_length)
        embeds = self.embedding(sequence)  # (batch_size, sequence_length, embedding_dim)
        lstm_out, _ = self.lstm(embeds)  # (batch_size, sequence_length, hidden_dim)
        # We only need the last output for prediction
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        output = self.fc(lstm_out)  # (batch_size, vocab_size)
        return output

# Load the enhanced model for testing
def load_model(model_path, token_to_idx, idx_to_token):
    """Load a trained model from a file."""
    # Get vocabulary size from token_to_idx
    vocab_size = len(token_to_idx)
    
    # Initialize model with the same parameters used during training
    model = SequencePredictor(vocab_size, embedding_dim=64, hidden_dim=128)
    
    # Load the state dict
    model.load_state_dict(torch.load(model_path))
    
    # Set model to evaluation mode
    model.eval()
    
    return model

# Function to prepare sequence for model input
def prepare_sequence(seq, token_to_idx):
    """Convert a sequence of tokens to tensor of indices."""
    return torch.tensor([token_to_idx[token] for token in seq], dtype=torch.long)

# Function for inference
def predict_next_token(model, sequence, token_to_idx, idx_to_token):
    """Predict the next token in a sequence."""
    with torch.no_grad():
        # Convert sequence to tensor
        sequence_tensor = prepare_sequence(sequence, token_to_idx).unsqueeze(0)
        
        # Get model prediction
        output = model(sequence_tensor)
        
        # Get probabilities for all tokens
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Get the most likely next token
        _, predicted_idx = torch.max(output, 1)
        predicted_token = idx_to_token[predicted_idx.item()]
        
        # Return the predicted token and all probabilities
        return predicted_token, {idx_to_token[i]: prob.item() for i, prob in enumerate(probabilities)}

# Load token mappings and sequences from the saved training data
def load_token_mappings():
    """Load token to index mappings from the saved training data."""
    # Load the saved training data
    with open('training_data.pkl', 'rb') as f:
        sequences = pickle.load(f)
    
    # Create the same token mappings
    all_tokens = set()
    for seq in sequences:
        for token in seq:
            all_tokens.add(token)
    
    # Add special token for blank
    all_tokens.add('')
    
    # Sort tokens for deterministic mapping
    unique_tokens = sorted(list(all_tokens), key=lambda x: (len(x), x))
    token_to_idx = {token: i for i, token in enumerate(unique_tokens)}
    idx_to_token = {i: token for i, token in enumerate(unique_tokens)}
    
    return token_to_idx, idx_to_token, sequences

# Main function to run inference tests
def run_inference_tests():
    """Run inference tests on various sequence patterns."""
    # Load token mappings and training data
    token_to_idx, idx_to_token, training_sequences = load_token_mappings()
    
    # Load the model
    model = load_model("enhanced_sequence_model.pth", token_to_idx, idx_to_token)
    
    # Define test sequences
    test_sequences = [
        ['A', 'C', 'D'],           # Basic sequence
        ['A', '', 'D'],            # Sequence with blank
        ['A/B', 'C'],              # Sequence with multiple inputs
        ['A', 'B', ''],            # Sequence ending with blank
        ['A/C', 'E', 'G'],         # Longer sequence with multiple inputs
        ['A', '', 'F', 'H'],       # Longer sequence with blank
        ['A/B/C', 'D/E', 'F'],     # Complex sequence with multiple inputs
        ['A', 'C', 'E/F'],         # Sequence with multiple inputs at the end
        ['A', 'B', 'C', 'D', 'E'], # Longer basic sequence
        ['A/D', '', 'G/H']         # Complex sequence with blank and multiple inputs
    ]
    
    # Run inference on each test sequence
    print("# 10パターンの推論テスト結果\n")
    
    for i, seq in enumerate(test_sequences):
        # Check if all tokens in the sequence are in the vocabulary
        if all(token in token_to_idx for token in seq):
            next_token, probabilities = predict_next_token(model, seq, token_to_idx, idx_to_token)
            
            # Format the sequence for display (replace empty string with '_')
            display_seq = ['_' if token == '' else token for token in seq]
            
            print(f"## テスト {i+1}: {'→'.join(display_seq)}→?\n")
            print(f"予測された次のトークン: **{next_token if next_token != '' else '_'}** (確率: {probabilities[next_token]*100:.2f}%)\n")
            
            print("上位トークンの確率:")
            for token, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]:
                display_token = '_' if token == '' else token
                print(f"- {display_token}: {prob*100:.2f}%")
            
            # Analyze the training data for this pattern
            next_tokens = []
            for train_seq in training_sequences:
                # Check if the sequence contains the pattern
                for j in range(len(train_seq) - len(seq)):
                    if train_seq[j:j+len(seq)] == seq and j+len(seq) < len(train_seq):
                        next_tokens.append(train_seq[j+len(seq)])
            
            if next_tokens:
                token_counts = defaultdict(int)
                for token in next_tokens:
                    token_counts[token] += 1
                
                total = len(next_tokens)
                print("\n教師データでの分布:")
                for token, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    display_token = '_' if token == '' else token
                    print(f"- {display_token}: {count/total*100:.2f}%")
            else:
                print("\n教師データには該当するパターンがありませんでした。")
            
            print("\n" + "-"*50 + "\n")
        else:
            print(f"## テスト {i+1}: {'→'.join(['_' if token == '' else token for token in seq])}→?\n")
            print("このシーケンスには語彙にないトークンが含まれているため、推論できません。\n")
            print("-"*50 + "\n")

# Run the tests
if __name__ == "__main__":
    run_inference_tests()
