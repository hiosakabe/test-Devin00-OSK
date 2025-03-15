import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import string
import matplotlib.pyplot as plt
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Generate random training data with blanks and multiple inputs
def generate_random_sequences(num_sequences=1000, min_length=3, max_length=10, blank_prob=0.1, multi_input_prob=0.15):
    """
    Generate random sequences that follow alphabetical order (A to Z) with possible skips.
    The sequences never go backwards in the alphabet.
    
    Parameters:
    - num_sequences: Number of sequences to generate
    - min_length: Minimum sequence length
    - max_length: Maximum sequence length
    - blank_prob: Probability of inserting a blank in the sequence
    - multi_input_prob: Probability of having multiple inputs at a position
    """
    alphabet = string.ascii_uppercase
    sequences = []
    
    for _ in range(num_sequences):
        # Randomly choose sequence length
        length = random.randint(min_length, max_length)
        
        # Start with 'A' or multiple letters including 'A'
        if random.random() < multi_input_prob:
            # Choose a random number of alternatives (2 or 3)
            num_alternatives = random.randint(2, 3)
            # Always include 'A' and some other early letters
            alternatives = ['A'] + random.sample(alphabet[1:5], num_alternatives - 1)
            sequence = ['/'.join(alternatives)]
        else:
            sequence = ['A']
        
        # Current position in alphabet
        current_idx = 0
        
        # Generate the rest of the sequence
        while len(sequence) < length and current_idx < len(alphabet) - 1:
            # Randomly decide how many letters to skip (0, 1, 2, or 3)
            skip = random.randint(0, 3)
            
            # Make sure we don't go beyond 'Z'
            next_idx = min(current_idx + skip + 1, len(alphabet) - 1)
            
            # Decide if this position should be a blank
            if random.random() < blank_prob:
                sequence.append('')  # Blank
            # Decide if this position should have multiple inputs
            elif random.random() < multi_input_prob:
                # Choose a random number of alternatives (2 or 3)
                num_alternatives = random.randint(2, 3)
                # Choose letters that are close to the next expected letter
                base_idx = next_idx
                # Get possible indices within a range of the base index
                possible_indices = [i for i in range(max(0, base_idx-1), min(len(alphabet), base_idx+2))]
                # Ensure we have enough unique indices
                if len(possible_indices) < num_alternatives:
                    possible_indices = list(range(max(0, base_idx-2), min(len(alphabet), base_idx+3)))
                
                # Sample the alternatives
                alt_indices = random.sample(possible_indices, num_alternatives)
                alternatives = [alphabet[i] for i in alt_indices]
                sequence.append('/'.join(alternatives))
                
                # Update current position to the highest letter used
                current_idx = max(alt_indices)
            else:
                # Add the next letter to the sequence
                sequence.append(alphabet[next_idx])
                
                # Update current position
                current_idx = next_idx
        
        sequences.append(sequence)
    
    return sequences

# Generate the training data
sequences = generate_random_sequences(1000)

# Print some example sequences
print("Example sequences:")
for i in range(10):
    print(f"Sequence {i+1}: {'→'.join(['_' if s == '' else s for s in sequences[i]])}")

# Create a mapping from characters and character combinations to indices
# We'll treat each unique token (single char, multi-char, or blank) as a separate entity
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
vocab_size = len(unique_tokens)

print(f"Vocabulary size: {vocab_size}")
print("Sample tokens:")
for i, token in enumerate(unique_tokens[:20]):
    print(f"{i}: '{token}'")
if len(unique_tokens) > 20:
    print("...")

# Convert sequences to numerical format
def prepare_sequence(seq, token_to_idx):
    """Convert a sequence of tokens to tensor of indices."""
    return torch.tensor([token_to_idx[token] for token in seq], dtype=torch.long)

# Prepare training data
X_train = []
y_train = []

for seq in sequences:
    for i in range(1, len(seq)):
        # Use a window of up to 3 previous tokens as input
        start_idx = max(0, i-3)
        X_train.append(prepare_sequence(seq[start_idx:i], token_to_idx))
        # Target: token at position i
        y_train.append(token_to_idx[seq[i]])

# Define the LSTM model for sequence prediction
class SequencePredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SequencePredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, sequence):
        # sequence shape: (batch_size, sequence_length)
        embeds = self.embedding(sequence)  # (batch_size, sequence_length, embedding_dim)
        lstm_out, _ = self.lstm(embeds)  # (batch_size, sequence_length, hidden_dim)
        # We only need the last output for prediction
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        output = self.fc(lstm_out)  # (batch_size, vocab_size)
        return output

# Hyperparameters
embedding_dim = 64
hidden_dim = 128
learning_rate = 0.001
num_epochs = 100
batch_size = 64

# Initialize the model
model = SequencePredictor(vocab_size, embedding_dim, hidden_dim)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print("Starting training...")
losses = []

for epoch in range(num_epochs):
    total_loss = 0
    
    # Create batches
    indices = list(range(len(X_train)))
    random.shuffle(indices)
    
    for start_idx in range(0, len(X_train), batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        
        # Get batch data
        batch_X = [X_train[i] for i in batch_indices]
        batch_y = [y_train[i] for i in batch_indices]
        
        # Pad sequences to the same length
        max_len = max(len(seq) for seq in batch_X)
        padded_batch_X = []
        
        for seq in batch_X:
            padding = torch.zeros(max_len - len(seq), dtype=torch.long)
            padded_seq = torch.cat([padding, seq])
            padded_batch_X.append(padded_seq)
        
        # Convert to tensors
        batch_X_tensor = torch.stack(padded_batch_X)
        batch_y_tensor = torch.tensor(batch_y, dtype=torch.long)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(batch_X_tensor)
        
        # Calculate loss
        loss = loss_function(output, batch_y_tensor)
        total_loss += loss.item()
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
    
    # Record loss
    avg_loss = total_loss / (len(X_train) // batch_size + 1)
    losses.append(avg_loss)
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

print("Training complete!")

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('enhanced_training_loss.png')
print("Training loss plot saved to enhanced_training_loss.png")

# Save the trained model
torch.save(model.state_dict(), "enhanced_sequence_model.pth")
print("Model saved to enhanced_sequence_model.pth")

# Function for inference
def predict_next_token(model, sequence, token_to_idx, idx_to_token):
    """Predict the next token in a sequence."""
    model.eval()
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

# Test the model with the inference example: A→C→D→?
test_sequence = ['A', 'C', 'D']
next_token, probabilities = predict_next_token(model, test_sequence, token_to_idx, idx_to_token)

print(f"\nInference example: {'→'.join(test_sequence)}→?")
print(f"Predicted next token: '{next_token}'")
print("Probabilities for top 10 tokens:")
for token, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"'{token}': {prob:.4f}")

# Test with a sequence containing a blank: A→_→D→?
test_sequence_with_blank = ['A', '', 'D']
if all(token in token_to_idx for token in test_sequence_with_blank):
    next_token, probabilities = predict_next_token(model, test_sequence_with_blank, token_to_idx, idx_to_token)
    print(f"\nInference example with blank: A→_→D→?")
    print(f"Predicted next token: '{next_token}'")
    print("Probabilities for top 10 tokens:")
    for token, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"'{token}': {prob:.4f}")
else:
    print("\nCannot perform inference with blank as some tokens are not in vocabulary")

# Test with a sequence containing multiple inputs: A/B→C→?
test_sequence_with_multi = ['A/B', 'C']
if all(token in token_to_idx for token in test_sequence_with_multi):
    next_token, probabilities = predict_next_token(model, test_sequence_with_multi, token_to_idx, idx_to_token)
    print(f"\nInference example with multiple inputs: {'→'.join(test_sequence_with_multi)}→?")
    print(f"Predicted next token: '{next_token}'")
    print("Probabilities for top 10 tokens:")
    for token, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"'{token}': {prob:.4f}")
else:
    print("\nCannot perform inference with multiple inputs as some tokens are not in vocabulary")

# Analyze the training data to see what typically follows specific patterns
def analyze_sequence_patterns(sequences, pattern):
    """Analyze what typically follows a specific pattern in the training data."""
    next_tokens = []
    for seq in sequences:
        # Check if the sequence contains the pattern
        for i in range(len(seq) - len(pattern)):
            if seq[i:i+len(pattern)] == pattern and i+len(pattern) < len(seq):
                next_tokens.append(seq[i+len(pattern)])
    
    if next_tokens:
        token_counts = defaultdict(int)
        for token in next_tokens:
            token_counts[token] += 1
        
        total = len(next_tokens)
        print(f"\nIn the training data, after {'→'.join(['_' if t == '' else t for t in pattern])}, the following tokens appear:")
        for token, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True):
            display_token = '_' if token == '' else token
            print(f"'{display_token}': {count} times ({count/total*100:.2f}%)")
    else:
        print(f"\nNo sequences in the training data match the pattern {'→'.join(['_' if t == '' else t for t in pattern])}.")

# Analyze patterns in the training data
analyze_sequence_patterns(sequences, ['A', 'C', 'D'])
analyze_sequence_patterns(sequences, ['A', '', 'D'])
analyze_sequence_patterns(sequences, ['A/B', 'C'])
