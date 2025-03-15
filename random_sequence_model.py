import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import string
import matplotlib.pyplot as plt

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Generate random training data
def generate_random_sequences(num_sequences=1000, min_length=3, max_length=10):
    """
    Generate random sequences that follow alphabetical order (A to Z) with possible skips.
    The sequences never go backwards in the alphabet.
    """
    alphabet = string.ascii_uppercase
    sequences = []
    
    for _ in range(num_sequences):
        # Randomly choose sequence length
        length = random.randint(min_length, max_length)
        
        # Start with 'A'
        sequence = ['A']
        
        # Current position in alphabet
        current_idx = 0
        
        # Generate the rest of the sequence
        while len(sequence) < length and current_idx < len(alphabet) - 1:
            # Randomly decide how many letters to skip (0, 1, 2, or 3)
            skip = random.randint(0, 3)
            
            # Make sure we don't go beyond 'Z'
            next_idx = min(current_idx + skip + 1, len(alphabet) - 1)
            
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
for i in range(5):
    print(f"Sequence {i+1}: {'→'.join(sequences[i])}")

# Create a mapping from characters to indices
unique_chars = sorted(list(set(char for seq in sequences for char in seq)))
char_to_idx = {char: i for i, char in enumerate(unique_chars)}
idx_to_char = {i: char for i, char in enumerate(unique_chars)}
vocab_size = len(unique_chars)

print(f"Vocabulary: {unique_chars}")
print(f"Vocabulary size: {vocab_size}")

# Convert sequences to numerical format
def prepare_sequence(seq, char_to_idx):
    """Convert a sequence of characters to tensor of indices."""
    return torch.tensor([char_to_idx[char] for char in seq], dtype=torch.long)

# Prepare training data
X_train = []
y_train = []

for seq in sequences:
    for i in range(1, len(seq)):
        # Use a window of up to 3 previous characters as input
        start_idx = max(0, i-3)
        X_train.append(prepare_sequence(seq[start_idx:i], char_to_idx))
        # Target: character at position i
        y_train.append(char_to_idx[seq[i]])

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
embedding_dim = 32
hidden_dim = 64
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
plt.savefig('training_loss.png')
print("Training loss plot saved to training_loss.png")

# Save the trained model
torch.save(model.state_dict(), "random_sequence_model.pth")
print("Model saved to random_sequence_model.pth")

# Function for inference
def predict_next_char(model, sequence, char_to_idx, idx_to_char):
    """Predict the next character in a sequence."""
    model.eval()
    with torch.no_grad():
        # Convert sequence to tensor
        sequence_tensor = prepare_sequence(sequence, char_to_idx).unsqueeze(0)
        
        # Get model prediction
        output = model(sequence_tensor)
        
        # Get probabilities for all characters
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Get the most likely next character
        _, predicted_idx = torch.max(output, 1)
        predicted_char = idx_to_char[predicted_idx.item()]
        
        # Return the predicted character and all probabilities
        return predicted_char, {idx_to_char[i]: prob.item() for i, prob in enumerate(probabilities)}

# Test the model with the inference example: A→C→D→?
test_sequence = ['A', 'C', 'D']
next_char, probabilities = predict_next_char(model, test_sequence, char_to_idx, idx_to_char)

print(f"\nInference example: {'→'.join(test_sequence)}→?")
print(f"Predicted next character: {next_char}")
print("Probabilities for each character:")
for char, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{char}: {prob:.4f}")

# Analyze the training data to see what typically follows A→C→D
acd_sequences = []
for seq in sequences:
    if len(seq) >= 3 and seq[0] == 'A' and seq[1] == 'C' and seq[2] == 'D':
        if len(seq) > 3:
            acd_sequences.append(seq[3])

if acd_sequences:
    print("\nIn the training data, after A→C→D, the following characters appear:")
    char_counts = {}
    for char in acd_sequences:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    total = len(acd_sequences)
    for char, count in sorted(char_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{char}: {count} times ({count/total*100:.2f}%)")
else:
    print("\nNo sequences in the training data match the pattern A→C→D.")
