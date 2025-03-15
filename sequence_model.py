import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the sequence data from the problem
sequences = [
    ['A', 'B', 'C'],
    ['A', 'B'],
    ['A', 'B', 'C', 'D', 'E', 'F'],
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
    ['A', 'B', 'C', 'D'],
    ['A', 'B', 'C']
]

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
        # Input: characters up to position i
        X_train.append(prepare_sequence(seq[:i], char_to_idx))
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
embedding_dim = 16
hidden_dim = 32
learning_rate = 0.01
num_epochs = 1000

# Initialize the model
model = SequencePredictor(vocab_size, embedding_dim, hidden_dim)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print("Starting training...")
for epoch in range(num_epochs):
    total_loss = 0
    
    for i in range(len(X_train)):
        # Get the input sequence and target
        sequence = X_train[i].unsqueeze(0)  # Add batch dimension
        target = torch.tensor([y_train[i]], dtype=torch.long)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(sequence)
        
        # Calculate loss
        loss = loss_function(output, target)
        total_loss += loss.item()
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
    
    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(X_train):.4f}")

print("Training complete!")

# Save the trained model
torch.save(model.state_dict(), "sequence_model.pth")
print("Model saved to sequence_model.pth")

# Function for inference
def predict_next_char(model, sequence, char_to_idx, idx_to_char):
    """Predict the next character in a sequence."""
    model.eval()
    with torch.no_grad():
        # Convert sequence to tensor
        sequence_tensor = prepare_sequence(sequence, char_to_idx).unsqueeze(0)
        
        # Get model prediction
        output = model(sequence_tensor)
        
        # Get the most likely next character
        _, predicted_idx = torch.max(output, 1)
        predicted_char = idx_to_char[predicted_idx.item()]
        
        # Get probabilities for all characters
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Return the predicted character and all probabilities
        return predicted_char, {idx_to_char[i]: prob.item() for i, prob in enumerate(probabilities)}

# Test the model with the inference example: A→B→?
test_sequence = ['A', 'B']
next_char, probabilities = predict_next_char(model, test_sequence, char_to_idx, idx_to_char)

print(f"\nInference example: {'→'.join(test_sequence)}→?")
print(f"Predicted next character: {next_char}")
print("Probabilities for each character:")
for char, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
    print(f"{char}: {prob:.4f}")
