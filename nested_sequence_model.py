import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import string
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from typing import List, Union, Tuple, Dict, Any

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Define the type for our nested sequence elements
NestedElement = Union[str, List[str]]
Sequence = List[NestedElement]

# Generate random training data with nested elements
def generate_nested_sequences(num_sequences=10000, min_length=3, max_length=10, 
                             blank_prob=0.1, nested_prob=0.2):
    """
    Generate random sequences that follow alphabetical order (A to Z) with possible skips.
    The sequences can contain nested elements like [A,B] to represent simultaneous occurrence.
    
    Parameters:
    - num_sequences: Number of sequences to generate
    - min_length: Minimum sequence length
    - max_length: Maximum sequence length
    - blank_prob: Probability of inserting a blank in the sequence
    - nested_prob: Probability of creating a nested element (list of characters)
    """
    alphabet = string.ascii_uppercase
    sequences = []
    
    for _ in range(num_sequences):
        # Randomly choose sequence length
        length = random.randint(min_length, max_length)
        
        # Start with 'A' or a nested element including 'A'
        if random.random() < nested_prob:
            # Choose a random number of elements in the nest (2 or 3)
            num_elements = random.randint(2, 3)
            # Always include 'A' and some other early letters
            elements = ['A'] + random.sample(alphabet[1:5], num_elements - 1)
            sequence = [elements]  # Nested element as a list
        else:
            sequence = ['A']  # Single character as a string
        
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
                sequence.append("")  # Blank as an empty string
            # Decide if this position should be a nested element
            elif random.random() < nested_prob:
                # Choose a random number of elements in the nest (2 or 3)
                num_elements = random.randint(2, 3)
                # Choose letters that are close to the next expected letter
                base_idx = next_idx
                # Get possible indices within a range of the base index
                possible_indices = [i for i in range(max(0, base_idx-1), min(len(alphabet), base_idx+2))]
                # Ensure we have enough unique indices
                if len(possible_indices) < num_elements:
                    possible_indices = list(range(max(0, base_idx-2), min(len(alphabet), base_idx+3)))
                
                # Sample the elements for the nest
                nest_indices = random.sample(possible_indices, num_elements)
                nest_elements = [alphabet[i] for i in nest_indices]
                sequence.append(nest_elements)  # Add the nested element
                
                # Update current position to the highest letter used
                current_idx = max(nest_indices)
            else:
                # Add the next letter to the sequence as a single character
                sequence.append(alphabet[next_idx])
                
                # Update current position
                current_idx = next_idx
        
        sequences.append(sequence)
    
    return sequences

# Function to format a sequence for display
def format_sequence(sequence):
    """Format a sequence with nested elements for display."""
    formatted = []
    for element in sequence:
        if isinstance(element, list):
            formatted.append(f"[{','.join(element)}]")
        elif element == "":
            formatted.append("_")
        else:
            formatted.append(element)
    return "→".join(formatted)

# Generate the training data
sequences = generate_nested_sequences(10000)

# Print some example sequences
print("Example sequences:")
for i in range(10):
    print(f"Sequence {i+1}: {format_sequence(sequences[i])}")

# Create a mapping from elements to indices
# We'll treat each unique element (single char, nested element, or blank) as a separate entity
def create_vocabulary(sequences):
    """Create a vocabulary from the sequences."""
    # First collect all unique elements
    unique_elements = set()
    for seq in sequences:
        for element in seq:
            if isinstance(element, list):
                # For nested elements, we'll use a tuple representation for the vocabulary
                unique_elements.add(tuple(sorted(element)))
            else:
                unique_elements.add(element)
    
    # Convert to a sorted list for deterministic mapping
    # Sort by length first, then by content
    sorted_elements = sorted(list(unique_elements), key=lambda x: (len(x) if isinstance(x, tuple) else 1, x))
    
    # Create mappings
    element_to_idx = {}
    for i, element in enumerate(sorted_elements):
        if isinstance(element, tuple):
            # Store the tuple version in the mapping
            element_to_idx[element] = i
        else:
            element_to_idx[element] = i
    
    idx_to_element = {i: element for element, i in element_to_idx.items()}
    
    return element_to_idx, idx_to_element, len(sorted_elements)

# Create the vocabulary
element_to_idx, idx_to_element, vocab_size = create_vocabulary(sequences)

print(f"Vocabulary size: {vocab_size}")
print("Sample elements in vocabulary:")
for i, element in enumerate(list(element_to_idx.keys())[:20]):
    if isinstance(element, tuple):
        print(f"{i}: {element}")
    else:
        print(f"{i}: '{element}'")
if len(element_to_idx) > 20:
    print("...")

# Convert sequences to numerical format
def prepare_sequence(seq, element_to_idx):
    """Convert a sequence of elements to tensor of indices."""
    indices = []
    for element in seq:
        if isinstance(element, list):
            # Convert list to sorted tuple for lookup
            element_tuple = tuple(sorted(element))
            indices.append(element_to_idx[element_tuple])
        else:
            indices.append(element_to_idx[element])
    return torch.tensor(indices, dtype=torch.long)

# Prepare training data
X_train = []
y_train = []

for seq in sequences:
    for i in range(1, len(seq)):
        # Use a window of up to 3 previous elements as input
        start_idx = max(0, i-3)
        X_train.append(prepare_sequence(seq[start_idx:i], element_to_idx))
        # Target: element at position i
        if isinstance(seq[i], list):
            target_element = tuple(sorted(seq[i]))
        else:
            target_element = seq[i]
        y_train.append(element_to_idx[target_element])

# Define the LSTM model for sequence prediction
class NestedSequencePredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(NestedSequencePredictor, self).__init__()
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
embedding_dim = 128
hidden_dim = 256
learning_rate = 0.001
num_epochs = 100
batch_size = 128

# Initialize the model
model = NestedSequencePredictor(vocab_size, embedding_dim, hidden_dim)
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
plt.savefig('nested_training_loss.png')
print("Training loss plot saved to nested_training_loss.png")

# Save the trained model
torch.save(model.state_dict(), "nested_sequence_model.pth")
print("Model saved to nested_sequence_model.pth")

# Save the training data and vocabulary
with open('nested_training_data.pkl', 'wb') as f:
    pickle.dump({
        'sequences': sequences,
        'element_to_idx': element_to_idx,
        'idx_to_element': idx_to_element,
        'vocab_size': vocab_size
    }, f)
print("Training data and vocabulary saved to nested_training_data.pkl")

# Function for inference
def predict_next_element(model, sequence, element_to_idx, idx_to_element):
    """Predict the next element in a sequence."""
    model.eval()
    with torch.no_grad():
        # Convert sequence to tensor
        sequence_tensor = prepare_sequence(sequence, element_to_idx).unsqueeze(0)
        
        # Get model prediction
        output = model(sequence_tensor)
        
        # Get probabilities for all elements
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Get the most likely next element
        _, predicted_idx = torch.max(output, 1)
        predicted_element = idx_to_element[predicted_idx.item()]
        
        # Return the predicted element and all probabilities
        return predicted_element, {idx_to_element[i]: prob.item() for i, prob in enumerate(probabilities)}

# Test the model with some inference examples
test_sequences = [
    ['A', 'C', 'D'],                  # Basic sequence
    ['A', '', 'D'],                   # Sequence with blank
    [['A', 'B'], 'C'],                # Sequence with nested input
    ['A', 'B', ''],                   # Sequence ending with blank
    [['A', 'C'], 'E', 'G'],           # Longer sequence with nested input
    ['A', '', 'F', 'H'],              # Longer sequence with blank
    [['A', 'B', 'C'], ['D', 'E'], 'F'], # Complex sequence with nested inputs
    ['A', 'C', ['E', 'F']],           # Sequence with nested element at the end
    ['A', 'B', 'C', 'D', 'E'],        # Longer basic sequence
    [['A', 'D'], '', ['G', 'H']]      # Complex sequence with blank and nested elements
]

print("\nInference examples:")
for i, seq in enumerate(test_sequences):
    # Check if all elements in the sequence are in the vocabulary
    valid_sequence = True
    for element in seq:
        if isinstance(element, list):
            element_tuple = tuple(sorted(element))
            if element_tuple not in element_to_idx:
                valid_sequence = False
                break
        elif element not in element_to_idx:
            valid_sequence = False
            break
    
    if valid_sequence:
        next_element, probabilities = predict_next_element(model, seq, element_to_idx, idx_to_element)
        
        # Format the sequence and prediction for display
        formatted_seq = format_sequence(seq)
        if isinstance(next_element, tuple):
            formatted_prediction = f"[{','.join(next_element)}]"
        elif next_element == "":
            formatted_prediction = "_"
        else:
            formatted_prediction = next_element
        
        print(f"\nTest {i+1}: {formatted_seq}→?")
        print(f"Predicted next element: {formatted_prediction}")
        
        # Format and display top 5 predictions
        print("Top 5 predictions:")
        for element, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]:
            if isinstance(element, tuple):
                formatted_element = f"[{','.join(element)}]"
            elif element == "":
                formatted_element = "_"
            else:
                formatted_element = element
            print(f"- {formatted_element}: {prob:.4f}")
    else:
        print(f"\nTest {i+1}: {format_sequence(seq)}→?")
        print("Cannot perform inference as some elements are not in vocabulary")

# Analyze the training data to see what typically follows specific patterns
def analyze_sequence_patterns(sequences, pattern):
    """Analyze what typically follows a specific pattern in the training data."""
    next_elements = []
    for seq in sequences:
        # Check if the sequence contains the pattern
        for i in range(len(seq) - len(pattern)):
            match = True
            for j, element in enumerate(pattern):
                seq_element = seq[i+j]
                if isinstance(element, list) and isinstance(seq_element, list):
                    # Compare sorted lists
                    if sorted(element) != sorted(seq_element):
                        match = False
                        break
                elif element != seq_element:
                    match = False
                    break
            
            if match and i+len(pattern) < len(seq):
                next_elements.append(seq[i+len(pattern)])
    
    if next_elements:
        element_counts = defaultdict(int)
        for element in next_elements:
            if isinstance(element, list):
                element_counts[tuple(sorted(element))] += 1
            else:
                element_counts[element] += 1
        
        total = len(next_elements)
        print(f"\nIn the training data, after {format_sequence(pattern)}, the following elements appear:")
        for element, count in sorted(element_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            if isinstance(element, tuple):
                formatted_element = f"[{','.join(element)}]"
            elif element == "":
                formatted_element = "_"
            else:
                formatted_element = element
            print(f"- {formatted_element}: {count} times ({count/total*100:.2f}%)")
    else:
        print(f"\nNo sequences in the training data match the pattern {format_sequence(pattern)}.")

# Analyze patterns in the training data for a few examples
analyze_sequence_patterns(sequences, [['A', 'B'], 'C'])
analyze_sequence_patterns(sequences, ['A', 'C', 'D'])
analyze_sequence_patterns(sequences, [['A', 'C'], ['D', 'E']])
