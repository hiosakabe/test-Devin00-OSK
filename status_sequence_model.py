import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import string
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

class StatusSequencePredictor(nn.Module):
    """
    A sequence prediction model that incorporates user status information.
    This model extends the nested sequence model by adding user attributes
    like age and gender as additional input features.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, status_dim):
        super(StatusSequencePredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.status_encoder = nn.Linear(status_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim + embedding_dim, vocab_size)
        
    def forward(self, sequence, status):
        # sequence shape: (batch_size, sequence_length)
        # status shape: (batch_size, status_dim)
        
        # Embed the sequence
        embeds = self.embedding(sequence)  # (batch_size, sequence_length, embedding_dim)
        
        # Process the sequence with LSTM
        lstm_out, _ = self.lstm(embeds)  # (batch_size, sequence_length, hidden_dim)
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Encode the status information
        status_encoded = self.status_encoder(status)  # (batch_size, embedding_dim)
        
        # Concatenate LSTM output with encoded status
        combined = torch.cat([lstm_out, status_encoded], dim=1)  # (batch_size, hidden_dim + embedding_dim)
        
        # Final prediction
        output = self.fc(combined)  # (batch_size, vocab_size)
        return output

# Generate random training data with status information
def generate_status_sequences(num_sequences=1000, min_length=3, max_length=10, 
                             blank_prob=0.1, multi_input_prob=0.15):
    """
    Generate random sequences with associated user status information.
    Each sequence is paired with random status attributes (age, gender).
    """
    alphabet = string.ascii_uppercase
    sequences = []
    status_info = []
    
    # Define possible status values
    ages = list(range(18, 65))  # Ages from 18 to 64
    genders = [0, 1]  # 0 for female, 1 for male
    
    for _ in range(num_sequences):
        # Randomly choose sequence length
        length = random.randint(min_length, max_length)
        
        # Generate a sequence (similar to nested_sequence_model.py)
        if random.random() < multi_input_prob:
            # Start with multiple letters
            num_alternatives = random.randint(2, 3)
            alternatives = ['A'] + random.sample(alphabet[1:5], num_alternatives - 1)
            sequence = [alternatives]
        else:
            sequence = [['A']]  # Start with 'A' as a single-element list
        
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
                sequence.append([''])  # Blank as a single-element list
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
                sequence.append(alternatives)
                
                # Update current position to the highest letter used
                current_idx = max(alt_indices)
            else:
                # Add the next letter to the sequence
                sequence.append([alphabet[next_idx]])
                
                # Update current position
                current_idx = next_idx
        
        # Generate random status information
        age = random.choice(ages)
        gender = random.choice(genders)
        
        # Add sequence and status to the lists
        sequences.append(sequence)
        status_info.append({
            'age': age,
            'gender': gender
        })
    
    return sequences, status_info

# Create a mapping from elements to indices
def create_vocabulary(sequences):
    """Create a vocabulary mapping from sequence elements to indices."""
    # Collect all unique elements
    unique_elements = set()
    for seq in sequences:
        for element_list in seq:
            # Sort the elements in each list to ensure consistent representation
            element_tuple = tuple(sorted(element_list))
            unique_elements.add(element_tuple)
    
    # Sort elements for deterministic mapping
    sorted_elements = sorted(list(unique_elements), key=lambda x: (len(x), x))
    
    # Create mappings
    element_to_idx = {element: i for i, element in enumerate(sorted_elements)}
    idx_to_element = {i: element for i, element in enumerate(sorted_elements)}
    
    return element_to_idx, idx_to_element, len(sorted_elements)

# Prepare sequence for model input
def prepare_sequence(seq, element_to_idx):
    """Convert a sequence of element lists to tensor of indices."""
    indices = []
    for element_list in seq:
        # Sort the elements in the list to ensure consistent representation
        element_tuple = tuple(sorted(element_list))
        indices.append(element_to_idx[element_tuple])
    return torch.tensor(indices, dtype=torch.long)

# Prepare status information for model input
def prepare_status(status, normalize=True):
    """Convert status dictionary to tensor."""
    # Extract status values
    age = status['age']
    gender = status['gender']
    
    # Normalize age to [0, 1] range if requested
    if normalize:
        age = (age - 18) / (65 - 18)  # Normalize based on the range defined in generate_status_sequences
    
    # Create tensor
    return torch.tensor([age, gender], dtype=torch.float)

# Main function to train the model
def train_status_sequence_model(num_sequences=3000, num_epochs=100, batch_size=64):
    """Train a sequence prediction model that incorporates status information."""
    print("Generating training data...")
    sequences, status_info = generate_status_sequences(num_sequences)
    
    # Create vocabulary
    element_to_idx, idx_to_element, vocab_size = create_vocabulary(sequences)
    print(f"Vocabulary size: {vocab_size}")
    
    # Prepare training data
    X_train = []
    status_train = []
    y_train = []
    
    for i, seq in enumerate(sequences):
        for j in range(1, len(seq)):
            # Use a window of up to 3 previous elements as input
            start_idx = max(0, j-3)
            X_train.append(prepare_sequence(seq[start_idx:j], element_to_idx))
            status_train.append(prepare_status(status_info[i]))
            # Target: element at position j
            target_element = tuple(sorted(seq[j]))
            y_train.append(element_to_idx[target_element])
    
    # Hyperparameters
    embedding_dim = 128
    hidden_dim = 256
    status_dim = 2  # Age and gender
    learning_rate = 0.001
    
    # Initialize the model
    model = StatusSequencePredictor(vocab_size, embedding_dim, hidden_dim, status_dim)
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
            batch_status = [status_train[i] for i in batch_indices]
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
            batch_status_tensor = torch.stack(batch_status)
            batch_y_tensor = torch.tensor(batch_y, dtype=torch.long)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch_X_tensor, batch_status_tensor)
            
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
    plt.savefig('status_training_loss.png')
    print("Training loss plot saved to status_training_loss.png")
    
    # Save the trained model
    torch.save(model.state_dict(), "status_sequence_model.pth")
    print("Model saved to status_sequence_model.pth")
    
    # Save the training data and vocabulary
    data = {
        'sequences': sequences,
        'status_info': status_info,
        'element_to_idx': element_to_idx,
        'idx_to_element': idx_to_element,
        'vocab_size': vocab_size
    }
    
    with open('status_training_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print("Training data saved to status_training_data.pkl")
    
    return model, sequences, status_info, element_to_idx, idx_to_element

# Function for inference
def predict_next_element(model, sequence, status, element_to_idx, idx_to_element):
    """Predict the next element in a sequence given status information."""
    model.eval()
    with torch.no_grad():
        # Convert sequence to tensor
        sequence_tensor = prepare_sequence(sequence, element_to_idx).unsqueeze(0)
        
        # Convert status to tensor
        status_tensor = prepare_status(status).unsqueeze(0)
        
        # Get model prediction
        output = model(sequence_tensor, status_tensor)
        
        # Get probabilities for all elements
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Get the most likely next element
        _, predicted_idx = torch.max(output, 1)
        predicted_element = idx_to_element[predicted_idx.item()]
        
        # Return the predicted element and all probabilities
        return predicted_element, {idx_to_element[i]: prob.item() for i, prob in enumerate(probabilities)}

# Function to format a sequence for display
def format_sequence(sequence):
    """Format a sequence with nested elements for display."""
    formatted = []
    for element_list in sequence:
        if len(element_list) == 0:
            formatted.append("_")  # Empty list
        elif len(element_list) == 1:
            if element_list[0] == "":
                formatted.append("_")  # Blank
            else:
                formatted.append(element_list[0])  # Single character
        else:
            formatted.append(f"[{','.join(element_list)}]")  # Multiple characters
    return "→".join(formatted)

# Function to format status for display
def format_status(status):
    """Format status information for display."""
    gender = "男性" if status['gender'] == 1 else "女性"
    return f"年齢：{status['age']}歳/性別：{gender}"

# Run inference tests with different status combinations
def run_status_inference_tests(model, element_to_idx, idx_to_element):
    """Run inference tests with different status combinations."""
    # Define test sequences
    test_sequences = [
        [['A'], ['C'], ['D']],                      # Basic sequence
        [['A'], [''], ['D']],                       # Sequence with blank
        [['A', 'B'], ['C']],                        # Sequence with nested input
        [['A'], ['B'], ['']],                       # Sequence ending with blank
        [['A', 'C'], ['E'], ['G']],                 # Longer sequence with nested input
    ]
    
    # Define test status combinations
    test_statuses = [
        {'age': 25, 'gender': 1},  # 25-year-old male
        {'age': 25, 'gender': 0},  # 25-year-old female
        {'age': 45, 'gender': 1},  # 45-year-old male
        {'age': 45, 'gender': 0},  # 45-year-old female
    ]
    
    print("# ステータス情報を加味した推論テスト結果\n")
    
    for i, seq in enumerate(test_sequences):
        print(f"## テストシーケンス {i+1}: {format_sequence(seq)}→?\n")
        
        for status in test_statuses:
            next_element, probabilities = predict_next_element(
                model, seq, status, element_to_idx, idx_to_element
            )
            
            # Format the prediction for display
            if isinstance(next_element, tuple):
                if len(next_element) == 0:
                    formatted_prediction = "_"
                elif len(next_element) == 1:
                    formatted_prediction = next_element[0] if next_element[0] != "" else "_"
                else:
                    formatted_prediction = f"[{','.join(next_element)}]"
            else:
                formatted_prediction = str(next_element)
            
            print(f"### {format_status(status)}\n")
            print(f"予測された次の要素: **{formatted_prediction}** (確率: {probabilities[next_element]*100:.2f}%)\n")
            
            print("上位予測の確率:")
            for element, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]:
                if isinstance(element, tuple):
                    if len(element) == 0:
                        formatted_element = "_"
                    elif len(element) == 1:
                        formatted_element = element[0] if element[0] != "" else "_"
                    else:
                        formatted_element = f"[{','.join(element)}]"
                else:
                    formatted_element = str(element)
                print(f"- {formatted_element}: {prob*100:.2f}%")
            
            print("\n" + "-"*30 + "\n")
        
        print("\n" + "="*50 + "\n")

# Main execution
if __name__ == "__main__":
    # Train the model
    model, sequences, status_info, element_to_idx, idx_to_element = train_status_sequence_model(
        num_sequences=3000,
        num_epochs=100,
        batch_size=64
    )
    
    # Run inference tests
    run_status_inference_tests(model, element_to_idx, idx_to_element)
