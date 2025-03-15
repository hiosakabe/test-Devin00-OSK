import torch
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the trained model and data
def load_model_and_data():
    """Load the trained model and data from files."""
    # Load the saved data
    with open('nested_training_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    sequences = data['sequences']
    element_to_idx = data['element_to_idx']
    idx_to_element = data['idx_to_element']
    vocab_size = data['vocab_size']
    
    # Define the model class
    class NestedSequencePredictor(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim):
            super(NestedSequencePredictor, self).__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.fc = torch.nn.Linear(hidden_dim, vocab_size)
            
        def forward(self, sequence):
            embeds = self.embedding(sequence)
            lstm_out, _ = self.lstm(embeds)
            lstm_out = lstm_out[:, -1, :]
            output = self.fc(lstm_out)
            return output
    
    # Initialize the model
    model = NestedSequencePredictor(vocab_size, embedding_dim=128, hidden_dim=256)
    
    # Load the state dict
    model.load_state_dict(torch.load('nested_sequence_model.pth'))
    model.eval()
    
    return model, sequences, element_to_idx, idx_to_element

# Function to prepare sequence for model input
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

# Function for inference
def predict_next_element(model, sequence, element_to_idx, idx_to_element):
    """Predict the next element in a sequence."""
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

# Define test sequences
def run_inference_tests():
    """Run inference tests on various sequence patterns."""
    # Load model and data
    model, sequences, element_to_idx, idx_to_element = load_model_and_data()
    
    # Define test sequences
    test_sequences = [
        ['A', 'C', 'D'],                      # Basic sequence
        ['A', '', 'D'],                       # Sequence with blank
        [['A', 'B'], 'C'],                    # Sequence with nested input
        ['A', 'B', ''],                       # Sequence ending with blank
        [['A', 'C'], 'E', 'G'],               # Longer sequence with nested input
        ['A', '', 'F', 'H'],                  # Longer sequence with blank
        [['A', 'B', 'C'], ['D', 'E'], 'F'],   # Complex sequence with nested inputs
        ['A', 'C', ['E', 'F']],               # Sequence with nested element at the end
        ['A', 'B', 'C', 'D', 'E'],            # Longer basic sequence
        [['A', 'D'], '', ['G', 'H']]          # Complex sequence with blank and nested elements
    ]
    
    # Run inference on each test sequence
    print("# 10パターンの推論テスト結果\n")
    
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
            
            print(f"## テスト {i+1}: {formatted_seq}→?\n")
            print(f"予測された次の要素: **{formatted_prediction}** (確率: {probabilities[next_element]*100:.2f}%)\n")
            
            print("上位予測の確率:")
            for element, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]:
                if isinstance(element, tuple):
                    formatted_element = f"[{','.join(element)}]"
                elif element == "":
                    formatted_element = "_"
                else:
                    formatted_element = element
                print(f"- {formatted_element}: {prob*100:.2f}%")
            
            # Analyze the training data for this pattern
            next_elements = []
            for train_seq in sequences:
                # Check if the sequence contains the pattern
                for j in range(len(train_seq) - len(seq)):
                    match = True
                    for k, element in enumerate(seq):
                        seq_element = train_seq[j+k]
                        if isinstance(element, list) and isinstance(seq_element, list):
                            # Compare sorted lists
                            if sorted(element) != sorted(seq_element):
                                match = False
                                break
                        elif element != seq_element:
                            match = False
                            break
                    
                    if match and j+len(seq) < len(train_seq):
                        next_elements.append(train_seq[j+len(seq)])
            
            if next_elements:
                element_counts = defaultdict(int)
                for element in next_elements:
                    if isinstance(element, list):
                        element_counts[tuple(sorted(element))] += 1
                    else:
                        element_counts[element] += 1
                
                total = len(next_elements)
                print("\n教師データでの分布:")
                for element, count in sorted(element_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    if isinstance(element, tuple):
                        formatted_element = f"[{','.join(element)}]"
                    elif element == "":
                        formatted_element = "_"
                    else:
                        formatted_element = element
                    print(f"- {formatted_element}: {count/total*100:.2f}%")
            else:
                print("\n教師データには該当するパターンがありませんでした。")
            
            print("\n" + "-"*50 + "\n")
        else:
            print(f"## テスト {i+1}: {format_sequence(seq)}→?\n")
            print("このシーケンスには語彙にない要素が含まれているため、推論できません。\n")
            print("-"*50 + "\n")

# Run the tests
if __name__ == "__main__":
    run_inference_tests()
