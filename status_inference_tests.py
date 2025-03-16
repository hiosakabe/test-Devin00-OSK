import torch
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the trained model and data
def load_model_and_data():
    """Load the trained model and data from files."""
    # Load the saved data
    with open('status_training_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    sequences = data['sequences']
    status_info = data['status_info']
    element_to_idx = data['element_to_idx']
    idx_to_element = data['idx_to_element']
    vocab_size = data['vocab_size']
    
    # Define the model class
    class StatusSequencePredictor(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, status_dim):
            super(StatusSequencePredictor, self).__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            self.status_encoder = torch.nn.Linear(status_dim, embedding_dim)
            self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.fc = torch.nn.Linear(hidden_dim + embedding_dim, vocab_size)
            
        def forward(self, sequence, status):
            embeds = self.embedding(sequence)
            lstm_out, _ = self.lstm(embeds)
            lstm_out = lstm_out[:, -1, :]
            status_encoded = self.status_encoder(status)
            combined = torch.cat([lstm_out, status_encoded], dim=1)
            output = self.fc(combined)
            return output
    
    # Initialize the model
    model = StatusSequencePredictor(vocab_size, embedding_dim=128, hidden_dim=256, status_dim=2)
    
    # Load the state dict
    model.load_state_dict(torch.load('status_sequence_model.pth'))
    model.eval()
    
    return model, sequences, status_info, element_to_idx, idx_to_element

# Function to prepare sequence for model input
def prepare_sequence(seq, element_to_idx):
    """Convert a sequence of element lists to tensor of indices."""
    indices = []
    for element_list in seq:
        element_tuple = tuple(sorted(element_list))
        indices.append(element_to_idx[element_tuple])
    return torch.tensor(indices, dtype=torch.long)

# Function to prepare status information for model input
def prepare_status(status, normalize=True):
    """Convert status dictionary to tensor."""
    age = status['age']
    gender = status['gender']
    
    if normalize:
        age = (age - 18) / (65 - 18)
    
    return torch.tensor([age, gender], dtype=torch.float)

# Function to format a sequence for display
def format_sequence(sequence):
    """Format a sequence with nested elements for display."""
    formatted = []
    for element_list in sequence:
        if len(element_list) == 0:
            formatted.append("_")
        elif len(element_list) == 1:
            if element_list[0] == "":
                formatted.append("_")
            else:
                formatted.append(element_list[0])
        else:
            formatted.append(f"[{','.join(element_list)}]")
    return "→".join(formatted)

# Function to format status for display
def format_status(status):
    """Format status information for display."""
    gender = "男性" if status['gender'] == 1 else "女性"
    return f"年齢：{status['age']}歳/性別：{gender}"

# Function for inference
def predict_next_element(model, sequence, status, element_to_idx, idx_to_element):
    """Predict the next element in a sequence given status information."""
    with torch.no_grad():
        sequence_tensor = prepare_sequence(sequence, element_to_idx).unsqueeze(0)
        status_tensor = prepare_status(status).unsqueeze(0)
        
        output = model(sequence_tensor, status_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        _, predicted_idx = torch.max(output, 1)
        predicted_element = idx_to_element[predicted_idx.item()]
        
        return predicted_element, {idx_to_element[i]: prob.item() for i, prob in enumerate(probabilities)}

# Run comprehensive inference tests
def run_status_comparison_tests():
    """Run tests comparing predictions with different status combinations."""
    # Load model and data
    model, sequences, status_info, element_to_idx, idx_to_element = load_model_and_data()
    
    # Define test sequences
    test_sequences = [
        [['A'], ['C'], ['D']],                      # Basic sequence
        [['A'], [''], ['D']],                       # Sequence with blank
        [['A', 'B'], ['C']],                        # Sequence with nested input
        [['A'], ['B'], ['']],                       # Sequence ending with blank
        [['A', 'C'], ['E'], ['G']],                 # Longer sequence with nested input
        [['A'], [''], ['F'], ['H']],                # Longer sequence with blank
        [['A', 'B', 'C'], ['D', 'E'], ['F']],       # Complex sequence with nested inputs
        [['A'], ['C'], ['E', 'F']],                 # Sequence with nested element at the end
        [['A'], ['B'], ['C'], ['D'], ['E']],        # Longer basic sequence
        [['A', 'D'], [''], ['G', 'H']]              # Complex sequence with blank and nested elements
    ]
    
    # Define test status combinations
    test_statuses = [
        {'age': 25, 'gender': 1},  # 25-year-old male
        {'age': 25, 'gender': 0},  # 25-year-old female
        {'age': 45, 'gender': 1},  # 45-year-old male
        {'age': 45, 'gender': 0},  # 45-year-old female
    ]
    
    print("# ステータス情報による予測の違い\n")
    
    for i, seq in enumerate(test_sequences):
        print(f"## テストシーケンス {i+1}: {format_sequence(seq)}→?\n")
        
        # Store predictions for each status
        predictions = {}
        
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
            
            status_key = format_status(status)
            predictions[status_key] = {
                'element': next_element,
                'formatted': formatted_prediction,
                'probability': probabilities[next_element],
                'top5': sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
            }
            
            print(f"### {status_key}\n")
            print(f"予測された次の要素: **{formatted_prediction}** (確率: {probabilities[next_element]*100:.2f}%)\n")
            
            print("上位予測の確率:")
            for element, prob in predictions[status_key]['top5']:
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
        
        # Compare predictions across different statuses
        print("### ステータスによる予測の違い\n")
        
        # Check if predictions differ by status
        predictions_differ = len(set(p['element'] for p in predictions.values())) > 1
        
        if predictions_differ:
            print("ステータスによって予測が変化しています。\n")
            
            # Compare specific differences
            for status1, pred1 in predictions.items():
                for status2, pred2 in predictions.items():
                    if status1 != status2 and pred1['element'] != pred2['element']:
                        print(f"- {status1}の場合: **{pred1['formatted']}** (確率: {pred1['probability']*100:.2f}%)")
                        print(f"- {status2}の場合: **{pred2['formatted']}** (確率: {pred2['probability']*100:.2f}%)")
                        print()
        else:
            print("このシーケンスではステータスによる予測の違いはありませんでした。\n")
        
        print("\n" + "="*50 + "\n")

# Run the tests
if __name__ == "__main__":
    run_status_comparison_tests()
