import torch
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the trained model and data
def load_model_and_data():
    """Load the trained model and data from files."""
    # Load the saved data
    with open('family_training_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    sequences = data['sequences']
    status_info = data['status_info']
    family_structures = data['family_structures']
    element_to_idx = data['element_to_idx']
    idx_to_element = data['idx_to_element']
    vocab_size = data['vocab_size']
    
    # Define the model class
    class FamilyStatusSequencePredictor(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, status_dim):
            super(FamilyStatusSequencePredictor, self).__init__()
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
    status_dim = 2 + len(family_structures)  # Age, gender, and one-hot encoded family structure
    model = FamilyStatusSequencePredictor(vocab_size, embedding_dim=128, hidden_dim=256, status_dim=status_dim)
    
    # Load the state dict
    model.load_state_dict(torch.load('family_sequence_model.pth'))
    model.eval()
    
    return model, sequences, status_info, family_structures, element_to_idx, idx_to_element

# Function to prepare sequence for model input
def prepare_sequence(seq, element_to_idx):
    """Convert a sequence of element lists to tensor of indices."""
    indices = []
    for element_list in seq:
        # Sort the elements in the list to ensure consistent representation
        element_tuple = tuple(sorted(element_list))
        indices.append(element_to_idx[element_tuple])
    return torch.tensor(indices, dtype=torch.long)

# Function to prepare status information for model input
def prepare_family_status(status, family_structures, normalize=True):
    """Convert status dictionary to tensor including family structure."""
    # Extract status values
    age = status['age']
    gender = status['gender']
    family = status['family']
    
    # Normalize age to [0, 1] range if requested
    if normalize:
        age = (age - 18) / (65 - 18)
    
    # One-hot encode family structure
    family_idx = family_structures.index(family)
    family_one_hot = [0] * len(family_structures)
    family_one_hot[family_idx] = 1
    
    # Create tensor: [age, gender, family_one_hot...]
    status_tensor = [age, gender] + family_one_hot
    
    return torch.tensor(status_tensor, dtype=torch.float)

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
def format_family_status(status):
    """Format status information including family structure for display."""
    gender = "男性" if status['gender'] == 1 else "女性"
    
    # Translate family structure to Japanese
    family_jp = {
        "single": "独身",
        "married_no_kids": "既婚・子供なし",
        "married_with_kids": "既婚・子供あり",
        "single_parent": "シングルペアレント",
        "extended_family": "拡大家族（親と同居など）"
    }
    
    family = family_jp.get(status['family'], status['family'])
    
    return f"年齢：{status['age']}歳/性別：{gender}/家族：{family}"

# Function for inference
def predict_next_element(model, sequence, status, family_structures, element_to_idx, idx_to_element):
    """Predict the next element in a sequence given status information including family structure."""
    with torch.no_grad():
        # Convert sequence to tensor
        sequence_tensor = prepare_sequence(sequence, element_to_idx).unsqueeze(0)
        
        # Convert status to tensor
        status_tensor = prepare_family_status(status, family_structures).unsqueeze(0)
        
        # Get model prediction
        output = model(sequence_tensor, status_tensor)
        
        # Get probabilities for all elements
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Get the most likely next element
        _, predicted_idx = torch.max(output, 1)
        predicted_element = idx_to_element[predicted_idx.item()]
        
        # Return the predicted element and all probabilities
        return predicted_element, {idx_to_element[i]: prob.item() for i, prob in enumerate(probabilities)}

# Run comprehensive inference tests
def run_family_comparison_tests():
    """Run tests comparing predictions with different family structure combinations."""
    # Load model and data
    model, sequences, status_info, family_structures, element_to_idx, idx_to_element = load_model_and_data()
    
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
        {'age': 25, 'gender': 1, 'family': 'single'},                # 25-year-old single male
        {'age': 25, 'gender': 0, 'family': 'single'},                # 25-year-old single female
        {'age': 35, 'gender': 1, 'family': 'married_no_kids'},       # 35-year-old married male without kids
        {'age': 35, 'gender': 0, 'family': 'married_no_kids'},       # 35-year-old married female without kids
        {'age': 40, 'gender': 1, 'family': 'married_with_kids'},     # 40-year-old married male with kids
        {'age': 40, 'gender': 0, 'family': 'married_with_kids'},     # 40-year-old married female with kids
        {'age': 45, 'gender': 1, 'family': 'single_parent'},         # 45-year-old single father
        {'age': 45, 'gender': 0, 'family': 'single_parent'},         # 45-year-old single mother
        {'age': 60, 'gender': 1, 'family': 'extended_family'},       # 60-year-old male in extended family
        {'age': 60, 'gender': 0, 'family': 'extended_family'},       # 60-year-old female in extended family
    ]
    
    print("# 家族構成による予測の違い\n")
    
    for i, seq in enumerate(test_sequences):
        print(f"## テストシーケンス {i+1}: {format_sequence(seq)}→?\n")
        
        # Group predictions by family structure
        family_predictions = {}
        
        for status in test_statuses:
            next_element, probabilities = predict_next_element(
                model, seq, status, family_structures, element_to_idx, idx_to_element
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
            
            print(f"### {format_family_status(status)}\n")
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
            
            # Store prediction by family structure
            family = status['family']
            if family not in family_predictions:
                family_predictions[family] = []
            
            family_predictions[family].append({
                'status': status,
                'element': next_element,
                'formatted': formatted_prediction,
                'probability': probabilities[next_element]
            })
        
        # Compare predictions across different family structures
        print("### 家族構成による予測の違い\n")
        
        # Check if predictions differ by family structure
        all_predictions = [pred['element'] for family_preds in family_predictions.values() for pred in family_preds]
        predictions_differ = len(set(all_predictions)) > 1
        
        if predictions_differ:
            print("家族構成によって予測が変化しています。\n")
            
            # Compare predictions between different family structures
            for family1, preds1 in family_predictions.items():
                for family2, preds2 in family_predictions.items():
                    if family1 != family2:
                        # Check if predictions differ between these family structures
                        for pred1 in preds1:
                            for pred2 in preds2:
                                if pred1['element'] != pred2['element']:
                                    status1 = format_family_status(pred1['status'])
                                    status2 = format_family_status(pred2['status'])
                                    print(f"- {status1}の場合: **{pred1['formatted']}** (確率: {pred1['probability']*100:.2f}%)")
                                    print(f"- {status2}の場合: **{pred2['formatted']}** (確率: {pred2['probability']*100:.2f}%)")
                                    print()
                                    break
                            else:
                                continue
                            break
        else:
            print("このシーケンスでは家族構成による予測の違いはありませんでした。\n")
        
        print("\n" + "="*50 + "\n")

# Run the tests
if __name__ == "__main__":
    run_family_comparison_tests()
