import pickle
import random
import string

# Define the sequence generation function directly to avoid importing the entire model
def generate_random_sequences(num_sequences=1000, min_length=3, max_length=10, blank_prob=0.1, multi_input_prob=0.15):
    """
    Generate random sequences that follow alphabetical order (A to Z) with possible skips.
    The sequences never go backwards in the alphabet.
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

# Set the same random seed as in training
random.seed(42)

# Generate the same sequences as in training
sequences = generate_random_sequences(1000)

# Save the training data to a file
with open('training_data.pkl', 'wb') as f:
    pickle.dump(sequences, f)

print(f"Saved {len(sequences)} training sequences to training_data.pkl")

# Print some example sequences
print("\nExample sequences:")
for i in range(10):
    print(f"Sequence {i+1}: {'→'.join(['_' if s == '' else s for s in sequences[i]])}")
