import os
import pickle

# Ensure the target directory exists
os.makedirs('data/raw/pickle/train', exist_ok=True)

# Create a sample data dictionary
sample_data = {
    'rooms': [
        {'type': 'bedroom', 'area': 120},
        {'type': 'kitchen', 'area': 80},
        {'type': 'bathroom', 'area': 50}
    ]
}

# Save the sample data as a pickle file
with open('data/raw/pickle/train/sample1.pickle', 'wb') as f:
    pickle.dump(sample_data, f)

print("Sample pickle file created at data/raw/pickle/train/sample1.pickle")