import pickle
import numpy as np

# Load the dataset
with open('dataset\dataset.pkl', 'rb') as file:
    dataset = pickle.load(file)

# Initialize lists to hold your data
notes = []
roots = []
styles = []

# Loop through each piece in the dataset
for piece_name, piece_data in dataset.items():
    nmat = np.array(piece_data['nmat'])  # Note matrix
    root = np.array(piece_data['root'])  # Root matrix
    style = piece_data['style']          # Style of the piece

    notes.append(nmat)
    roots.append(root)
    styles.append(style)

# Convert lists to numpy arrays for easier manipulation later on
notes = np.array(notes, dtype=object)  # Use dtype=object to accommodate arrays of varying lengths
roots = np.array(roots, dtype=object)
styles = np.array(styles)

# Example of accessing the first piece's data
print("First piece notes:", notes[0])
print("First piece roots:", roots[0])
print("First piece style:", styles[0])
import pickle
import numpy as np

# Load the dataset
with open('dataset\dataset.pkl', 'rb') as file:
    dataset = pickle.load(file)

# Initialize lists to hold your data
notes = []
roots = []
styles = []

# Loop through each piece in the dataset
for piece_name, piece_data in dataset.items():
    nmat = np.array(piece_data['nmat'])  # Note matrix
    root = np.array(piece_data['root'])  # Root matrix
    style = piece_data['style']          # Style of the piece

    notes.append(nmat)
    roots.append(root)
    styles.append(style)

# Convert lists to numpy arrays for easier manipulation later on
notes = np.array(notes, dtype=object)  # Use dtype=object to accommodate arrays of varying lengths
roots = np.array(roots, dtype=object)
styles = np.array(styles)

# Example of accessing the first piece's data
print("First piece notes:", notes[0])
print("First piece roots:", roots[0])
print("First piece style:", styles[0])
