import pickle

# Load the preprocessed data
with open('preprocessed_data.pickle', 'rb') as f:
    data = pickle.load(f)

# Print the type and structure of the data
print(type(data))
print(data)
