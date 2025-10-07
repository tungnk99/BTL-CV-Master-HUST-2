import pickle

# Open the file in binary mode
with open("data/cifar-10-batches-py/batches.meta", "rb") as f:
    data = pickle.load(f)

# Inspect keys/values
print(data)