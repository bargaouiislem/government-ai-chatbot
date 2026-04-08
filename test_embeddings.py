import pickle

with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())
print(len(data["documents"]))
print(data["embeddings"].shape)