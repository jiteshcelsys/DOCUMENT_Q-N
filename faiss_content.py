import faiss
import numpy as np

# Load the index file
index = faiss.read_index("faiss_index\index.faiss")

# View basic metadata
print(f"Number of vectors: {index.ntotal}")
print(f"Vector dimensions: {index.d}")
print(f"Is trained: {index.is_trained}")

# For Flat indexes, you can directly access the stored vectors
if hasattr(index, 'xb'):
    # Retrieve the first 5 vectors
    vectors = faiss.vector_to_array(index.xb).reshape(index.ntotal, index.d)
    print(vectors[:5])

# Retrieve all 1221 vectors (from index 0 to index.ntotal)
# # This returns a numpy array of shape (1221, 384)
# vectors = index.reconstruct_n(0, index.ntotal)

# # Print the first vector to see the content
# print("--- First Vector Content ---")
# print(vectors[0])

# # Print a slice of the first vector (first 10 numbers)
# print("\n--- First 10 dimensions of the first vector ---")
# print(vectors[0][:10])

# # Optional: Save to a CSV if you want to inspect it in Excel
# # np.savetxt("my_vectors.csv", vectors, delimiter=",")
