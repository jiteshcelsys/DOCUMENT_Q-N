import faiss
import pickle
import os

# Define paths
# Using join ensures it works on Windows correctly
faiss_path = os.path.join("faiss_index", "index.faiss")
pkl_path = os.path.join("faiss_index", "index.pkl") 

# 1. Load the FAISS index (The math/vectors)
if os.path.exists(faiss_path):
    index = faiss.read_index(faiss_path)
    print(f"FAISS Index Loaded: {index.ntotal} vectors found.")
    
    # Show the first vector
    vectors = index.reconstruct_n(0, 1)
    print("First Vector sample:", vectors[0][:5]) # first 5 dimensions
else:
    print(f"Could not find FAISS file at {faiss_path}")

# 2. Load the PKL file (Usually contains the text/metadata)
if os.path.exists(pkl_path):
    with open(pkl_path, "rb") as f:
        metadata = pickle.load(f)
    print(f"Metadata Loaded. Type: {type(metadata)}")
    # If it's a list or dict, print the first entry
    print("Metadata Sample:", str(metadata)[:200]) 
else:
    print(f"Could not find Pickle file at {pkl_path}")
    
# Assuming 'metadata' is the tuple you just loaded
docstore = metadata[0]  # The InMemoryDocstore object
index_to_id = metadata[1] # The dictionary mapping index (0, 1, 2...) to UUIDs

print("\n--- Mapping Vectors to Text ---")

# Let's look at the first 3 entries
for i in range(3):
    # 1. Get the UUID for this index
    chunk_id = index_to_id[i]
    
    # 2. Look up the document in the docstore
    document = docstore.search(chunk_id)
    
    if document:
        print(f"\n[Vector {i}] (ID: {chunk_id})")
        # Print the first 200 characters of the text chunk
        print(f"Content: {document.page_content[:200]}...")
    else:
        print(f"Could not find document for ID: {chunk_id}")