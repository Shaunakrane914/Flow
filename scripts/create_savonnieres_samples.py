"""
Create sample Savonnières chunks for UI demo
Extracts 20 sample chunks from the full rock and saves as .npy files
"""

import numpy as np
import os

# Load full rock
print("Loading Savonnières rock...")
full_rock = np.fromfile("data/SAVII2_mid_1000x1000x1000x8b-3phase-cleaner.dat", dtype=np.uint8)
full_rock = full_rock.reshape((1000, 1000, 1000))

# Convert to binary (0=pore based on earlier analysis)
pore_label = 0
binary_rock = (full_rock == pore_label).astype(np.uint8)

print(f"Global porosity: {np.mean(binary_rock):.2%}")

# Create output directory
os.makedirs("data/savonnieres_chunks", exist_ok=True)

# Extract sample chunks (same positions as successful graphs)
CHUNK_SIZE = 128
chunk_positions = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 1),
    (0, 2, 2),
    (1, 1, 3),
    (1, 2, 4),
    (2, 2, 3),
    (2, 3, 4),
    (3, 3, 3),
    (3, 4, 5),
    (4, 4, 4),
    (4, 5, 5),
    (5, 5, 5),
    (5, 6, 6),
    (6, 6, 6),
    (0, 3, 6),
    (1, 4, 5),
    (2, 5, 6),
    (3, 5, 4),
    (4, 6, 5)
]

print(f"\nExtracting {len(chunk_positions)} sample chunks...")

for i, (ix, iy, iz) in enumerate(chunk_positions):
    x = ix * CHUNK_SIZE
    y = iy * CHUNK_SIZE
    z = iz * CHUNK_SIZE
    
    # Extract chunk
    chunk = binary_rock[x:x+CHUNK_SIZE, y:y+CHUNK_SIZE, z:z+CHUNK_SIZE].copy()
    
    # Calculate porosity
    phi = np.mean(chunk)
    
    # Save
    filename = f"savonnieres_{i:02d}_phi{phi:.3f}.npy"
    filepath = os.path.join("data/savonnieres_chunks", filename)
    np.save(filepath, chunk)
    
    print(f"  {i+1:2d}. {filename} - Porosity: {phi:.3f}")

print(f"\n✅ Created {len(chunk_positions)} Savonnières chunks in data/savonnieres_chunks/")
print("These can now be uploaded to the Streamlit app!")
