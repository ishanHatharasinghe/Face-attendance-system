import cv2
import face_recognition
import os
import pickle
import hashlib

def get_image_hash(image_path):
    """පින්තූරයක අන්තර්ගතය අනුව Unique ID එකක් හදනවා"""
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def train_system(verbose=True):
    """
    Enhanced training function with:
    - Smart Sync: Remove deleted images/folders from model
    - Hash-based encoding: Skip already encoded images
    - Cleanup: Remove encodings for deleted student folders
    """
    path = 'images'
    os.makedirs(path, exist_ok=True)
    model_file = "trained_face_model.pkl"
    
    # Load existing model data if available
    existing_encodings = []
    existing_names = []
    existing_hashes = []
    existing_image_paths = []
    
    if os.path.exists(model_file):
        try:
            with open(model_file, "rb") as f:
                data = pickle.load(f)
                existing_encodings = data.get("encodings", [])
                existing_names = data.get("names", [])
                existing_hashes = data.get("hashes", [])
                existing_image_paths = data.get("image_paths", [])
            if verbose:
                print(f"📂 Loaded existing model with {len(existing_hashes)} images")
        except Exception as e:
            if verbose:
                print(f"⚠️ Could not load existing model: {e}")
    else:
        if verbose:
            print("✨ Creating new model from scratch")
    
    # Step 1: Scan current images directory
    current_images = {}  # {folder_name: [image_paths]}
    current_hashes = set()  # All current image hashes
    current_folder_names = set()  # All current folder names
    
    for root, dirs, files in os.walk(path):
        folder_name = os.path.basename(root)
        if folder_name == 'images':
            continue  # Skip the root images folder itself
            
        folder_images = []
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)
                folder_images.append(img_path)
                current_hashes.add(get_image_hash(img_path))
                current_folder_names.add(folder_name)
        
        if folder_images:
            current_images[folder_name] = folder_images
    
    total_folders = len(current_images)
    
    if verbose:
        print(f"🔍 Found {total_folders} student folders to process")
        print("--- 🧠 Scanning for new or modified images ---")
    
    # Step 2: Smart Sync - Identify what to keep, add, and remove
    # Build a map of existing data by image path for efficient lookup
    existing_by_path = {}
    for i, img_path in enumerate(existing_image_paths):
        if img_path in existing_by_path:
            existing_by_path[img_path].append(i)
        else:
            existing_by_path[img_path] = [i]
    
    # Indices to keep from existing data (images that still exist and haven't changed)
    indices_to_keep = set()
    new_encodings = []
    new_names = []
    new_hashes = []
    new_image_paths = []
    new_count = 0
    kept_count = 0
    
    # Step 3: Process each folder
    for folder_name, image_paths in current_images.items():
        if verbose:
            print(f"⚡ Processing: {folder_name}")
        
        for img_path in image_paths:
            img_hash = get_image_hash(img_path)
            
            # Check if this exact image (by hash) already exists in model
            if img_hash in existing_hashes and img_path in existing_by_path:
                # Keep existing encoding for this image
                for idx in existing_by_path[img_path]:
                    indices_to_keep.add(idx)
                kept_count += 1
            else:
                # New or modified image - needs encoding
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        if verbose:
                            print(f"⚠️ Could not read: {img_path}")
                        continue
                    
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb)
                    
                    if encodings:
                        new_encodings.append(encodings[0])
                        new_names.append(folder_name.upper().strip())
                        new_hashes.append(img_hash)
                        new_image_paths.append(img_path)
                        new_count += 1
                        if verbose:
                            print(f"✅ Encoded: {folder_name} ({os.path.basename(img_path)})")
                except Exception as e:
                    if verbose:
                        print(f"❌ Error processing {img_path}: {e}")
    
    # Step 4: Cleanup - Remove encodings for deleted folders and images
    indices_to_remove = set()
    removed_names = set()
    removed_count = 0
    
    for i, (name, img_path) in enumerate(zip(existing_names, existing_image_paths)):
        if i not in indices_to_keep:
            indices_to_remove.add(i)
            removed_names.add(name)
            removed_count += 1
    
    # Build final data by keeping only valid indices
    final_encodings = [existing_encodings[i] for i in range(len(existing_encodings)) if i not in indices_to_remove]
    final_names = [existing_names[i] for i in range(len(existing_names)) if i not in indices_to_remove]
    final_hashes = [existing_hashes[i] for i in range(len(existing_hashes)) if i not in indices_to_remove]
    final_image_paths = [existing_image_paths[i] for i in range(len(existing_image_paths)) if i not in indices_to_remove]
    
    # Add new encodings
    final_encodings.extend(new_encodings)
    final_names.extend(new_names)
    final_hashes.extend(new_hashes)
    final_image_paths.extend(new_image_paths)
    
    # Step 5: Save updated model
    data = {
        "encodings": final_encodings,
        "names": final_names,
        "hashes": final_hashes,
        "image_paths": final_image_paths
    }
    
    with open(model_file, "wb") as f:
        pickle.dump(data, f)
    
    # Print summary
    if verbose:
        print("\n" + "="*50)
        print("🔥 TRAINING COMPLETE!")
        print("="*50)
        print(f"📈 Kept existing: {kept_count} images")
        print(f"🆕 Newly encoded: {new_count} images")
        print(f"🗑️ Removed: {removed_count} (deleted/old)")
        print(f"📊 Total in model: {len(final_encodings)} encodings")
        
        if removed_names:
