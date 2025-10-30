# dataset_downloader.py
import kagglehub
import shutil
import os

def download_catvnoncat(destination_folder="datasets"):
    """
    Downloads the 'cat vs non-cat' dataset from Kaggle using kagglehub
    and copies the HDF5 files into the destination folder directly.
    """
    # Step 1: Download the dataset via kagglehub
    path = kagglehub.dataset_download("muhammeddalkran/catvnoncat")
    print("Downloaded dataset from Kaggle:")
    print("   ", path)

    # Step 2: Find the actual folder containing the .h5 files
    nested_folder = os.path.join(path, "catvnoncat")
    if os.path.exists(nested_folder):
        path = nested_folder  # use inner folder if present

    # Step 3: Copy files directly into the destination folder
    os.makedirs(destination_folder, exist_ok=True)

    for item in os.listdir(path):
        source_item = os.path.join(path, item)
        dest_item = os.path.join(destination_folder, item)

        # Only copy .h5 files
        if item.endswith(".h5"):
            shutil.copy2(source_item, dest_item)
            print(f"Copied: {item}")

    print(f"Dataset files copied to: {os.path.abspath(destination_folder)}")
    return os.path.abspath(destination_folder)

dataset_path = download_catvnoncat()