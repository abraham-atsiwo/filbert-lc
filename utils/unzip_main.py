import zipfile
import os

# Path to the zip file
zip_file_path = "src.zip"

# Directory to extract files to
extract_to_path = "./"

# Ensure the extraction path exists
os.makedirs(extract_to_path, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)

print(f"Extracted all files to {extract_to_path}")