import os
import hashlib
from typing import Dict, List, Any
from pathlib import Path

def calculate_hashes(file_path: str) -> Dict[str, str]:
    """Calculate MD5, SHA1, SHA256, SHA512 hashes for a file."""
    hashes = {
        'md5': hashlib.md5(),
        'sha1': hashlib.sha1(),
        'sha256': hashlib.sha256(),
        'sha512': hashlib.sha512()
    }
    
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                for h in hashes.values():
                    h.update(chunk)
        
        return {k: v.hexdigest() for k, v in hashes.items()}
    except Exception as e:
        print(f"Error hashing {file_path}: {e}")
        return {}

def scan_directory(directory: str) -> List[Dict[str, Any]]:
    """Scan directory and return list of file fingerprints."""
    results = []
    base_path = Path(directory)
    
    if not base_path.exists():
        raise FileNotFoundError(f"Directory {directory} not found")

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # Skip .git directories and other common ignores
            if '.git' in file_path or '__pycache__' in file_path:
                continue
                
            relative_path = os.path.relpath(file_path, directory)
            hashes = calculate_hashes(file_path)
            
            if hashes:
                results.append({
                    'path': relative_path,
                    'hashes': hashes,
                    'size': os.path.getsize(file_path)
                })
    
    return results
