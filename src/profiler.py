from typing import List, Dict, Any
import os
import json
import numpy as np

try:
    from sklearn.feature_extraction.text import CountVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.onnx
    from .model import create_model
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

FIXED_INPUT_SIZE = 100

def extract_features(text: str) -> Dict[str, int]:
    """Extract simple features from text content."""
    keywords = ['import', 'def', 'class', 'return', 'if', 'else', 'for', 'while']
    counts = {}
    for kw in keywords:
        counts[kw] = text.count(kw)
    return counts

def export_onnx_model(model, input_size, path="fingerprinter.onnx"):
    """Export the PyTorch model to ONNX format."""
    dummy_input = torch.randn(1, input_size)
    torch.onnx.export(
        model, 
        dummy_input, 
        path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {path}")

def generate_profile(fingerprints: List[Dict[str, Any]], directory: str) -> Dict[str, Any]:
    """
    Generate a ML-ready profile from fingerprints.
    """
    profile = {
        'total_files': len(fingerprints),
        'extensions': {},
        'code_features': {},
        'fingerprints': fingerprints
    }
    
    code_contents = []
    
    for item in fingerprints:
        path = item['path']
        ext = os.path.splitext(path)[1]
        profile['extensions'][ext] = profile['extensions'].get(ext, 0) + 1
        
        if ext in ['.py', '.go', '.js', '.ts', '.c', '.cpp', '.java', '.rs']:
            full_path = os.path.join(directory, path)
            try:
                with open(full_path, 'r', errors='ignore') as f:
                    content = f.read()
                    profile['code_features'][path] = extract_features(content)
                    code_contents.append(content)
            except Exception:
                pass

    # 1. Vectorize Code (Bag of Words)
    project_vector = [0] * FIXED_INPUT_SIZE
    vocab_list = []
    
    if SKLEARN_AVAILABLE and code_contents:
        try:
            # We enforce a fixed size for the Neural Net
            vectorizer = CountVectorizer(max_features=FIXED_INPUT_SIZE, stop_words='english')
            X = vectorizer.fit_transform(code_contents)
            vocab = vectorizer.get_feature_names_out()
            
            # Sum features for project-level vector
            raw_sum = X.sum(axis=0).tolist()[0]
            
            # Pad if necessary (if vocab < FIXED_INPUT_SIZE)
            if len(raw_sum) < FIXED_INPUT_SIZE:
                raw_sum += [0] * (FIXED_INPUT_SIZE - len(raw_sum))
            
            project_vector = raw_sum[:FIXED_INPUT_SIZE] # Ensure exact size
            vocab_list = list(vocab)
            
            profile['ml_vector'] = {
                'vocabulary': vocab_list,
                'counts': project_vector
            }
        except Exception as e:
            profile['ml_error'] = str(e)

    # 2. Neural Net Embedding & ONNX Export
    if TORCH_AVAILABLE:
        try:
            # Initialize model
            model = create_model(input_size=FIXED_INPUT_SIZE)
            model.eval() # Set to eval mode
            
            # Create input tensor
            input_tensor = torch.tensor([project_vector], dtype=torch.float32)
            
            # Get embedding (fingerprint)
            with torch.no_grad():
                embedding = model(input_tensor)
                profile['neural_fingerprint'] = embedding.numpy().tolist()[0]
                
            # Export to ONNX
            export_onnx_model(model, FIXED_INPUT_SIZE)
            profile['onnx_model'] = "fingerprinter.onnx"
            
        except Exception as e:
            print(f"Torch/ONNX Error: {e}")
            profile['neural_error'] = str(e)
            
    return profile
