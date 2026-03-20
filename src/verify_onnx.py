import onnxruntime as ort
import numpy as np
import sys
import os

def verify_onnx_model(model_path: str):
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return

    print(f"Loading ONNX model from {model_path}...")
    
    try:
        session = ort.InferenceSession(model_path)
        
        # Get input metadata
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"Input Name: {input_name}, Shape: {input_shape}")
        
        # Create dummy input matching the shape (Batch Size=1, Input Size=100)
        # Note: input_shape[0] might be dynamic (string 'batch_size'), so we use 1
        input_size = input_shape[1] if isinstance(input_shape[1], int) else 100
        dummy_input = np.random.randn(1, input_size).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {input_name: dummy_input})
        
        print("Inference successful!")
        print(f"Output Shape: {outputs[0].shape}")
        print(f"Output Vector (First 5): {outputs[0][0][:5]}")
        
    except Exception as e:
        print(f"ONNX Verification Failed: {e}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "fingerprinter.onnx"
    verify_onnx_model(path)
