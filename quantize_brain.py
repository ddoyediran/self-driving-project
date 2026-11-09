from onnxruntime.quantization import quantize_dynamic, QuantType
import os

print("Starting Selective INT8 Quantization...")

model_input = "dinov2_encoder.onnx"
model_output = "dinov2_encoder_quantized.onnx"

if not os.path.exists(model_input):
    print(f"Error: Could not find {model_input}")
    exit()

print(f"Original Model Size: {os.path.getsize(model_input) / (1024*1024):.2f} MB")

# --- THE FIX ---
# We explicitly tell ONNX to ONLY quantize the heavy weights (MatMul, Gemm, Conv).
# This leaves the sensitive mathematical operators (like Resize/Bicubic) in 32-bit Float!
quantize_dynamic(
    model_input=model_input,
    model_output=model_output,
    weight_type=QuantType.QUInt8,
    op_types_to_quantize=['MatMul', 'Gemm', 'Conv'] 
)

print(f"Quantized Model Size: {os.path.getsize(model_output) / (1024*1024):.2f} MB")
print(f"Success! Generated: {model_output}")
print("You can now transfer this safe, quantized file to the Raspberry Pi 5!")