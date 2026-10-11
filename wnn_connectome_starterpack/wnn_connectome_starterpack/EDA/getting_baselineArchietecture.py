from pathlib import Path
import onnx

model_path = Path("D:\\Projects\\Wonder fund\\wnn_connectome_starterpack\\wnn_connectome_starterpack\\baseline\\baseline.onnx")
model = onnx.load(str(model_path))

print("=== MODEL GRAPH ===")
print("IR version:", model.ir_version)
print("Opset imports:", [(x.domain, x.version) for x in model.opset_import])

print("\n=== INPUTS ===")
for value in model.graph.input:
    dims = [
        d.dim_value if d.dim_value else d.dim_param
        for d in value.type.tensor_type.shape.dim
    ]
    print(value.name, dims)

print("\n=== OUTPUTS ===")
for value in model.graph.output:
    dims = [
        d.dim_value if d.dim_value else d.dim_param
        for d in value.type.tensor_type.shape.dim
    ]
    print(value.name, dims)

print("\n=== OPERATORS ===")
from collections import Counter
print(Counter(node.op_type for node in model.graph.node))

print("\n=== INITIALIZER PARAMETERS ===")
total = 0
for tensor in model.graph.initializer:
    count = 1
    for dim in tensor.dims:
        count *= dim
    total += count
    print(tensor.name, list(tensor.dims), "parameters:", count)

print("Total initializer parameters:", total)


