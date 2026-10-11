
from pathlib import Path
import torch

checkpoint_path = Path("D:\\Projects\\Wonder fund\\submission\\baseline_gru.pt")

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

print("Checkpoint type:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("Top-level keys:", list(checkpoint.keys())[:30])

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if isinstance(state_dict, dict):
        print("\nModel parameter names and shapes:")
        for name, value in state_dict.items():
            if torch.is_tensor(value):
                print(name, tuple(value.shape))
