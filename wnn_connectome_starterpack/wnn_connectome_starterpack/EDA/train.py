import numpy as np 
import pyarrow.parquet as pq
import torch 
import torch.nn as nn 
import pandas as pd
from pathlib import Path

# writing the configs 
TRAIN_PATH = "D:\\Projects\\Wonder fund\\wnn_connectome_starterpack\\wnn_connectome_starterpack\\datasets\\train.parquet"
MODEL_PATH = str(Path(__file__).resolve().parents[1] / "submission" / "baseline_gru.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using for training:",DEVICE)

#Defining the necessary placeholders 
LEARNING_RATE = 0.001 
HIDDEN_SIZE = 128
NUM_LAYERS = 1
N_TRAIN_SEQ = 500
EPOCHS = 5
SEED = 42    #FOR THE REPRODUCIBILITY OF THE OUTPUT 
CHUNK_SIZE = 1024     # to devide the sequence into smaller chunks for training and validation



def get_feature_columns():
    cols = []
    
    for instrument in ["i0","i1"]:
        cols += [f"{instrument}_p{i}" for i in range(22)]
        cols += [f"{instrument}_v{i}" for i in range(22)]
        cols += [f"{instrument}_dp{i}" for i in range(4)]
        cols += [f"{instrument}_dv{i}" for i in range(4)]
    
    cols += [f"a{i}" for i in range(8)]
    
    return cols

INPUT_COLS = get_feature_columns();
OUTPUT_COLS = ["t0","t1"]


# now writing the baseline GRU i.e model ka archietecture

class BaselineGRU(nn.Module):
    def __init__(
        self,
        input_size = 112,
        hidden_size = 128,
        num_layers = 1,
    ):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size= input_size,
            hidden_size= hidden_size,
            num_layers= num_layers,
            batch_first= True,
        )
        
        self.head = nn.Linear(hidden_size, 2)
        
    def forward(self, x, hidden =None):
        output, hidden = self.gru(x, hidden)
        prediction = self.head(output)
        
        return prediction, hidden
    
def load_sequence(parquet_file, row_group_idx):
    table = parquet_file.read_row_group(
        row_group_idx,
        columns = INPUT_COLS + OUTPUT_COLS,
    )
    
    df = table.to_pandas()
    
    X = df[INPUT_COLS].to_numpy(
        dtype = np.float32,
        copy = True,
    )
    
    y = df[OUTPUT_COLS].to_numpy(
        dtype = np.float32,
        copy = True
    )
    
    del table, df
    return X,y

#model ko ek sequence par train karne ka procedure.

def train_one_sequence(model, optimizer,X,y):
    X = torch.from_numpy(X).unsqueeze(0).to(DEVICE)
    y = torch.from_numpy(y).unsqueeze(0).to(DEVICE)    #unsqueez(0) isliye kyuki apna batch_first = true hai and so gru input lega (B,T,F) but hai (T,F) so unsqueez se ek add ho jayega 0 index pr as a batch so it will become (1,20000,112)
    
    #warm up ke  99 steps 
    with torch.no_grad():
        _, hidden = model(X[:, :99, :])    #_ ek conventional variable name hai jo yahan indicate karta hai ki is returned value ko filhaal use nahi karna.
        
        
    total_loss = 0.0
    n_chunks = 0
    
    for i in range (99, X.shape[1], CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, X.shape[1])
        
        x_chunk = X[:, i:end, :]
        y_chunk = y[:, i:end, :]
        
        prediction , hidden = model(x_chunk, hidden)
        loss = nn.functional.mse_loss(prediction, y_chunk)
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   #gradient clipping to prevent exploding gradients
        optimizer.step()
        
        total_loss += loss.item()
        n_chunks += 1
        
        hidden = hidden.detach()

    return total_loss / max(1, n_chunks)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    parquet_file = pq.ParquetFile(TRAIN_PATH)
    total_seq = parquet_file.num_row_groups
    
    rng = np.random.default_rng(SEED)
    selected_ids = rng.choice(
        total_seq,
        size=N_TRAIN_SEQ,
        replace=False,
    )

    selected_ids = sorted(selected_ids.tolist())

    print("Total sequences:", total_seq)
    print("Selected sequences:", len(selected_ids))
    print("Model:", "GRU(112 -> 128) -> Linear(128 -> 2)")

    model = BaselineGRU(
        input_size=len(INPUT_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    for epoch in range(EPOCHS):
        model.train()
        sequence_losses = []

        for count, row_group_idx in enumerate(selected_ids, start=1):
            X, y = load_sequence(parquet_file, row_group_idx)

            loss = train_one_sequence(model, optimizer, X, y)
            sequence_losses.append(loss)

            if count % 10 == 0 or count == len(selected_ids):
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Sequences {count}/{len(selected_ids)} | "
                    f"Mean chunk MSE: "
                    f"{np.mean(sequence_losses):.6f}"
                )

            del X, y

        print(
            f"Epoch {epoch + 1} complete. "
            f"Mean sequence loss: "
            f"{np.mean(sequence_losses):.6f}"
        )    
        
    torch.save(
    {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    },
        MODEL_PATH,
    )
    
    # Export trained GRU model to ONNX
    model.eval()

    ONNX_PATH = MODEL_PATH.replace(".pt", ".onnx")

    dummy_x = torch.zeros(1, 1, 112, device=DEVICE)
    dummy_hidden = torch.zeros(
        NUM_LAYERS, 1, HIDDEN_SIZE, device=DEVICE
    )

    torch.onnx.export(
        model,
        (dummy_x, dummy_hidden),
        ONNX_PATH,
        input_names=["features", "hidden"],
        output_names=["prediction", "next_hidden"],
        dynamic_axes={
            "features": {1: "sequence_length"},
            "prediction": {1: "sequence_length"},
        },
        opset_version=17,
    )

    print("ONNX model saved to:", ONNX_PATH)

    print("\nTraining complete.")
    print("Saved weights to:", MODEL_PATH)


if __name__ == "__main__":
    main()   