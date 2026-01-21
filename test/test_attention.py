from llm.attention import SimpleAttentionBlock
import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("testing")
    model  = SimpleAttentionBlock(d_model=128, num_heads=8).to(device)
    print("=" * 10, "Model Architecture", "=" * 10)
    print(model)
    print("=" * 10, "Model Initialization", "=" * 10)

    hidden_states = torch.randn(3, 10, 128).to(device)
    print("=" * 10, "Forward Pass", "=" * 10)
    output = model(hidden_states)
    print("=" * 10, "Forward Pass", "=" * 10)
    print(output)