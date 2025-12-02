import torch
import numpy as np
import os
import sys
import argparse

# We need to add the current directory to sys.path so imports work if run from encoder/
sys.path.append(os.getcwd())

# Import project modules
# Note: We delay importing config.configurator until after we parse our own args
# because it might try to parse args itself. However, we can just let it parse.

def dump():
    # 1. Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--weight_file', type=str, required=True, help='Path to .pth weight file')
    parser.add_argument('--cuda', type=str, default='0', help='CUDA device index')
    
    # We use parse_known_args to allow other args that might be picked up by configurator later if needed,
    # though here we are driving the process.
    args, unknown = parser.parse_known_args()

    # 2. Setup Environment for Configurator
    # The configurator parses args from sys.argv. We need to ensure it sees what we want.
    # It looks for --model, --dataset, --cuda.
    # We can modify sys.argv to ensure compatibility or rely on it picking up the same flags.
    # Since we use the same flag names, it should work fine.
    
    from config.configurator import configs
    from models.bulid_model import build_model
    from data_utils.build_data_handler import build_data_handler
    from trainer.trainer import init_seed

    # Force device config if needed
    if torch.cuda.is_available():
        configs['device'] = torch.device(f'cuda:{args.cuda}')
    else:
        configs['device'] = torch.device('cpu')
    
    print(f"Model: {configs['model']['name']}")
    print(f"Dataset: {configs['data']['name']}")
    print(f"Device: {configs['device']}")

    # 3. Load Data
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()

    # 4. Build Model
    # This will use the config loaded by configurator
    model = build_model(data_handler).to(configs['device'])

    # 5. Load Weights
    print(f"Loading weights from: {args.weight_file}")
    if not os.path.exists(args.weight_file):
        print(f"Error: Weight file not found at {args.weight_file}")
        return

    # Checkpoint loading logic from Trainer
    # The Trainer saves: torch.save(model_state_dict, path)
    # So we load state_dict directly.
    state_dict = torch.load(args.weight_file, map_location=configs['device'])
    model.load_state_dict(state_dict)
    model.eval()

    # 6. Extract Embeddings
    # Based on inspection, attributes are 'item_embeds' and 'user_embeds' (nn.Parameter)
    if hasattr(model, 'item_embeds'):
        item_emb = model.item_embeds.detach().cpu().numpy()
        user_emb = model.user_embeds.detach().cpu().numpy()
    else:
        print("Error: Model does not have 'item_embeds' attribute. Checking for 'item_emb'...")
        if hasattr(model, 'item_emb'):
             # Some implementations might use embedding layer
             item_emb = model.item_emb.weight.detach().cpu().numpy()
             user_emb = model.user_emb.weight.detach().cpu().numpy()
        else:
            print("Error: Could not find embedding attributes.")
            return

    # 7. Save
    save_dir = f"saved_emb/{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)
    
    item_save_path = os.path.join(save_dir, f"{args.model}_item.npy")
    user_save_path = os.path.join(save_dir, f"{args.model}_user.npy")
    
    np.save(item_save_path, item_emb)
    np.save(user_save_path, user_emb)
    
    print(f"Successfully saved embeddings to:")
    print(f"  {item_save_path}")
    print(f"  {user_save_path}")

if __name__ == "__main__":
    dump()
