import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run RLMRec Correlation Analysis")
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name')
    parser.add_argument('--backbone', type=str, default='lightgcn', help='Backbone model name')
    return parser.parse_args()

args = parse_args()

# Settings
DATASET = args.dataset
BACKBONE = args.backbone
K_NEIGHBORS = 20
SAMPLE_SIZE = 2000 
SEED = 2025

# Paths
EMB_PRE_PATH = f"saved_emb/{DATASET}/{BACKBONE}_item.npy"
EMB_CON_PATH = f"saved_emb/{DATASET}/{BACKBONE}_plus_item.npy"
EMB_GEN_PATH = f"saved_emb/{DATASET}/{BACKBONE}_gene_item.npy"
EMB_TXT_PATH = f"data/{DATASET}/itm_emb_np.pkl"
OUTPUT_DIR = f"analysis_results/{DATASET}_{BACKBONE}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    print(f"Loading embeddings for {DATASET} / {BACKBONE}...")
    if not os.path.exists(EMB_PRE_PATH): return None
    emb_pre = np.load(EMB_PRE_PATH)
    emb_con = np.load(EMB_CON_PATH)
    emb_gen = np.load(EMB_GEN_PATH)
    with open(EMB_TXT_PATH, "rb") as f:
        emb_txt = pickle.load(f)
    
    print(f"Shapes: Pre {emb_pre.shape}, Con {emb_con.shape}, Gen {emb_gen.shape}, Txt {emb_txt.shape}")
    return emb_pre, emb_con, emb_gen, emb_txt

def get_jaccard_scores(emb, emb_txt, k=100):
    # Recalculate Jaccard for CDF plot
    # Increased k to 100 to make the CDF smoother (more granular scores)
    print(f"Computing Neighbors for Jaccard CDF (k={k})...")
    # Limit to first 3000 to be fast
    limit = 3000
    emb = emb[:limit]
    emb_txt = emb_txt[:limit]
    
    nn_model = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(emb)
    nn_txt = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(emb_txt)
    
    _, idx_model = nn_model.kneighbors(emb)
    _, idx_txt = nn_txt.kneighbors(emb_txt)
    
    idx_model = idx_model[:, 1:]
    idx_txt = idx_txt[:, 1:]
    
    scores = []
    for i in range(len(emb)):
        s1 = set(idx_model[i])
        s2 = set(idx_txt[i])
        u = len(s1 | s2)
        i_sect = len(s1 & s2)
        scores.append(i_sect / u if u > 0 else 0)
    return np.array(scores)

def plot_cdf(scores_list, labels, filename):
    plt.figure(figsize=(8, 6))
    
    for scores, label in zip(scores_list, labels):
        sorted_scores = np.sort(scores)
        p = 1. * np.arange(len(sorted_scores)) / (len(sorted_scores) - 1)
        plt.plot(sorted_scores, p, label=f"{label} (Mean: {scores.mean():.3f})", linewidth=2.5)
        
    plt.xlabel("Jaccard Similarity (Top-100 Neighbors)")
    plt.ylabel("Cumulative Fraction of Items")
    plt.title("CDF of Semantic Alignment (Smoothed via k=100)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    print(f"Saved {filename}")

def plot_binned_trend(emb_models, emb_txt, labels, filename):
    print("Generating Binned Trend Plot...")
    
    # 1. Sample pairs
    # We need a lot of pairs to get stable means in high-sim bins
    # Sample 5000 items -> ~12M pairs. We can compute row-wise to save memory.
    
    # Let's take a random sample of items
    idx = np.random.choice(len(emb_txt), size=3000, replace=False)
    sub_txt = emb_txt[idx]
    
    # Compute Text Sim Upper Triangle
    sim_txt = cosine_similarity(sub_txt)
    mask = np.triu_indices_from(sim_txt, k=1)
    vals_txt = sim_txt[mask]
    
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'orange', 'green']
    
    # Define bins for Text Similarity
    # Text Sim usually clusters around 0-0.2, but we care about the trend up to 1.0
    bins = np.linspace(-0.1, 1.0, 22)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    for emb, label, color in zip(emb_models, labels, colors):
        sub_model = emb[idx]
        sim_model = cosine_similarity(sub_model)
        vals_model = sim_model[mask]
        
        # Binning
        # Digitizing
        inds = np.digitize(vals_txt, bins)
        
        means = []
        valid_centers = []
        
        for i in range(1, len(bins)):
            # Find pairs falling in this Text Sim bin
            matches = vals_model[inds == i]
            if len(matches) > 50: # Threshold for stability
                means.append(np.mean(matches))
                valid_centers.append(bin_centers[i-1])
        
        plt.plot(valid_centers, means, marker='o', label=label, color=color, linewidth=2.5)
        
    plt.xlabel("Text Similarity (Semantic Reference)")
    plt.ylabel("Average Model Similarity (ID Space)")
    plt.title("Alignment Trend: Does Semantic Similarity imply Model Similarity?")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    print(f"Saved {filename}")

def plot_high_sim_density(emb_models, emb_txt, labels, filename):
    print("Generating High-Sim Density Plot...")
    
    idx = np.random.choice(len(emb_txt), size=4000, replace=False)
    sub_txt = emb_txt[idx]
    sim_txt = cosine_similarity(sub_txt)
    mask = np.triu_indices_from(sim_txt, k=1)
    vals_txt = sim_txt[mask]
    
    # Filter: Only look at pairs that are Semantically Similar
    # Threshold: top 5-10% roughly, or absolute > 0.7
    threshold = 0.75
    high_sim_mask = vals_txt > threshold
    
    print(f"Found {np.sum(high_sim_mask)} pairs with Text Sim > {threshold}")
    
    plt.figure(figsize=(10, 6))
    
    for emb, label in zip(emb_models, labels):
        sub_model = emb[idx]
        sim_model = cosine_similarity(sub_model)
        vals_model = sim_model[mask]
        
        # Get the Model Sim for these specific high-text-sim pairs
        target_vals = vals_model[high_sim_mask]
        
        sns.kdeplot(target_vals, label=label, linewidth=2.5, fill=True, alpha=0.1)
        
    plt.xlabel(f"Model Similarity (for pairs with Text Sim > {threshold})")
    plt.ylabel("Density")
    plt.title(f"Distribution of Model Similarity for Semantically Related Items")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    print(f"Saved {filename}")

def run():
    data = load_data()
    if data is None:
        return
    emb_pre, emb_con, emb_gen, emb_txt = data
    min_len = min(len(emb_pre), len(emb_con), len(emb_gen), len(emb_txt))
    
    # Trim
    emb_pre = emb_pre[:min_len]
    emb_con = emb_con[:min_len]
    emb_gen = emb_gen[:min_len]
    emb_txt = emb_txt[:min_len]
    
    # 1. CDF Plot (Keep existing logic)
    jac_pre = get_jaccard_scores(emb_pre, emb_txt, k=100)
    jac_con = get_jaccard_scores(emb_con, emb_txt, k=100)
    jac_gen = get_jaccard_scores(emb_gen, emb_txt, k=100)
    
    plot_cdf([jac_pre, jac_con, jac_gen], 
             ["Backbone", "Contrastive", "Generative"], 
             "jaccard_cdf.png")
             
    # 2. New Trend Analysis (Replaces Hexbins)
    models = [emb_pre, emb_con, emb_gen]
    labels = ["Backbone", "Contrastive", "Generative"]
    
    plot_binned_trend(models, emb_txt, labels, "alignment_trend.png")
    # Density plot removed as per request (results were noisy/unclear)

if __name__ == "__main__":
    run()

if __name__ == "__main__":
    run()
