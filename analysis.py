import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import umap
import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run RLMRec Analysis")
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name (e.g., amazon, steam, yelp)')
    parser.add_argument('--backbone', type=str, default='lightgcn', help='Backbone model name (e.g., lightgcn, gccf)')
    return parser.parse_args()

args = parse_args()

# Settings
DATASET = args.dataset
BACKBONE = args.backbone
K_NEIGHBORS = 20
SAMPLE_SIZE_UMAP = 3000
SEED = 2025
NUM_CLUSTERS = 5

# Paths
# Assumes naming convention: backbone, backbone_plus, backbone_gene
EMB_PRE_PATH = f"saved_emb/{DATASET}/{BACKBONE}_item.npy"
EMB_CON_PATH = f"saved_emb/{DATASET}/{BACKBONE}_plus_item.npy"
EMB_GEN_PATH = f"saved_emb/{DATASET}/{BACKBONE}_gene_item.npy"
EMB_TXT_PATH = f"data/{DATASET}/itm_emb_np.pkl"
PRF_TXT_PATH = f"data/{DATASET}/itm_prf.pkl"

# Specific output dir for this run
OUTPUT_DIR = f"analysis_results/{DATASET}_{BACKBONE}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    print(f"Loading embeddings for {DATASET} / {BACKBONE}...")
    
    # Check if files exist
    if not os.path.exists(EMB_PRE_PATH):
        print(f"Error: Backbone embedding not found: {EMB_PRE_PATH}")
        return None
    if not os.path.exists(EMB_CON_PATH):
        print(f"Error: Contrastive embedding not found: {EMB_CON_PATH}")
        return None
    if not os.path.exists(EMB_GEN_PATH):
        print(f"Error: Generative embedding not found: {EMB_GEN_PATH}")
        return None

    emb_pre = np.load(EMB_PRE_PATH)
    emb_con = np.load(EMB_CON_PATH)
    emb_gen = np.load(EMB_GEN_PATH)
    
    with open(EMB_TXT_PATH, "rb") as f:
        emb_txt = pickle.load(f)
    
    with open(PRF_TXT_PATH, "rb") as f:
        prf_txt = pickle.load(f)
        
    print(f"Shapes: Pre {emb_pre.shape}, Con {emb_con.shape}, Gen {emb_gen.shape}, Txt {emb_txt.shape}")
    return emb_pre, emb_con, emb_gen, emb_txt, prf_txt

def get_neighbors(emb, k):
    print(f"Computing Nearest Neighbors (k={k})...")
    nn = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(emb)
    dist, idx = nn.kneighbors(emb)
    return idx[:, 1:] # Exclude self

def jaccard_similarity(a, b):
    s1 = set(a)
    s2 = set(b)
    return len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 0

def run_analysis():
    data = load_data()
    if data is None:
        return
    emb_pre, emb_con, emb_gen, emb_txt, prf_txt = data
    
    # Ensure shapes match
    min_len = min(len(emb_pre), len(emb_con), len(emb_gen), len(emb_txt))
    emb_pre = emb_pre[:min_len]
    emb_con = emb_con[:min_len]
    emb_gen = emb_gen[:min_len]
    emb_txt = emb_txt[:min_len]
    
    # 1. Neighbors & Jaccard
    nbr_pre = get_neighbors(emb_pre, K_NEIGHBORS)
    nbr_con = get_neighbors(emb_con, K_NEIGHBORS)
    nbr_gen = get_neighbors(emb_gen, K_NEIGHBORS)
    nbr_txt = get_neighbors(emb_txt, K_NEIGHBORS)
    
    jac_pre = []
    jac_con = []
    jac_gen = []
    
    print("Calculating Jaccard similarities...")
    for i in range(min_len):
        jac_pre.append(jaccard_similarity(nbr_pre[i], nbr_txt[i]))
        jac_con.append(jaccard_similarity(nbr_con[i], nbr_txt[i]))
        jac_gen.append(jaccard_similarity(nbr_gen[i], nbr_txt[i]))
        
    jac_pre = np.array(jac_pre)
    jac_con = np.array(jac_con)
    jac_gen = np.array(jac_gen)
    
    print(f"Mean Jaccard (Backbone vs Txt):   {jac_pre.mean():.4f}")
    print(f"Mean Jaccard (Contrastive vs Txt):{jac_con.mean():.4f}")
    print(f"Mean Jaccard (Generative vs Txt): {jac_gen.mean():.4f}")
    
    # Plot Jaccard Histogram (Step Style for clarity)
    plt.figure(figsize=(10, 6))
    # Focus on the range where data actually exists (usually < 0.5 for Jaccard)
    max_val = max(jac_pre.max(), jac_con.max(), jac_gen.max())
    bins = np.linspace(0, min(1.0, max_val + 0.1), 50)
    
    plt.hist(jac_pre, bins=bins, histtype='step', linewidth=2, label=f"Backbone (Mean: {jac_pre.mean():.3f})")
    plt.hist(jac_con, bins=bins, histtype='step', linewidth=2, label=f"Contrastive (Mean: {jac_con.mean():.3f})")
    plt.hist(jac_gen, bins=bins, histtype='step', linewidth=2, label=f"Generative (Mean: {jac_gen.mean():.3f})")
    
    plt.xlabel("Jaccard Similarity with Text Neighbors")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.title("Distribution of Semantic Alignment (Step Histogram)")
    plt.savefig(os.path.join(OUTPUT_DIR, "jaccard_hist_step.png"), dpi=200)
    print("Saved jaccard_hist_step.png")
    
    # 2. UMAP Visualization with Semantic Coloring
    print(f"Running Clustering & UMAP on {SAMPLE_SIZE_UMAP} samples...")
    
    # Sample indices
    idx = np.random.choice(min_len, size=min(SAMPLE_SIZE_UMAP, min_len), replace=False)
    
    # Step 1: Cluster the Text Space (Ground Truth Semantics)
    print("Clustering Text Space to find semantic categories...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=SEED, n_init=10)
    labels = kmeans.fit_predict(emb_txt[idx])
    
    # Step 2: Fit UMAP on ALL data to ensure shared coordinate system?
    # Actually, fitting on each space independently is often better to see *local* structure preservation,
    # BUT coloring them by the *Text* labels reveals if that structure matches semantics.
    # If ID space is random w.r.t semantics, colors will be mixed.
    # If ID space is aligned, colors will form clusters.
    
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
    
    Y_pre = reducer.fit_transform(emb_pre[idx])
    Y_con = reducer.fit_transform(emb_con[idx])
    Y_gen = reducer.fit_transform(emb_gen[idx])
    Y_txt = reducer.fit_transform(emb_txt[idx])
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    # Plot function
    def plot_emb(ax, Y, title):
        scatter = ax.scatter(Y[:, 0], Y[:, 1], s=3, c=labels, cmap='tab10', alpha=0.7)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return scatter

    plot_emb(axes[0], Y_pre, "Backbone (ID only)\nColors mixed = Poor Semantic Structure")
    plot_emb(axes[1], Y_con, "Contrastive (RLMRec-Con)\nBetter Color Separation?")
    plot_emb(axes[2], Y_gen, "Generative (RLMRec-Gen)\nBetter Color Separation?")
    plot_emb(axes[3], Y_txt, "Target Text Space\n(Semantic Reference Clusters)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "umap_semantic_clusters.png"), dpi=200)
    print("Saved umap_semantic_clusters.png")
    
    # 3. Case Study Update
    # Find item where Gen > Con > Pre or similar
    print("\n--- Case Study ---")
    improvement = jac_gen - jac_pre
    top_idx = np.argsort(improvement)[-3:]
    
    with open(os.path.join(OUTPUT_DIR, "case_study_3models.txt"), "w") as f:
        for i in top_idx:
            header = f"\nItem ID: {i}"
            print(header)
            f.write(header + "\n")
            
            # Profile
            profile_text = prf_txt.get(i, "N/A")
            prof_disp = (profile_text[:200] + '...') if len(profile_text) > 200 else profile_text
            f.write(f"Profile: {profile_text}\n")
            
            # Neighbors
            def format_neighbors(indices, prf_dict):
                res = []
                for idx in indices:
                    txt = prf_dict.get(idx, "N/A")
                    snippet = (txt[:50] + '...') if len(txt) > 50 else txt
                    res.append(f"[{idx}] {snippet}")
                return "\n    ".join(res)

            pre_n = format_neighbors(nbr_pre[i][:5], prf_txt)
            con_n = format_neighbors(nbr_con[i][:5], prf_txt)
            gen_n = format_neighbors(nbr_gen[i][:5], prf_txt)
            txt_n = format_neighbors(nbr_txt[i][:5], prf_txt)
            
            output = f"""
  Jaccard Scores: Backbone={jac_pre[i]:.3f}, Con={jac_con[i]:.3f}, Gen={jac_gen[i]:.3f}
            
  Neighbors (Backbone):
    {pre_n}
  Neighbors (Contrastive):
    {con_n}
  Neighbors (Generative):
    {gen_n}
  Neighbors (Text/Semantic Reference):
    {txt_n}
            """
            print(output)
            f.write(output + "\n")
            
    print(f"Case study saved to {os.path.join(OUTPUT_DIR, 'case_study_3models.txt')}")

if __name__ == "__main__":
    run_analysis()
