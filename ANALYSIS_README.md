# RLMRec: Project Analysis & Guide


## Part 1: The Basics (For "Non-Experts")

### 1. What is the "Graph"?
In this project, we are building a **Recommender System** using a method called **LightGCN**. To understand it, imagine a giant network (a Graph):
*   **Nodes (Dots):** There are two types of dots.
    *   **Users:** The people (e.g., User A, User B).
    *   **Items:** The books on Amazon (e.g., Book #4607).
*   **Edges (Lines):** A line connects a User to an Item if that user **bought** or **rated** that book.

We call this a **Bipartite Graph** because lines only go from Users to Items (users don't "buy" other users).

### 2. What are "Inputs" and "Outputs"?
*   **Input:**
    1.  The **Adjacency Matrix**: A giant spreadsheet showing which User is connected to which Book.
    2.  **Initial IDs**: At the very start, every User and every Book is assigned a random list of numbers (e.g., `[0.1, -0.5, 0.03, ...]`). This list is called an **Embedding Vector**.
    3.  **User & Item Semantic Profiles**: We also have text descriptions for every user (generated from their history) and every item (from the book description).
*   **Output:**
    *   **Trained Embeddings**: Refined lists of numbers for every User and Book.
    *   **Prediction**: To see if User A likes Book B, we calculate the "Dot Product" (similarity) of their vectors. High score = Good recommendation.

### 3. What exactly are we "Training"?
We are not training a neural network with weights/biases like a CNN. **We are training the Embeddings themselves.**
*   The "Parameters" of our model are literally the vector coordinates for every user and book.
*   **LightGCN process:** It updates a book's vector by looking at *who bought it*. If User A and User B both bought "Harry Potter," the LightGCN algorithm pulls User A, User B, and "Harry Potter" closer together in the vector space.


## Part 2: Clarification - What is "Ground Truth"?

It is important to distinguish between two types of "truth" in this project:

1.  **Recommendation Ground Truth (User Behavior):**
    *   *What is it?* The actual historical data of **who bought what**.
    *   *Usage:* This is used to train the model (train set) and calculate accuracy metrics like Recall/NDCG (test set).
    *   *Note:* Our analysis assumes the model is already reasonably accurate at this.

2.  **Semantic Reference (Item Content):**
    *   *What is it?* The actual **meaning** of the book (e.g., "This is a Romance novel"). We use the **LLM Text Embeddings** as a proxy for this.
    *   *Usage:* This is used in our **Analysis** to see if the model understands *what* it is recommending, not just *who* buys it.
    *   *Why it matters:* A model can have good accuracy (by memorizing patterns) but poor semantic understanding. We want **both**.


## Part 3: The Problem & The Solution

### The Problem: "ID Embeddings are Blind"
*   **Scenario:** Imagine a new book comes out. It has an ID (e.g., Item #9999).
*   **The Issue:** The model only knows this is "Item #9999". It has **zero idea** that the title is "Advanced Python Programming".
*   **Consequence:** Until hundreds of people buy it (creating Edges), the model creates a random vector for it. It might accidentally recommend it to someone who likes cooking books because it's just guessing based on ID patterns.

### The Solution: RLMRec (Alignment)
The goal is **Alignment**: forcing the ID Space (User Behavior) and the Text Space (Content) to agree with each other.

**Method 1: Contrastive Alignment (Our Hero)**
*   **How it works:** It pulls the ID vector of a book closer to its own Text vector and pushes it away from other books' Text vectors.
*   **Effect:** It "injects" semantic meaning directly into the ID training.

**Method 2: Generative Alignment**
*   **How it works:** It tries to *reconstruct* the Text vector from the ID vector (like translating a language).
*   **Effect:** It forces the ID vector to contain enough information to generate the semantic description.

### Key Concept: Mutual Denoising
*   **It's not just "Copying the LLM":** LLMs can also be noisy or hallucinate.
*   **The Benefit:** By training on **both** User Behavior (Graph) and Item Content (LLM) simultaneously, the model finds the "common truth" between them.
    *   If the Graph is noisy (accidental clicks), the Text helps correct it.
    *   If the Text is noisy (vague description), the Graph helps correct it.
    *   **Result:** A robust representation that is better than either source alone.


## Part 4: Interpreting the Visualizations

We performed 3 types of analysis to prove this works.

### 1. Cumulative Distribution Function (CDF)
*   **File:** `jaccard_cdf.png`
*   **What is it?** It shows the quality of recommendations.
    *   **X-Axis (Jaccard):** How much do the model's recommendations overlap with the "Semantic Reference Recommendations"? (0 = No overlap, 1 = Perfect).
    *   **Y-Axis:** Cumulative % of items.
*   **How to read:** Look for the line that is furthest to the **right**.
*   **Result:** The **Contrastive (Orange)** line is to the right of the Backbone (Blue). This means it has significantly better semantic understanding.

### 2. Alignment Trend
*   **File:** `alignment_trend.png`
*   **What is it?** It answers: *"If two books are actually similar (Text), does the model know they are similar (ID)?"*
*   **X-Axis:** True Text Similarity (Semantic Reference).
*   **Y-Axis:** Model's ID Similarity.
*   **Result:**
    *   **Backbone (Blue):** Flat/Low slope. Even if two books are almost identical in content, the model barely notices.
    *   **Contrastive (Orange):** Steep slope. As Text Similarity goes up, Model Similarity goes up. It has "learned" the relationship.

### 3. UMAP Clusters (The Map)
*   **File:** `umap_semantic_clusters.png`
*   **What is it?** We took all the books and colored them by their Genre (using the Text).
*   **Backbone:** The colors are mixed up like confetti. The model groups books by "who bought them," which is messy.
*   **Contrastive:** The colors start to separate into clumps. This visually proves the model organized the books by Genre.


## Part 5: Case Study (Concrete Example)

**File:** `case_study_3models.txt`

We looked at **Item 4607**, which is a book in the *Stephanie Plum* mystery series.

| Model | What did it recommend? | Verdict |
| :--- | :--- | :--- |
| **Backbone** | Random popular books, maybe a cookbook or a thriller. | **Fail.** It doesn't know the series relationship. |
| **Generative** | *Metro Girl*, *Heat Wave*, *Twelve Sharp*. (Mix of same author and similar genre). | **Good.** It understands the author/genre vibe but missed some specific series links. |
| **Contrastive** | *Hot Six*, *Twelve Sharp*, *To the Nines* (All Stephanie Plum books). | **Excellent.** It perfectly understood the series connection. |
| **Semantic Reference** | *Hot Six*, *Eleven on Top*, *Lean Mean Thirteen*. | (This is what the LLM sees). |


## Part 6: How to Run Everything

We have created a **One-Click Script** to generate all results for all datasets and models automatically.

### 1. One-Click Execution
```bash
python run_all_analysis.py
```
This script will:
1.  Scan your `RLMRec_results/my_weights/` folder.
2.  Identify all trained models (LightGCN, SGL, DCCF, etc.) and datasets (Amazon, Yelp, Steam).
3.  Extract embeddings for every model variant (Backbone, Contrastive, Generative).
4.  Run the full analysis suite (CDF, Trend, UMAP, Case Study) for each.

### 2. Where are the results?
The results are organized by folders in `analysis_results/`. For example:
*   `analysis_results/amazon_lightgcn/`: Results for LightGCN on Amazon.
*   `analysis_results/yelp_sgl/`: Results for SGL on Yelp.
*   `analysis_results/steam_gccf/`: Results for GCCF on Steam.

Inside each folder, you will find:
*   `alignment_trend.png`
*   `jaccard_cdf.png`
*   `umap_semantic_clusters.png`
*   `case_study_3models.txt`

### 3. Individual Manual Run
If you only want to run one specific analysis:
```bash
# Dump embeddings
python encoder/dump_emb.py --model lightgcn --dataset amazon --weight_file ...

# Run Analysis
python analysis.py --dataset amazon --backbone lightgcn
python analysis_correlation.py --dataset amazon --backbone lightgcn
```