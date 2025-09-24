!pip install pandas numpy scikit-learn seaborn matplotlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

# upload data
embeddings_a = np.load("embeddings_a.npy")
embeddings_b = np.load("embeddings_b.npy")

...
embedding_pairs = [
    (embeddings_a, embeddings_b, 'M_gpt5mini', 'M-F_gpt5mini'),
...    ]

# Combine all the data and perform a unified t-SNE projection.
all_embeddings = []
for a_emb, b_emb, a_name, b_name in embedding_pairs:
    all_embeddings.append(a_emb)
    all_embeddings.append(b_emb)

combined_all = np.vstack(all_embeddings)


tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
reduced_all = tsne.fit_transform(combined_all)



start_idx = 0
fig, axes = plt.subplots(2, 3, figsize=(18, 12)) 
axes = axes.flatten()

for i, (a_emb, b_emb, a_name, b_name) in enumerate(embedding_pairs):
    a_size = len(a_emb)
    b_size = len(b_emb)

    a_tsne = reduced_all[start_idx:start_idx + a_size]
    b_tsne = reduced_all[start_idx + a_size:start_idx + a_size + b_size]

    start_idx += a_size + b_size
    current_tsne = np.vstack([a_tsne, b_tsne])
    current_labels = np.concatenate([[a_name] * a_size, [b_name] * b_size])
    ax = axes[i]
    palette = sns.color_palette("husl", 2)
    color_dict = {a_name: palette[0], b_name: palette[1]}

    sns.scatterplot(x=current_tsne[:, 0], y=current_tsne[:, 1],
                    hue=current_labels, palette=color_dict, alpha=0.7, s=60, ax=ax)

    for group in [a_name, b_name]:
        group_data = current_tsne[np.array(current_labels) == group]
        if len(group_data) > 1:
            sns.kdeplot(
                x=group_data[:, 0], y=group_data[:, 1],
                levels=2, color=color_dict[group], linewidths=1.5, alpha=0.8,
                ax=ax, label=f"{group} density"
            )

    ax.set_title(f"t-SNE: {a_name} vs {b_name}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# t-SNE evaluation metrics
n_samples = combined_all.shape[0]
k = min(15, int(np.sqrt(n_samples)))
trust = trustworthiness(combined_all, reduced_all, n_neighbors=k)
knn_orig = NearestNeighbors(n_neighbors=k).fit(combined_all).kneighbors(return_distance=False)
knn_emb = NearestNeighbors(n_neighbors=k).fit(reduced_all).kneighbors(return_distance=False)
preserved = np.mean([
    len(np.intersect1d(knn_orig[i], knn_emb[i])) / k
    for i in range(n_samples)
])

print("\n t-SNE Quality Metrics (All Data):")
print(f"- KL Divergence: {tsne.kl_divergence_:.4f}")
print(f"- Trustworthiness (k={k}): {trust:.4f}")
print(f"- k-NN Preservation Rate (k={k}): {preserved:.4f}")


# extract representative text from selected region
TSNE1_RANGE = (x, x)     # set range of t-SNE 1
TSNE2_RANGE = (x, x)  # set range of t-SNE 2
MAX_SAMPLES = x

def extract_texts_in_tsne_region(reduced_tsne, texts_all, group_labels,
                                  tsne1_range, tsne2_range, max_samples=10):
    tsne1_min, tsne1_max = tsne1_range
    tsne2_min, tsne2_max = tsne2_range

    mask = (
        (reduced_tsne[:, 0] >= tsne1_min) & (reduced_tsne[:, 0] <= tsne1_max) &
        (reduced_tsne[:, 1] >= tsne2_min) & (reduced_tsne[:, 1] <= tsne2_max)
    )
    indices = np.where(mask)[0]
    selected_indices = indices[:max_samples]

    result = []
    for idx in selected_indices:
        result.append({
            "t-SNE 1": reduced_tsne[idx, 0],
            "t-SNE 2": reduced_tsne[idx, 1],
            "Group": group_labels[idx],
            "Text": texts_all[idx]

        })

    return pd.DataFrame(result)

df_region = extract_texts_in_tsne_region(
    reduced_tsne=reduced_tsne,
    texts_all=texts_all,
    group_labels=group_labels,
    tsne1_range=TSNE1_RANGE,
    tsne2_range=TSNE2_RANGE,
    max_samples=MAX_SAMPLES
)

from IPython.display import display
display(df_region)

df_region.to_excel("tsne_selected_region.xlsx", index=False)
