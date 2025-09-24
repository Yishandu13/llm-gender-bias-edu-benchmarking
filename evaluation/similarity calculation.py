# Install necessary packages
!pip install pingouin openai pandas scipy numpy openpyxl tqdm matplotlib statsmodels seaborn

# Import required libraries
from scipy.stats import f_oneway
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
embeddings_a = np.load("embeddings_a.npy")
embeddings_b = np.load("embeddings_b.npy")

# Check shapes
print(f"Group A shape: {embeddings_a.shape}")
print(f"Group B shape: {embeddings_b.shape}")

'''
If you have more than 2 groups of embedding to evaluate, please load them
embeddings_c = np.load("embeddings_c.npy")
print(f"Group C shape: {embeddings_c.shape}")
...
'''


# 1. Pairwise Similarity Assessment
def assess_similarity(matrix1, matrix2):
    """Compute cosine similarity and Euclidean distance between aligned vectors"""
    cosine_sim = np.array([np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
                      for v1,v2 in zip(matrix1, matrix2)])
    euclidean_dist = np.linalg.norm(matrix1 - matrix2, axis=1)

    return pd.DataFrame({
        'cosine_similarity': cosine_sim,
        'euclidean_distance': euclidean_dist
    })


# 2. Enhanced Permutation Test with Original Plots
def permutation_test(groups, group_names, n_permutations=5000, metric='cosine', seed=42):
    np.random.seed(seed)
    n_groups = len(groups)
    combined = np.vstack(groups)
    group_sizes = [len(g) for g in groups]
    full_perm_stats = {}
    observed_stats = {}
    pairwise_results = {}

    for i in range(n_groups):
        for j in range(i+1, n_groups):
            key = f"{group_names[i]}-{group_names[j]}"


            obs_stat = np.mean(cdist(groups[i], groups[j], metric=metric))
            perm_stats = []


            for _ in tqdm(range(n_permutations), desc=f"Permuting {key}", leave=False):
                perm_indices = np.random.permutation(len(combined))
                perm_group1 = combined[perm_indices[:group_sizes[i]]]
                perm_group2 = combined[perm_indices[group_sizes[i]:group_sizes[i]+group_sizes[j]]]
                perm_stats.append(np.mean(cdist(perm_group1, perm_group2, metric=metric)))

            perm_stats = np.array(perm_stats)
            full_perm_stats[key] = perm_stats


            p_value = np.mean(np.abs(perm_stats - np.mean(perm_stats)) >= np.abs(obs_stat - np.mean(perm_stats)))


            if metric == 'cosine':
                paired_dists = 1 - np.sum(groups[i] * groups[j], axis=1) / (
                    np.linalg.norm(groups[i], axis=1) * np.linalg.norm(groups[j], axis=1))
            else:
                paired_dists = np.linalg.norm(groups[i] - groups[j], axis=1)

            paired_std = np.std(paired_dists, ddof=1)
            effect_size = (obs_stat - np.mean(perm_stats)) / paired_std

            traditional_z = (obs_stat - np.mean(perm_stats)) / np.std(perm_stats)

            if metric == 'cosine':
                all_dists = 1 - cosine_similarity(groups[i], groups[j]).flatten()
                within_dists = []
                for g in [i, j]:
                    within_dists.extend(1 - cosine_similarity(groups[g]).flatten())
            else:
                all_dists = cdist(groups[i], groups[j], metric=metric).flatten()
                within_dists = []
                for g in [i, j]:
                    within_dists.extend(squareform(pdist(groups[g], metric=metric)).flatten())

            cohen_d = pg.compute_effsize(all_dists, within_dists, eftype='cohen')

            pairwise_results[key] = {
                'observed': obs_stat,
                'perm_mean': np.mean(perm_stats),
                'p_value': p_value,
                'paired_effect_size': effect_size,
                'effect_size': traditional_z,
                'cohen_d': cohen_d,
                'perm_dist': perm_stats,
                'std_used': 'paired_std'
            }


            plt.figure(figsize=(10, 6))
            plt.hist(perm_stats, bins=50, alpha=0.7, label='Permutation Distribution')
            plt.axvline(obs_stat, color='red', linestyle='--', linewidth=2, label='Observed')
            plt.axvline(np.mean(perm_stats), color='green', linestyle=':', linewidth=2, label='Perm Mean')
            plt.legend()
            plt.title(f"Permutation Test: {group_names[i]} vs {group_names[j]} ({metric}, n_perm={n_permutations})")
            plt.xlabel("Distance" if metric != 'cosine' else "1 - Cosine Similarity")
            plt.ylabel("Frequency")
            plt.show()

    return {
        'observed_stats': observed_stats,
        'pairwise_results': pairwise_results,
        'full_perm_stats': full_perm_stats
    }

def print_results(results_dict, metric_name):
    print(f"\n--- Pairwise Results for {metric_name.upper()} ---")
    for key, res in results_dict['pairwise_results'].items():
        print(f"\n{key}")
        print(f"Observed Mean Distance: {res['observed']:.4f}")
        print(f"Permutation Mean: {res['perm_mean']:.4f}")
        print(f"p-value: {res['p_value']:.4f}")
        print(f"Pooled Effect Size (Z): {res['paired_effect_size']:.4f}")
        print(f"Traditional Effect Size (Z): {res['effect_size']:.4f}")
        print(f"Cohen's d: {res['cohen_d']:.4f}")


# 3. Enhanced Visualization Functions
def plot_similarity_distributions(groups, group_names):
    """Plot similarity/distance distributions for all pairs"""
    # Create all pairwise combinations
    pairs = [(i,j) for i in range(len(groups)) for j in range(i+1, len(groups))]

    for metric in ['cosine', 'euclidean']:
        plt.figure(figsize=(15, 5*len(pairs)))
        for idx, (i,j) in enumerate(pairs, 1):
            # Compute similarities
            if metric == 'cosine':
                values = cosine_similarity(groups[i], groups[j]).diagonal()
                xlabel = "Cosine Similarity"
            else:
                values = np.linalg.norm(groups[i] - groups[j], axis=1)
                xlabel = "Euclidean Distance"

            # Plot
            plt.subplot(len(pairs), 2, 2*idx-1)
            sns.histplot(values, bins=30, kde=True)
            plt.title(f"{group_names[i]} vs {group_names[j]} {xlabel} Distribution")
            plt.xlabel(xlabel)

            plt.subplot(len(pairs), 2, 2*idx)
            sns.boxplot(x=values)
            plt.title(f"{group_names[i]} vs {group_names[j]} {xlabel} Spread")
            plt.xlabel(xlabel)

        plt.tight_layout()
        plt.show()

def plot_multi_group_comparison(results_dict, metric):
    comparisons = list(results_dict['pairwise_results'].keys())
    obs_values = [res['observed'] for res in results_dict['pairwise_results'].values()]
    effect_sizes = [res['effect_size'] for res in results_dict['pairwise_results'].values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.barplot(x=comparisons, y=obs_values, hue=comparisons, palette="viridis", ax=ax1, legend=False)
    ax1.set_title(f"Observed {metric} Statistics")
    ax1.set_ylabel("Mean Distance" if metric == 'euclidean' else "1 - Mean Cosine Similarity")

    sns.barplot(x=comparisons, y=effect_sizes, hue=comparisons, palette="magma", ax=ax2, legend=False)
    ax2.set_title(f"Effect Sizes ({metric})")
    ax2.axhline(0.2, color='red', linestyle='--', alpha=0.5, label='Small effect')
    ax2.axhline(0.5, color='red', linestyle=':', alpha=0.5, label='Medium effect')
    ax2.legend()

    plt.tight_layout()
    plt.show()


# 4. Main Analysis Pipeline
if __name__ == "__main__":
    groups = [embeddings_a, embeddings_b]
    group_names = ['A', 'B']

  '''
  when compare more than 2 groups
    groups = [embeddings_a, embeddings_b, embeddings_c]
    group_names = ['A', 'B', 'C']
  '''

    # (1) Pairwise distribution visualization
    print("\n=== Pairwise Distribution Visualizations ===")
    plot_similarity_distributions(groups, group_names)

    # (2) Run permutation tests with original plots
    print("\n=== Cosine Similarity Analysis ===")
    cos_results = permutation_test(groups, group_names, metric='cosine')

    '''
    When comparing three or more groups, use the following code to display a multi-group comparison chart.
    '''
    plot_multi_group_comparison(cos_results, 'cosine')

    print("\n=== Euclidean Distance Analysis ===")
    euc_results = permutation_test(groups, group_names, metric='euclidean')
    '''
    When comparing three or more groups, use the following code to display a multi-group comparison chart.
    '''
    plot_multi_group_comparison(euc_results, 'euclidean')

    print_results(cos_results, 'cosine')
    print_results(euc_results, 'euclidean')


    # (3) ANOVA and post-hoc tests
    print("\n=== ANOVA (Equivalent to T-test for 2 groups) ===")
    for metric in ['cosine', 'euclidean']:
        data = []
        for i in range(len(groups)):
            if metric == 'cosine':
                dists = 1 - cosine_similarity(groups[i]).flatten()
            else:
                dists = squareform(pdist(groups[i], 'euclidean')).flatten()
            data.append(dists[~np.isclose(dists, 0)])

        f_val, p_val = f_oneway(*data)
        print(f"\n{metric.upper()} Distance:")
        print(f"F-statistic: {f_val:.2f}, p-value: {p_val:.4f}")

        if p_val < 0.05:
            print("\nPost-hoc Tukey HSD:")
            all_data = np.concatenate(data)
            group_labels = np.concatenate([[f"Group{group_names[i]}"]*len(arr)
                                         for i, arr in enumerate(data)])
            tukey = pairwise_tukeyhsd(all_data, group_labels)
            print(tukey)
