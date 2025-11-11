"""
Comprehensive visualization of input protein trajectories.

This script randomly loads 5 protein trajectory from an LMDB database,
and visualizes the temporal dynamics of the protein.

The script uses the following features:
- Residue Temporal Evolution Heatmap
- Frame-to-Frame Dynamics
- Residue Temporal Activity
- Frame-to-Frame Correlation
- Amino Acid Flexibility Analysis
- Dynamic Pattern Analysis
- Temporal Memory
- Collective Motions
- Residue Communication Network
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add statistical analysis functions at the top
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import silhouette_score

# Add the project root to Python path
sys.path.append('/scratch/groups/rbaltman/ziyiw23/EMPROT')
from emprot.data.data_loader import LMDBLoader
from emprot.data.metadata import MetadataManager

# Amino acid properties for biological interpretation

AMINO_ACID_PROPERTIES = {
    'A': {'type': 'nonpolar (hydrophobic)', 'size': '88 (small)',  'flexibility': '0.360 (flexible)'},
    'V': {'type': 'nonpolar (hydrophobic)', 'size': '140 (small)', 'flexibility': '0.310 (rigid)'},
    'L': {'type': 'nonpolar (hydrophobic)', 'size': '166 (large)', 'flexibility': '0.365 (moderate)'},
    'I': {'type': 'nonpolar (hydrophobic)', 'size': '166 (large)', 'flexibility': '0.300 (rigid)'},
    'M': {'type': 'nonpolar (hydrophobic)', 'size': '162 (large)', 'flexibility': '0.295 (moderate)'},
    'F': {'type': 'nonpolar (hydrophobic)', 'size': '189 (large)', 'flexibility': '0.310 (rigid)'},
    'Y': {'type': 'polar (hydrophilic)',    'size': '193 (large)', 'flexibility': '0.310 (rigid)'},
    'W': {'type': 'nonpolar (hydrophobic)', 'size': '227 (large)', 'flexibility': '0.305 (rigid)'},
    'S': {'type': 'polar (hydrophilic)',    'size': '89 (small)',  'flexibility': '0.510 (flexible)'},
    'T': {'type': 'polar (hydrophilic)',    'size': '116 (small)', 'flexibility': '0.480 (flexible)'},
    'N': {'type': 'polar (hydrophilic)',    'size': '114 (small)', 'flexibility': '0.460 (flexible)'},
    'Q': {'type': 'polar (hydrophilic)',    'size': '143 (large)', 'flexibility': '0.370 (moderate)'},
    'C': {'type': 'polar (hydrophilic)',    'size': '108 (small)', 'flexibility': '0.350 (rigid)'},
    'G': {'type': 'nonpolar (hydrophobic)', 'size': '60 (small)',  'flexibility': '0.540 (very_flexible)'},
    'P': {'type': 'nonpolar (hydrophobic)', 'size': '112 (small)', 'flexibility': '0.310 (very_rigid)'},
    'R': {'type': 'positive (basic)',       'size': '173 (large)', 'flexibility': '0.530 (moderate)'},
    'H': {'type': 'positive (basic)',       'size': '153 (large)', 'flexibility': '0.395 (flexible)'},
    'K': {'type': 'positive (basic)',       'size': '168 (large)', 'flexibility': '0.500 (very_flexible)'},
    'D': {'type': 'negative (acidic)',      'size': '111 (small)', 'flexibility': '0.510 (flexible)'},
    'E': {'type': 'negative (acidic)',      'size': '138 (large)', 'flexibility': '0.500 (flexible)'},
}

# Source:
# Bhaskaran, R., & Ponnuswamy, P. K. (1988). Average flexibility index of amino acids in proteins. 
# Int. J. Pept. Protein Res., 32(4), 241–255. https://doi.org/10.1111/j.1399-3011.1988.tb01261.x
#
# Volume: https://www.researchgate.net/figure/Volumes-of-the-standard-20-amino-acids-ATP-and-AMP-Amino-acid-volumes-are-from-22_tbl1_342497620
# 
# Type: https://www.imgt.org/IMGTeducation/Aide-memoire/_UK/aminoacids/IMGTclasses.html


def load_lmdb(lmdb_path, num_frames):
    with LMDBLoader(lmdb_path) as loader:
        metadata = loader.get_metadata()
        print(metadata)
        print(f"Database: {metadata['num_frames']} frames, {metadata['num_residues']} residues")
        # Load embeddings and extract residue information
        actual_frames = min(num_frames, metadata['num_frames'])
        embeddings, times = loader.load_sequence(0, actual_frames)
        
        # Load first frame to get residue information
        first_frame = loader.load_frame(0)
        residue_info = extract_residue_info(first_frame)
    return metadata, embeddings, times, residue_info

def get_protein_info(lmdb_path, metadata_path = "traj_metadata.csv"):
    dynamic_id = os.path.basename(lmdb_path).split("_")[2]
    metadata = MetadataManager(metadata_path)
    protein_info = metadata.get_protein_info(dynamic_id)
    return protein_info

def evolution_heatmap(ax1, embeddings, times, n_frames, n_residues, global_dynamics):
    
    # Sample frames for visualization
    frame_sample = np.arange(0, n_frames, max(1, n_frames//100))
    
    heatmap_data = global_dynamics[frame_sample].T  # (residues, sampled_frames)
    im1 = ax1.imshow(heatmap_data, aspect='auto', cmap='RdBu_r',
                     extent=[times[frame_sample[0]], times[frame_sample[-1]], 0, n_residues])
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Residue Index')
    ax1.set_title('1. Residue Temporal Evolution\n(Average across all 512 dimensions)')
    plt.colorbar(im1, ax=ax1)

def frame_to_frame_dynamics(ax2, mean_frame_change, times):
    ax2.plot(times[1:], mean_frame_change, alpha=0.7, linewidth=1)
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Mean Frame-to-Frame Change')
    ax2.set_title('2. Temporal Dynamics\n(Frame-to-Frame Changes)')
    ax2.grid(True, alpha=0.3)
    mean_change = np.mean(mean_frame_change)
    std_change = np.std(mean_frame_change)
    ax2.axhline(y=mean_change, color='red', linestyle='--', alpha=0.7, 
                label=f'Mean: {mean_change:.3f}')
    ax2.axhline(y=mean_change + 2*std_change, color='orange', linestyle='--', alpha=0.5,
                label=f'Mean + 2(sigma): {mean_change + 2*std_change:.3f}')
    ax2.legend(fontsize=8)

def residue_temporal_activity(ax3, mean_temporal_activity, n_residues, residue_info):
    # Color by amino acid type if available
    if residue_info and 'amino_acids' in residue_info:
        amino_acids = residue_info['amino_acids'][:n_residues]
        colors = []
        for aa in amino_acids:
            if aa in AMINO_ACID_PROPERTIES:
                aa_type = AMINO_ACID_PROPERTIES[aa]['type']
                if aa_type == 'nonpolar (hydrophobic)':
                    colors.append('blue')
                elif aa_type == 'polar (hydrophilic)':
                    colors.append('green')
                elif aa_type == 'positive (basic)':
                    colors.append('red')
                elif aa_type == 'negative (acidic)':
                    colors.append('orange')
                else:
                    colors.append('gray')
            else:
                colors.append('gray')
        
        bars = ax3.bar(range(n_residues), mean_temporal_activity, color=colors, alpha=0.7)
        
        # Add legend
        handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.7) for c in ['blue', 'green', 'red', 'orange', 'gray']]
        labels = ['Hydrophobic', 'Polar', 'Positive', 'Negative', 'Other']
        ax3.legend(handles, labels, loc='upper right', fontsize=8)
    else:
        ax3.bar(range(n_residues), mean_temporal_activity, alpha=0.7)
    
    ax3.set_xlabel('Residue Index')
    ax3.set_ylabel('Temporal Activity')
    ax3.set_title('3. Residue Temporal Activity\n(Colored by Amino Acid Type)')

def comprehensive_protein_analysis(lmdb_path, num_frames=2500, ):
    """
    Comprehensive analysis of protein dynamics with biological insights.
    """
    print(f"Loading data from: {lmdb_path}")
    metadata, embeddings, times, residue_info = load_lmdb(lmdb_path, num_frames)

    n_frames, n_residues, n_dims = embeddings.shape
    print(f"Analyzing {n_frames} frames, {n_residues} residues, {n_dims} dimensions")

    protein_info = get_protein_info(lmdb_path)
    
    # Calculate basic temporal metrics
    temporal_activity = np.var(embeddings, axis=0)  # (residues, dims)
    mean_temporal_activity = np.mean(temporal_activity, axis=1)  # (residues,)
    
    # Average embeddings across all dimensions for global dynamics
    global_dynamics = np.mean(embeddings, axis=2)  # (frames, residues)
    
    # Frame-to-frame changes
    frame_changes = np.diff(global_dynamics, axis=0)  # (frames-1, residues)
    mean_frame_change = np.mean(np.abs(frame_changes), axis=1)  # (frames-1,)
    
    # Create comprehensive visualization (3x4 grid)
    fig = plt.figure(figsize=(24, 18))
    

    # 1. Residue Temporal Evolution Heatmap (enhanced)
    # each point on the plot represents the average of the 512 dimensions of a residue at that time point

    ax1 = plt.subplot(3, 4, 1)
    evolution_heatmap(ax1, embeddings, times, n_frames, n_residues, global_dynamics)
    
    # 2. Frame-to-Frame Dynamics
    # We first calculate the mean across the 512 embedding dimensions for each residue at each time point.
    # This gives us a vector of length n (number of residues) for each frame—representing the protein state at that frame.
    # Then, we compute the absolute difference between consecutive frames: |state_t+1 - state_t|.
    # We take the mean of that difference vector across all residues, resulting in a single scalar per time step.
    # This scalar represents the magnitude of conformational change between frame t and t+1, and is plotted at time t+1.


    ax2 = plt.subplot(3, 4, 2)
    frame_to_frame_dynamics(ax2, mean_frame_change, times)
    
    # 3. Residue Temporal Activity 
    # For each residue, we examine its 512-dimensional embedding across all time frames.
    # We calculate the variance of each embedding dimension over time, resulting in a 512D variance vector per residue.
    # Then we take the mean of that variance vector, giving a single scalar representing how much that residue's embedding fluctuates over time.
    # This scalar is plotted as a bar for each residue.
    # Bars are colored based on the amino acid type of the residue (hydrophobic, polar, positive, negative, other).

    ax3 = plt.subplot(3, 4, 3)
    residue_temporal_activity(ax3, mean_temporal_activity, n_residues, residue_info)
    
    # 4. Temporal Smoothness (Frame-to-Frame Correlation)
    # For each time point t, we compute the mean embedding across 512 dimensions for each residue,
    # resulting in an n-dimensional vector representing the protein state at frame t.
    # We then calculate the Pearson correlation between this vector and the one at frame t+1.
    # Each correlation value reflects how similar the residue-level activation pattern is between consecutive frames.
    # A correlation near 1.0 means the relative pattern of residue embeddings is preserved — the protein is evolving smoothly.
    # A correlation near 0.0 indicates no consistent pattern between frames, suggesting unstructured or noisy transitions.
    # Negative correlations imply the residue-level changes flipped direction — a sign of opposing motion or a sharp transition.
    # These values are plotted over time starting from t=1, helping identify regions of stability vs. transition.

    ax4 = plt.subplot(3, 4, 4)
    
    if n_frames >= 2:
        # Correlation between consecutive frames
        frame_correlations = []
        for i in range(n_frames - 1):
            corr = np.corrcoef(global_dynamics[i], global_dynamics[i + 1])[0, 1]
            frame_correlations.append(corr)
        
        ax4.plot(times[1:], frame_correlations, alpha=0.7, linewidth=1, color='purple')
        ax4.set_xlabel('Time (ns)')
        ax4.set_ylabel('Temporal Smoothness (Frame-to-Frame Correlation)')
        ax4.set_title('4. Temporal Smoothness\n(Consecutive Frame Correlation)')
        ax4.grid(True, alpha=0.3)
        
        # Add statistical annotations
        mean_corr = np.mean(frame_correlations)
        ax4.axhline(y=mean_corr, color='red', linestyle='--', alpha=0.7, 
                    label=f'Mean: {mean_corr:.3f}')
        ax4.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='High similarity (0.95)')
        ax4.legend(fontsize=8)
    
    # 5. Biochemical Validation (Activity vs Flexibility Group)
    # Each residue is assigned a known flexibility score based on its amino acid identity.
    # We group residues into four categories: Rigid (≤0.33), Moderate (0.34–0.39), Flexible (0.40–0.46), and Very Flexible (≥0.47).
    # For each group, we calculate the temporal activity of its residues — defined as the mean variance of their 512D embeddings over time.
    # These distributions are visualized as boxplots, with color indicating increasing flexibility.
    # A positive Spearman correlation (rho) between flexibility group and temporal activity suggests that more flexible amino acids tend to exhibit higher embedding variability,
    # supporting the biological relevance of the learned embeddings.
    ax5 = plt.subplot(3, 4, 5)

    def extract_flexibility_value(flex_label):
        return float(flex_label.split()[0])

    if residue_info and 'amino_acids' in residue_info:
        flexibility_bins = {
            'Rigid (≤0.33)': [],
            'Moderate (0.34–0.39)': [],
            'Flexible (0.40–0.46)': [],
            'Very Flexible (≥0.47)': []
        }

        amino_acids = residue_info['amino_acids'][:n_residues]
        for i, aa in enumerate(amino_acids):
            if aa in AMINO_ACID_PROPERTIES:
                flex_value = extract_flexibility_value(AMINO_ACID_PROPERTIES[aa]['flexibility'])
                activity = mean_temporal_activity[i]
                if flex_value <= 0.33:
                    flexibility_bins['Rigid (≤0.33)'].append(activity)
                elif flex_value <= 0.39:
                    flexibility_bins['Moderate (0.34–0.39)'].append(activity)
                elif flex_value <= 0.46:
                    flexibility_bins['Flexible (0.40–0.46)'].append(activity)
                else:
                    flexibility_bins['Very Flexible (≥0.47)'].append(activity)

        labels_for_plot = list(flexibility_bins.keys())
        data_for_plot = [flexibility_bins[k] for k in labels_for_plot]

        if data_for_plot:
            bp = ax5.boxplot(data_for_plot, tick_labels=labels_for_plot, patch_artist=True)
            colors = plt.cm.viridis(np.linspace(0, 1, len(data_for_plot)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax5.set_ylabel('Temporal Activity')
            ax5.set_title('5. Biochemical Validation\n(Activity vs Flexibility Group)')
            ax5.tick_params(axis='x', rotation=30)

            # Correlation: assign numeric rank per bin
            flat_activities = [v for group in data_for_plot for v in group]
            bin_ranks = []
            for i, group in enumerate(data_for_plot):
                bin_ranks.extend([i + 1] * len(group))  # 1-based rank per bin

            if len(flat_activities) == len(bin_ranks):
                corr, p_value = spearmanr(flat_activities, bin_ranks)
                ax5.text(0.02, 0.98, f'rho = {corr:.3f}\np = {p_value:.3f}',
                        transform=ax5.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


    # 6. Dynamic Pattern Analysis
    ax6 = plt.subplot(3, 4, 6)
    
    # Analyze dynamic patterns using sliding window
    window_size = min(10, n_residues // 5)
    if window_size >= 3:
        sliding_activity = []
        residue_positions = []
        
        for i in range(n_residues - window_size + 1):
            window_activity = np.mean(mean_temporal_activity[i:i+window_size])
            sliding_activity.append(window_activity)
            residue_positions.append(i + window_size // 2)
        
        ax6.plot(residue_positions, sliding_activity, 'b-', linewidth=2, alpha=0.8, label='Sliding Average')
        ax6.scatter(range(n_residues), mean_temporal_activity, alpha=0.4, s=20, color='gray', label='Individual Residues')
        
        # Identify dynamic hotspots
        threshold = np.percentile(sliding_activity, 75)
        hotspots = np.array(sliding_activity) > threshold
        if np.any(hotspots):
            hotspot_positions = np.array(residue_positions)[hotspots]
            hotspot_activities = np.array(sliding_activity)[hotspots]
            ax6.scatter(hotspot_positions, hotspot_activities, color='red', s=50, 
                       label=f'Dynamic Hotspots (top 25%)', zorder=5)
        
        ax6.set_xlabel('Residue Index')
        ax6.set_ylabel('Smoothed Temporal Activity')
        ax6.set_title('6. Dynamic Pattern Analysis\n(Sliding Window, Hotspot Detection)')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
    
    # 7. Temporal Autocorrelation Function (Enhanced)
    ax7 = plt.subplot(3, 4, 7)
    
    # Calculate autocorrelation for the most active residue
    most_active_residue = np.argmax(mean_temporal_activity)
    residue_trajectory = global_dynamics[:, most_active_residue]
    
    # Autocorrelation
    max_lag = min(50, n_frames // 4)
    autocorr = []
    lags = range(max_lag)
    
    for lag in lags:
        if lag == 0:
            autocorr.append(1.0)
        else:
            corr = np.corrcoef(residue_trajectory[:-lag], residue_trajectory[lag:])[0, 1]
            autocorr.append(corr)
    
    ax7.plot(lags, autocorr, 'o-', linewidth=2, markersize=4, color='darkgreen')
    ax7.set_xlabel('Time Lag (frames)')
    ax7.set_ylabel('Autocorrelation')
    ax7.set_title(f'7. Temporal Memory\n(Residue {most_active_residue} autocorrelation)')
    ax7.grid(True, alpha=0.3)
    ax7.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Find correlation decay time
    decay_threshold = 1/np.e  # ~0.37
    decay_frame = next((i for i, corr in enumerate(autocorr) if corr < decay_threshold), max_lag)
    ax7.axhline(y=decay_threshold, color='red', linestyle='--', alpha=0.7, label=f'1/e decay')
    ax7.text(decay_frame + 2, decay_threshold + 0.1, f'τ ≈ {decay_frame} frames', 
             fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax7.legend(fontsize=8)
    
    # 8. Collective Motion Analysis (Enhanced)
    ax8 = plt.subplot(3, 4, 8)
    
    # PCA of residue dynamics to find collective motions
    scaler = StandardScaler()
    dynamics_scaled = scaler.fit_transform(global_dynamics)  # (frames, residues)
    
    pca = PCA(n_components=min(5, n_residues))
    pc_scores = pca.fit_transform(dynamics_scaled)
    
    # Plot first few principal components
    for i in range(min(3, pc_scores.shape[1])):
        ax8.plot(times, pc_scores[:, i], label=f'PC{i+1} ({pca.explained_variance_ratio_[i]:.1%})', 
                linewidth=2, alpha=0.8)
    
    ax8.set_xlabel('Time (ns)')
    ax8.set_ylabel('PC Score')
    ax8.set_title('8. Collective Motions\n(Principal Components of Dynamics)')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)
    
    # Add cumulative variance explained
    cum_var = np.cumsum(pca.explained_variance_ratio_[:3])
    ax8.text(0.02, 0.98, f'Cumulative variance:\nPC1-3: {cum_var[-1]:.1%}', 
             transform=ax8.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 9. Residue Communication Network (IMPROVED)
    ax9 = plt.subplot(3, 4, 9)
    
    # Calculate dynamic cross-correlation
    residue_correlations = np.corrcoef(global_dynamics.T)  # (residues, residues)
    
        # # Smart sampling: include most active residues + regular sampling
        # n_top_active = min(20, n_residues // 3)
        # top_active_indices = np.argsort(mean_temporal_activity)[-n_top_active:]
        
        # # Regular sampling for remaining residues
        # remaining_indices = [i for i in range(n_residues) if i not in top_active_indices]
        # n_regular = min(30, len(remaining_indices))
        # if remaining_indices:
        #     regular_sample = np.linspace(0, len(remaining_indices)-1, n_regular, dtype=int)
        #     regular_indices = [remaining_indices[i] for i in regular_sample]
        # else:
        #     regular_indices = []
        
        # # Combine indices
        # sample_indices = sorted(list(top_active_indices) + regular_indices)
        # residue_correlations = residue_correlations[sample_indices][:, sample_indices]
    
    # im9 = ax9.imshow(residue_correlations, cmap='RdBu_r', vmin=-1, vmax=1)
    # ax9.set_xlabel('Residue Index (sampled)')
    # ax9.set_ylabel('Residue Index (sampled)')
    # ax9.set_title('9. Residue Communication Network\n(Dynamic Cross-Correlation)')
    
    im9 = ax9.imshow(residue_correlations, cmap='RdBu_r', vmin=-1, vmax=1)

    if residue_info and 'residue_numbers' in residue_info and 'amino_acids' in residue_info:
        labels = [f"{aa}{num}" for aa, num in zip(residue_info['amino_acids'], residue_info['residue_numbers'])]
        
        # Show fewer ticks if too many residues
        max_ticks = 50
        step = max(1, len(labels) // max_ticks)
        tick_indices = np.arange(0, len(labels), step)
        
        ax9.set_xticks(tick_indices)
        ax9.set_xticklabels([labels[i] for i in tick_indices], rotation=90, fontsize=6)
        ax9.set_yticks(tick_indices)
        ax9.set_yticklabels([labels[i] for i in tick_indices], fontsize=6)
    else:
        ax9.set_xlabel('Residue Index')
        ax9.set_ylabel('Residue Index')

    ax9.set_title('9. Residue Communication Network\n(Dynamic Cross-Correlation)')
    
    # Add colorbar and sample info
    cbar9 = plt.colorbar(im9, ax=ax9)
    cbar9.set_label('Correlation', rotation=270, labelpad=15)
    
    # Mark high activity residues
    # n_top = len(top_active_indices)
    # ax9.axhline(y=n_top-0.5, color='yellow', linestyle='--', alpha=0.7)
    # ax9.axvline(x=n_top-0.5, color='yellow', linestyle='--', alpha=0.7)
    # ax9.text(0.02, 0.98, f'Top {n_top} active\nresidues shown first', 
    #          transform=ax9.transAxes, verticalalignment='top',
    #          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # 10. Conformational Fluctuation Analysis (REFRAMED from "Energy Landscape")
    ax10 = plt.subplot(3, 4, 10)
    
    # Calculate conformational fluctuations from mean structure
    mean_conformation = np.mean(global_dynamics, axis=0)
    fluctuations = np.sum((global_dynamics - mean_conformation)**2, axis=1)
    
    ax10.plot(times, fluctuations, alpha=0.7, linewidth=1, color='darkgreen')
    ax10.set_xlabel('Time (ns)')
    ax10.set_ylabel('Conformational Fluctuation')
    ax10.set_title('10. Conformational Fluctuation Analysis\n(Deviation from Mean Structure)')
    ax10.grid(True, alpha=0.3)
    
    # Identify stable vs fluctuating periods
    low_fluct = fluctuations < np.percentile(fluctuations, 25)
    high_fluct = fluctuations > np.percentile(fluctuations, 75)
    
    ax10.fill_between(times, fluctuations, alpha=0.3, where=low_fluct, 
                      color='blue', label='Stable periods (bottom 25%)')
    ax10.fill_between(times, fluctuations, alpha=0.3, where=high_fluct, 
                      color='red', label='Fluctuating periods (top 25%)')
    ax10.legend(fontsize=8)
    
    # Add statistical annotations
    mean_fluct = np.mean(fluctuations)
    ax10.axhline(y=mean_fluct, color='black', linestyle='--', alpha=0.7,
                 label=f'Mean: {mean_fluct:.3f}')
    
    # 11. Conformational States (Enhanced)
    ax11 = plt.subplot(3, 4, 11)
    
    # Cluster frames by similarity
    from sklearn.cluster import KMeans
    
    n_clusters = min(5, n_frames // 100)  # Adaptive number of clusters
    if n_clusters >= 2:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        frame_clusters = kmeans.fit_predict(global_dynamics)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(global_dynamics, frame_clusters)
        
        # Plot cluster assignments over time
        colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
        for cluster_id in range(n_clusters):
            cluster_mask = frame_clusters == cluster_id
            cluster_times = times[cluster_mask]
            cluster_y = np.full_like(cluster_times, cluster_id)
            ax11.scatter(cluster_times, cluster_y, c=[colors[cluster_id]], 
                        alpha=0.6, s=20, label=f'State {cluster_id} ({np.sum(cluster_mask)} frames)')
        
        ax11.set_xlabel('Time (ns)')
        ax11.set_ylabel('Conformational State')
        ax11.set_title('11. Conformational States\n(K-means Clustering)')
        ax11.set_yticks(range(n_clusters))
        ax11.set_yticklabels([f'State {i}' for i in range(n_clusters)])
        ax11.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add silhouette score
        ax11.text(0.02, 0.02, f'Silhouette Score: {silhouette_avg:.3f}', 
                  transform=ax11.transAxes, verticalalignment='bottom',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 12. Dynamic Variability Map (REFRAMED from "Prediction Difficulty")
    ax12 = plt.subplot(3, 4, 12)
    
    # Calculate local variability for each residue
    variability = []
    window_size = min(10, n_frames // 10)
    
    for res_idx in range(n_residues):
        residue_traj = global_dynamics[:, res_idx]
        
        # Calculate rolling variance
        if len(residue_traj) >= window_size:
            rolling_vars = []
            for i in range(len(residue_traj) - window_size + 1):
                window_var = np.var(residue_traj[i:i+window_size])
                rolling_vars.append(window_var)
            local_variability = np.mean(rolling_vars)
        else:
            local_variability = np.var(residue_traj)
        
        variability.append(local_variability)
    
    variability = np.array(variability)
    
    # Color by variability level
    norm_variability = variability / np.max(variability)
    bars = ax12.bar(range(n_residues), variability, 
                    color=plt.cm.viridis(norm_variability), alpha=0.8)
    ax12.set_xlabel('Residue Index')
    ax12.set_ylabel('Local Dynamic Variability')
    ax12.set_title('12. Dynamic Variability Map\n(Local Fluctuation Intensity)')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=np.max(variability)))
    sm.set_array([])
    cbar12 = plt.colorbar(sm, ax=ax12)
    cbar12.set_label('Variability', rotation=270, labelpad=15)
    
    # Identify highly variable regions
    high_var_threshold = np.percentile(variability, 90)
    high_var_residues = np.where(variability > high_var_threshold)[0]
    if len(high_var_residues) > 0:
        ax12.text(0.02, 0.98, f'High variability residues:\n{high_var_residues[:10]}...', 
                  transform=ax12.transAxes, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save comprehensive analysis
    output_dir = "/scratch/groups/rbaltman/ziyiw23/EMPROT/data/results/analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    
    trajectory_id = os.path.basename(lmdb_path)
    plot_path = os.path.join(output_dir, f"comprehensive_analysis_{trajectory_id}.png")
    txt_path = os.path.join(output_dir, f"comprehensive_analysis_{trajectory_id}.txt")
    
    # Helper print function to write to both console and file
    import builtins
    def dual_print(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=txtfile)
    
    with open(txt_path, 'w') as txtfile:
        dual_print("\n" + "="*60)
        dual_print(f"PROTEIN INFO")
        dual_print("="*60)
        dual_print(protein_info)
        dual_print(f"\nComprehensive analysis saved to: {plot_path}")
        
        # Print summary statistics
        dual_print("\n" + "="*60)
        dual_print("ANALYSIS SUMMARY")
        dual_print("="*60)
        dual_print(f"Most active residue: {most_active_residue} (activity: {mean_temporal_activity[most_active_residue]:.4f})")
        dual_print(f"Mean temporal activity: {np.mean(mean_temporal_activity):.4f} ± {np.std(mean_temporal_activity):.4f}")
        dual_print(f"Temporal memory decay: ~{decay_frame if 'decay_frame' in locals() else 'N/A'} frames")
        
        if 'frame_correlations' in locals():
            dual_print(f"Mean frame-to-frame correlation: {np.mean(frame_correlations):.4f}")
            dual_print(f"Temporal smoothness: {'High' if np.mean(frame_correlations) > 0.9 else 'Moderate' if np.mean(frame_correlations) > 0.7 else 'Low'}")
        
        if 'pca' in locals():
            dual_print(f"Collective motion captured by PC1-3: {np.sum(pca.explained_variance_ratio_[:3]):.1%}")
        
        if 'silhouette_avg' in locals():
            dual_print(f"Conformational state clustering quality: {silhouette_avg:.3f}")
        
        # Additional biological insights
        dual_print("\n" + "-"*40)
        dual_print("BIOLOGICAL INSIGHTS")
        dual_print("-"*40)
        
        # Flexibility vs rigidity analysis
        rigid_residues = np.where(mean_temporal_activity < np.percentile(mean_temporal_activity, 25))[0]
        flexible_residues = np.where(mean_temporal_activity > np.percentile(mean_temporal_activity, 75))[0]
        dual_print(f"Rigid regions (bottom 25%): residues {rigid_residues[:10]}{'...' if len(rigid_residues) > 10 else ''}")
        dual_print(f"Flexible regions (top 25%): residues {flexible_residues[:10]}{'...' if len(flexible_residues) > 10 else ''}")
        
        # Dynamic range analysis
        dynamic_range = np.max(mean_temporal_activity) / np.min(mean_temporal_activity)
        dual_print(f"Dynamic range (max/min activity): {dynamic_range:.2f}x")
        
        # Temporal coherence analysis
        if 'frame_correlations' in locals():
            stability_periods = np.where(np.array(frame_correlations) > 0.95)[0]
            transition_periods = np.where(np.array(frame_correlations) < 0.8)[0]
            dual_print(f"Highly stable periods: {len(stability_periods)}/{len(frame_correlations)} frames ({len(stability_periods)/len(frame_correlations)*100:.1f}%)")
            dual_print(f"Transition periods: {len(transition_periods)}/{len(frame_correlations)} frames ({len(transition_periods)/len(frame_correlations)*100:.1f}%)")
        
        # Hotspot identification
        if 'variability' in locals():
            hotspot_threshold = np.percentile(variability, 95)
            hotspot_residues = np.where(variability > hotspot_threshold)[0]
            dual_print(f"Dynamic hotspots (top 5% variability): residues {hotspot_residues}")
            
            # Check if hotspots are clustered or distributed
            if len(hotspot_residues) > 1:
                hotspot_gaps = np.diff(hotspot_residues)
                clustered_hotspots = np.sum(hotspot_gaps <= 3)  # Within 3 residues
                dual_print(f"Clustered hotspots: {clustered_hotspots}/{len(hotspot_gaps)} gaps ≤ 3 residues")
        
        # Domain motion hypothesis
        if 'pca' in locals() and pca.n_components_ >= 2:
            # Check if first PC shows domain-like motion (bimodal distribution)
            pc1_scores = pc_scores[:, 0]
            pc1_hist, _ = np.histogram(pc1_scores, bins=20)
            bimodality_ratio = np.min(pc1_hist) / np.max(pc1_hist)
            if bimodality_ratio < 0.3:
                dual_print(f"PC1 shows potential domain motion (bimodality ratio: {bimodality_ratio:.3f})")
            else:
                dual_print(f"PC1 shows continuous motion (bimodality ratio: {bimodality_ratio:.3f})")
        
        # Amino acid property correlations (if available)
        if residue_info and 'amino_acids' in residue_info:
            amino_acids = residue_info['amino_acids'][:n_residues]
            
            # Hydrophobic vs hydrophilic activity
            hydrophobic_activity = []
            hydrophilic_activity = []
            
            for i, aa in enumerate(amino_acids):
                if aa in AMINO_ACID_PROPERTIES:
                    aa_type = AMINO_ACID_PROPERTIES[aa]['type']
                    if 'Hydrophobic' in aa_type:
                        hydrophobic_activity.append(mean_temporal_activity[i])
                    elif 'hydrophilic' in aa_type:
                        hydrophilic_activity.append(mean_temporal_activity[i])
            
            if hydrophobic_activity and hydrophilic_activity:
                hydrophobic_mean = np.mean(hydrophobic_activity)
                hydrophilic_mean = np.mean(hydrophilic_activity)
                dual_print(f"Hydrophobic residue activity: {hydrophobic_mean:.4f} ± {np.std(hydrophobic_activity):.4f}")
                dual_print(f"Hydrophilic residue activity: {hydrophilic_mean:.4f} ± {np.std(hydrophilic_activity):.4f}")
                dual_print(f"Hydrophobic/Hydrophilic ratio: {hydrophobic_mean/hydrophilic_mean:.2f}")
        
        dual_print("="*60)
    # Also print to console for the main output
    print(f"\nComprehensive analysis saved to: {plot_path}")


    # Additional analysis for model design recommendations
    print(f"\nText analysis saved to: {txt_path}")
    
    # Append model design recommendations to the text file
    with open(txt_path, 'a') as txtfile:
        def dual_print(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, **kwargs, file=txtfile)
            
        dual_print("\n" + "="*60)
        dual_print("MODEL DESIGN RECOMMENDATIONS")
        dual_print("="*60)
        
        # 1. Temporal Architecture Recommendations
        dual_print("\n🏗️  ARCHITECTURE RECOMMENDATIONS:")
        dual_print("-" * 40)
        
        if 'frame_correlations' in locals():
            mean_corr = np.mean(frame_correlations)
            if mean_corr > 0.9:
                dual_print("✅ HIGH TEMPORAL SMOOTHNESS detected:")
                dual_print("   → Use LSTM/GRU with longer sequence windows")
                dual_print("   → Transformer with relative position encoding")
                dual_print("   → Consider gradient clipping for stable training")
            elif mean_corr > 0.7:
                dual_print("⚖️  MODERATE TEMPORAL SMOOTHNESS detected:")
                dual_print("   → Hybrid CNN-RNN architecture recommended")
                dual_print("   → Attention mechanisms with local bias")
                dual_print("   → Use residual connections for gradient flow")
            else:
                dual_print("⚡ LOW TEMPORAL SMOOTHNESS (jumpy dynamics):")
                dual_print("   → Focus on frame-to-frame prediction")
                dual_print("   → Use attention over RNNs (less temporal bias)")
                dual_print("   → Consider adversarial training for stability")
        
        # 2. Memory and Sequence Length Recommendations
        if 'decay_frame' in locals():
            dual_print(f"\n🧠 MEMORY & SEQUENCE LENGTH:")
            dual_print("-" * 40)
            dual_print(f"   Temporal memory decay: ~{decay_frame} frames")
            if decay_frame < 10:
                dual_print("   → SHORT MEMORY: Use sequence length 15-30 frames")
                dual_print("   → Focus on local temporal patterns")
            elif decay_frame < 30:
                dual_print("   → MEDIUM MEMORY: Use sequence length 50-100 frames")
                dual_print("   → Balance local and global temporal features")
            else:
                dual_print("   → LONG MEMORY: Use sequence length 100+ frames")
                dual_print("   → Essential to capture long-range dependencies")
        
        # 3. Residue-Specific Modeling
        dynamic_range = np.max(mean_temporal_activity) / np.min(mean_temporal_activity)
        dual_print(f"\n🎯 RESIDUE-SPECIFIC MODELING:")
        dual_print("-" * 40)
        dual_print(f"   Dynamic range: {dynamic_range:.2f}x")
        
        if dynamic_range > 5.0:
            dual_print("   ⚠️  HIGH HETEROGENEITY detected:")
            dual_print("   → Use residue-specific learning rates")
            dual_print("   → Implement adaptive loss weighting")
            dual_print("   → Consider mixture of experts architecture")
        elif dynamic_range > 2.0:
            dual_print("   📊 MODERATE HETEROGENEITY:")
            dual_print("   → Use position-dependent dropout")
            dual_print("   → Implement residue-type embeddings")
        else:
            dual_print("   ✅ LOW HETEROGENEITY:")
            dual_print("   → Standard uniform architecture sufficient")
        
        # 4. Collective Motion Modeling
        if 'pca' in locals():
            cum_var_3 = np.sum(pca.explained_variance_ratio_[:3])
            dual_print(f"\n🌊 COLLECTIVE MOTION MODELING:")
            dual_print("-" * 40)
            dual_print(f"   PC1-3 explain {cum_var_3:.1%} of variance")
            
            if cum_var_3 > 0.8:
                dual_print("   🎯 HIGHLY COLLECTIVE dynamics:")
                dual_print("   → Use graph neural networks")
                dual_print("   → Implement global attention mechanisms")
                dual_print("   → Consider PCA-regularized loss functions")
            elif cum_var_3 > 0.5:
                dual_print("   ⚖️  MODERATELY COLLECTIVE:")
                dual_print("   → Use multi-head attention")
                dual_print("   → Implement residue-residue interaction layers")
            else:
                dual_print("   🔀 INDEPENDENT residue dynamics:")
                dual_print("   → Per-residue modeling sufficient")
                dual_print("   → Use local convolutions")
        
        # 5. Training Strategy Recommendations
        dual_print(f"\n🎓 TRAINING STRATEGY:")
        dual_print("-" * 40)
        
        # Stability analysis
        if 'frame_correlations' in locals():
            stability_periods = np.where(np.array(frame_correlations) > 0.95)[0]
            transition_periods = np.where(np.array(frame_correlations) < 0.8)[0]
            stability_ratio = len(stability_periods) / len(frame_correlations)
            
            if stability_ratio > 0.7:
                dual_print("   🔒 HIGHLY STABLE protein:")
                dual_print("   → Use curriculum learning (stable → dynamic)")
                dual_print("   → Lower learning rates for fine-tuning")
                dual_print("   → Focus on rare transition events")
            elif stability_ratio > 0.3:
                dual_print("   ⚖️  BALANCED stability/dynamics:")
                dual_print("   → Use mixed training strategies")
                dual_print("   → Weighted sampling for transitions")
            else:
                dual_print("   ⚡ HIGHLY DYNAMIC protein:")
                dual_print("   → Use higher learning rates")
                dual_print("   → Focus on capturing rapid changes")
                dual_print("   → Consider data augmentation")
        
        # 6. Loss Function Design
        dual_print(f"\n🎯 LOSS FUNCTION DESIGN:")
        dual_print("-" * 40)
        
        # Hotspot analysis for loss weighting
        if 'variability' in locals():
            hotspot_threshold = np.percentile(variability, 95)
            hotspot_residues = np.where(variability > hotspot_threshold)[0]
            hotspot_ratio = len(hotspot_residues) / len(variability)
            
            dual_print(f"   Dynamic hotspots: {hotspot_ratio:.1%} of residues")
            if hotspot_ratio < 0.1:
                dual_print("   → Weight hotspot residues 3-5x in loss")
                dual_print("   → Use focal loss for rare events")
            else:
                dual_print("   → Standard weighting sufficient")
            
            dual_print("   💡 RECOMMENDED LOSS COMPONENTS:")
            dual_print("   → MSE for stable regions")
            dual_print("   → Huber loss for dynamic regions") 
            dual_print("   → Temporal consistency regularization")
            if cum_var_3 > 0.6:
                dual_print("   → PCA reconstruction loss")
        
        # 7. Data Augmentation Strategy
        dual_print(f"\n🔄 DATA AUGMENTATION:")
        dual_print("-" * 40)
        
        if 'frame_correlations' in locals():
            mean_corr = np.mean(frame_correlations)
            if mean_corr > 0.9:
                dual_print("   → Time warping (gentle stretching/compression)")
                dual_print("   → Gaussian noise injection (low variance)")
            else:
                dual_print("   → Frame dropout for robustness")
                dual_print("   → Temporal jittering")
        
        # 8. Validation Strategy
        dual_print(f"\n✅ VALIDATION STRATEGY:")
        dual_print("-" * 40)
        dual_print("   → Split by time (not random) for temporal models")
        dual_print("   → Validate on both stable and transition periods")
        dual_print("   → Monitor biochemical property conservation")
        if 'silhouette_avg' in locals():
            dual_print(f"   → Target silhouette score > {silhouette_avg:.2f}")
        
        # 9. Computational Considerations
        dual_print(f"\n💻 COMPUTATIONAL OPTIMIZATION:")
        dual_print("-" * 40)
        if n_residues > 200:
            dual_print("   → Use gradient checkpointing for long sequences")
            dual_print("   → Consider hierarchical attention")
        if 'decay_frame' in locals() and decay_frame > 50:
            dual_print("   → Use truncated backpropagation")
            dual_print("   → Implement efficient attention mechanisms")
        
        # 10. Architecture-Specific Recommendations
        dual_print(f"\n🏛️  ARCHITECTURE-SPECIFIC GUIDANCE:")
        dual_print("-" * 40)
        
        # Transformer recommendations
        dual_print("   🤖 FOR TRANSFORMER ARCHITECTURES:")
        if 'frame_correlations' in locals() and np.mean(frame_correlations) > 0.8:
            dual_print("   → Use relative position encoding")
            dual_print("   → Local attention windows + global connections")
        if cum_var_3 > 0.6:
            dual_print("   → Multi-head attention across residues")
        
        # RNN recommendations  
        dual_print("   🔄 FOR RNN ARCHITECTURES:")
        if 'decay_frame' in locals():
            if decay_frame > 30:
                dual_print("   → Use LSTM with large hidden states")
            else:
                dual_print("   → GRU sufficient for short memory")
        
        # CNN recommendations
        dual_print("   🧮 FOR CNN ARCHITECTURES:")
        dual_print("   → Use dilated convolutions for long-range patterns")
        if hotspot_ratio < 0.2:
            dual_print("   → Focus kernel sizes on hotspot regions")
        
        dual_print("\n" + "="*60)
        dual_print("💡 KEY TAKEAWAY:")
        dual_print("Your temporal analysis reveals the protein's 'personality'")
        dual_print("Design your model to match this specific behavior!")
        dual_print("="*60)
    # Also print to console for the main output
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    
    # Save results in JSON format for training integration
    json_path = os.path.join(output_dir, f"temporal_analysis_{trajectory_id}.json")
    analysis_results = {
        'trajectory_id': trajectory_id,
        'metadata': {
            'num_frames': n_frames,
            'num_residues': n_residues,
            'embedding_dim': n_dims,
            'path': lmdb_path
        },
        'temporal_smoothness': np.mean(frame_correlations) if 'frame_correlations' in locals() else None,
        'temporal_activity': [float(x) for x in mean_temporal_activity.tolist()],
        'autocorr_decay': int(decay_frame) if 'decay_frame' in locals() else None,
        'dynamic_range': float(np.max(mean_temporal_activity) / np.min(mean_temporal_activity)),
        'most_active_residue': int(most_active_residue),
        'stability_ratio': len(np.where(np.array(frame_correlations) > 0.95)[0]) / len(frame_correlations) if 'frame_correlations' in locals() else None,
        'hotspot_residues': np.where(variability > np.percentile(variability, 95))[0].tolist() if 'variability' in locals() else [],
        'collective_motion_variance': float(np.sum(pca.explained_variance_ratio_[:3])) if 'pca' in locals() else None,
        'silhouette_score': float(silhouette_avg) if 'silhouette_avg' in locals() else None,
        'residue_info': {
            'amino_acids': residue_info.get('amino_acids', []) if residue_info else [],
            'residue_numbers': residue_info.get('residue_numbers', []) if residue_info else []
        }
    }
    
    with open(json_path, 'w') as f:
        import json
        json.dump(analysis_results, f, indent=2)
    
    print(f"JSON analysis saved to: {json_path}")
    
    return {
        'embeddings': embeddings,
        'times': times,
        'temporal_activity': mean_temporal_activity,
        'frame_correlations': frame_correlations if 'frame_correlations' in locals() else None,
        'residue_info': residue_info,
        'autocorr_decay': decay_frame if 'decay_frame' in locals() else None,
        'collective_motions': pca if 'pca' in locals() else None,
        'conformational_states': frame_clusters if 'frame_clusters' in locals() else None,
        'variability_map': variability
    }

def extract_numeric_prefix(label):
    try:
        return float(label.split()[0])
    except:
        return float('inf')
    
def extract_residue_info(first_frame):
    """Extract amino acid sequence and other residue information."""
    residue_info = {}
    
    try:
        # Try to extract amino acid information from the frame
        if 'atoms' in first_frame and hasattr(first_frame['atoms'], 'resname'):
            # Get unique residues in order
            atoms_df = first_frame['atoms']
            residue_df = atoms_df[['residue', 'resname']].drop_duplicates().sort_values('residue')
            
            # Map 3-letter to 1-letter amino acid codes
            AA_MAP = {
                'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
            }
            
            amino_acids = []
            for _, row in residue_df.iterrows():
                resname = row['resname']
                if resname in AA_MAP:
                    amino_acids.append(AA_MAP[resname])
                else:
                    amino_acids.append('X')  # Unknown
            
            residue_info['amino_acids'] = amino_acids
            residue_info['residue_numbers'] = residue_df['residue'].tolist()
            
            print(f"Extracted sequence: {''.join(amino_acids[:50])}{'...' if len(amino_acids) > 50 else ''}")
            
    except Exception as e:
        print(f"Could not extract residue information: {e}")
    
    return residue_info

if __name__ == "__main__":
    import random

    seed = 196
    random.seed(seed)

    lmdb_paths = os.listdir("/scratch/groups/rbaltman/ziyiw23/traj_embeddings")
    lmdb_paths = random.sample(lmdb_paths, 1)

    for trajectory_path in lmdb_paths:
        print("-"*80)
        print(f"Running comprehensive protein dynamics analysis on {trajectory_path}...")
        trajectory_path = os.path.join("/scratch/groups/rbaltman/ziyiw23/traj_embeddings", trajectory_path)
        results = comprehensive_protein_analysis(trajectory_path, num_frames=2500)

        # protein_info = get_protein_info(metadata_path = "traj_metadata.csv", lmdb_path = trajectory_path)
        # print(protein_info)

        print("\nComprehensive analysis complete!")
        print("-"*80)