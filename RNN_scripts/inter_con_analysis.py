# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:09:42 2024

@author: RHIRAsimulation
"""
import numpy as np

class analyze_inter_con:
    def __init__(self,S_A,S_B,act_avg_A,act_avg_B,nUnit,nInh,range_min=0, range_max=180):
        self.nUnit=nUnit
        self.nInh=nInh
        self.nExc=nUnit-nInh
        self.S_A=S_A
        self.S_B=S_B
        self.act_avg_A=act_avg_A
        self.act_avg_B=act_avg_B
        
        self.peak_times_A = np.argmax(act_avg_A[range_min:range_max,:], axis=0)  # Shape: (N,)
        self.peak_times_B = np.argmax(act_avg_B[range_min:range_max,:], axis=0)  # Shape: (N,)
        
    def analyze_connections(self,  is_S_A=True, use_phase=False, top_percent=100):
        """
        Analyze connections and plot the relationship between connection strength and time difference of peak activity.
        
        Parameters:
            S_matrix (np.ndarray): Connection matrix.
            peak_times_source (np.ndarray): Peak times of source neurons.
            peak_times_target (np.ndarray): Peak times of target neurons.
            title (str): Title for the plot.
            is_S_A (bool): If True, analyzing S_A; if False, analyzing S_B.
            use_phase (bool): If True, calculate time differences considering cyclic activity.
            top_percent (float): Percentage of top connections to analyze (between 0 and 100).
        """
        if is_S_A:
            S_matrix=self.S_A
            peak_times_source=self.peak_times_A
            peak_times_target=self.peak_times_B
            title='S_A: Sorted Connection Matrix'
        else:
            S_matrix=self.S_B
            peak_times_source=self.peak_times_B
            peak_times_target=self.peak_times_A
            title='S_B: Sorted Connection Matrix'
        # S_matrix[i, j]: Connection from neuron i in source to neuron j in target
        indices = np.nonzero(S_matrix)
        i_indices = indices[0]
        j_indices = indices[1]
        S_values = S_matrix[i_indices, j_indices]
        
        if is_S_A:
            # For S_A, only excitatory neurons in A have outgoing connections
            valid_indices = i_indices < nExc
        else:
            # For S_B, only excitatory neurons in B have outgoing connections
            valid_indices = i_indices < nExc  # Adjust if necessary
        
        i_valid = i_indices[valid_indices]
        j_valid = j_indices[valid_indices]
        S_nonzero = S_matrix[i_valid, j_valid]
        
        # Calculate time differences
        t_source = peak_times_source[i_valid]
        t_target = peak_times_target[j_valid]
        
        if use_phase:
            # Calculate minimal time difference considering cyclic activity
            time_diffs = np.minimum((t_source - t_target) % T, (t_target - t_source) % T)
        else:
            # Calculate absolute time differences
            time_diffs = np.abs(t_source - t_target)
        
        # Select top x% of connections
        if top_percent < 100:
            num_top = int(len(S_nonzero) * top_percent / 100)
            if num_top < 1:
                num_top = 1  # Ensure at least one connection is selected
            # Get indices of top connections
            sorted_indices = np.argsort(-S_nonzero)  # Negative sign for descending sort
            top_indices = sorted_indices[:num_top]
            # Filter the data
            S_nonzero = S_nonzero[top_indices]
            time_diffs = time_diffs[top_indices]
        
        # Plotting
        plt.figure()
        plt.scatter(time_diffs, S_nonzero,s=1, alpha=0.5, label='Data points')
        plt.ylabel('Connection Strength')
        plt.xlabel('Time Difference of Peak Activity')
        if use_phase:
            plt.xlabel('Minimal Time Difference (Cyclic)')
        plt.title(title)
        
        # Linear regression
        if len(time_diffs) > 1:
            coefficients = np.polyfit(time_diffs, S_nonzero, 1)
            slope = coefficients[0]
            intercept = coefficients[1]
            
            # Generate x values for the line
            x_fit = np.linspace(np.min(time_diffs), np.max(time_diffs), 100)
            y_fit = slope * x_fit + intercept
            
            # Plot the best fit line
            plt.plot(x_fit, y_fit, color='red', label=f'Best fit line (slope = {slope:.2E})')
        else:
            slope = np.nan  # Not enough points to compute slope
            plt.text(0.5, 0.5, 'Not enough data for regression', transform=plt.gca().transAxes, ha='center')
        
        # Add legend
        plt.legend()
        
        plt.show()
        return time_diffs, S_nonzero
    
    def plot_time_of_max_activity(S_matrix, peak_times_source, peak_times_target, title, is_S_A=True, use_alpha=False,top_percent=100):
        """
        Plot a scatter plot where each point represents a non-zero connection,
        and its position corresponds to the peak activity times of the source and target neurons.
        
        Parameters:
            S_matrix (np.ndarray): Connection matrix.
            peak_times_source (np.ndarray): Peak times of source neurons.
            peak_times_target (np.ndarray): Peak times of target neurons.
            title (str): Title for the plot.
            is_S_A (bool): If True, analyzing S_A; if False, analyzing S_B.
            use_alpha (bool): If True, use alpha to represent connection strength; otherwise, use color.
        """
        if is_S_A:
            S_matrix=self.S_A
            peak_times_source=self.peak_times_A
            peak_times_target=self.peak_times_B
            title='S_A: Sorted Connection Matrix'
        else:
            S_matrix=self.S_B
            peak_times_source=self.peak_times_B
            peak_times_target=self.peak_times_A
            title='S_B: Sorted Connection Matrix'
        
        # S_matrix[i, j]: Connection from neuron i in source to neuron j in target
        indices = np.nonzero(S_matrix)
        i_indices = indices[0]
        j_indices = indices[1]
        S_values = S_matrix[i_indices, j_indices]
        
        if is_S_A:
            # For S_A, only excitatory neurons in A have outgoing connections
            valid_indices = i_indices < nExc
        else:
            # For S_B, only excitatory neurons in B have outgoing connections
            valid_indices = i_indices < nExc  # Adjust if necessary
        
        i_valid = i_indices[valid_indices]
        j_valid = j_indices[valid_indices]
        S_nonzero = S_matrix[i_valid, j_valid]
        
        # Get peak times
        t_source = peak_times_source[i_valid]
        t_target = peak_times_target[j_valid]
        
    
        # get only top percent of the connection
        # Select top x% of connections
        if top_percent < 100:
            num_top = int(len(S_nonzero) * top_percent / 100)
            if num_top < 1:
                num_top = 1  # Ensure at least one connection is selected
            # Get indices of top connections
            sorted_indices = np.argsort(-S_nonzero)  # Negative sign for descending sort
            top_indices = sorted_indices[:num_top]
            # Filter the data
            S_nonzero = S_nonzero[top_indices]  
            t_source =  t_source[top_indices]  
            t_target =  t_target[top_indices]  
    
        # Normalize connection strengths for color or alpha mapping
        S_min = np.min(S_nonzero)
        S_max = np.max(S_nonzero)
        S_norm = (S_nonzero - S_min) / (S_max - S_min + 1e-8)  # Avoid division by zero
        
        plt.figure(figsize=(8, 6))
        
        if use_alpha:
            colorvec=np.zeros((np.shape(S_norm)[0],4))
            colorvec[:,2]=1
            colorvec[:,3]=S_norm
            # Use alpha to represent connection strength
            plt.scatter(t_source+np.random.normal(0,0.3,size=np.shape(t_source)), t_target+np.random.normal(0,0.3,size=np.shape(t_source)),s=1,color=colorvec)
        else:
            # Use color to represent connection strength
            plt.scatter(t_source+np.random.normal(0,0.3,size=np.shape(t_source)), t_target,s=1+np.random.normal(0,0.3,size=np.shape(t_source)), c=S_norm, cmap='viridis')
            cbar = plt.colorbar()
            cbar.set_label('Normalized Connection Strength')
        
        plt.xlabel('Peak Time of Source Neuron')
        plt.ylabel('Peak Time of Target Neuron')
        plt.title(title)
        plt.grid(True)
        plt.show()
    
    def plot_sorted_connection_matrix(S_matrix, peak_times_source, peak_times_target, title, is_S_A=True, top_percent=100, cmap='viridis',vmin=0, vmax=0.05):
        """
        Plot a sorted connection matrix using imshow, where neurons are ordered based on their peak activity times.
    
        Parameters:
            S_matrix (np.ndarray): Connection matrix (N x N).
            peak_times_source (np.ndarray): Peak times of source neurons (length N).
            peak_times_target (np.ndarray): Peak times of target neurons (length N).
            title (str): Title for the plot.
            is_S_A (bool): If True, analyzing S_A; if False, analyzing S_B.
            top_percent (float): Percentage of top connections to display (0-100). If <100, only top x% connections are shown.
            cmap (str): Colormap for imshow.
        """
        if is_S_A:
            S_matrix=self.S_A
            peak_times_source=self.peak_times_A
            peak_times_target=self.peak_times_B
            title='S_A: Sorted Connection Matrix'
        else:
            S_matrix=self.S_B
            peak_times_source=self.peak_times_B
            peak_times_target=self.peak_times_A
            title='S_B: Sorted Connection Matrix'
            
        # Validate top_percent
        if not (0 < top_percent <= 100):
            raise ValueError("top_percent must be between 0 and 100.")
        
        
        # remove inh-> connections because they are all 0
        peak_times_source=peak_times_source[:nExc]
        # Sort source neurons based on peak times
        source_sorted_indices = np.argsort(peak_times_source)
        target_sorted_indices = np.argsort(peak_times_target)
        
        # Reorder the connection matrix
        S_sorted = S_matrix[source_sorted_indices, :][:, target_sorted_indices]
        
        # If top_percent < 100, mask out connections outside the top x%
        if top_percent < 100:
            # Flatten the matrix and get top x% values
            S_nonzero = S_sorted[S_sorted > 0]
            if len(S_nonzero) == 0:
                print("No non-zero connections to display.")
                return
            threshold = np.percentile(S_nonzero, 100 - top_percent)
            # Create a mask for connections below the threshold
            mask = S_sorted < threshold
            # Apply mask
            S_display = np.copy(S_sorted)
            S_display[mask] = 0
        else:
            S_display = S_sorted.copy()
        
        plt.figure(figsize=(8, 6))
        im = plt.imshow(np.log(S_display), aspect='auto', cmap=cmap, interpolation='none',vmin=vmin,vmax=vmax)
        plt.colorbar(im, label='Connection Strength')
        
        plt.xlabel('Target Neurons (Sorted by Peak Time)')
        plt.ylabel('Source Neurons (Sorted by Peak Time)')
        plt.title(title)
        
        # Optionally, add grid lines or ticks
        plt.tight_layout()
        plt.show()
        return S_display
        
    from scipy import stats
    def plotdistribution(S_nonzero_A, time_diffs_A, minthreash, maxthreash):
        """
        Plots the distribution of connection strengths for two subsets of data based on time differences.
        Additionally, it plots the mean values for each distribution and assesses the statistical significance
        of the difference between the means using an independent t-test and Mann-Whitney U test.
    
        Parameters:
            S_nonzero_A (np.ndarray): Array of non-zero connection strengths.
            time_diffs_A (np.ndarray): Array of time differences corresponding to connection strengths.
            minthreash (float): Minimum threshold for the first subset (time_diffs_A < minthreash).
            maxthreash (float): Maximum threshold for the second subset (time_diffs_A > maxthreash).
        """
        # Subset the data based on thresholds
        subset_low = S_nonzero_A[time_diffs_A < minthreash]
        subset_high = S_nonzero_A[time_diffs_A > maxthreash]
    
        # Ensure there are elements in each subset
        if len(subset_low) == 0 or len(subset_high) == 0:
            raise ValueError("One of the subsets is empty. Please check your data or thresholds.")
    
        # Combine subsets to determine common bin edges
        combined_data = np.concatenate((subset_low, subset_high))
        num_bins = 100  # Number of bins
        # Define bin edges based on the combined data
        bins = np.linspace(combined_data.min(), combined_data.max(), num_bins)
    
        # Create the plot
        plt.figure(figsize=(12, 7))
    
        # Plot the first subset (time_diffs_A < minthreash)
        counts_low, bins_low, patches_low = plt.hist(
            subset_low, bins=bins, density=True, histtype='step', linewidth=2, 
            label=f'time_diffs_A < {minthreash}'
        )
    
        # Plot the second subset (time_diffs_A > maxthreash)
        counts_high, bins_high, patches_high = plt.hist(
            subset_high, bins=bins, density=True, histtype='step', linewidth=2, 
            label=f'time_diffs_A > {maxthreash}'
        )
    
        # Calculate means
        mean_low = np.mean(subset_low)
        mean_high = np.mean(subset_high)
    
        # Plot vertical lines for means
        plt.axvline(mean_low, color='blue', linestyle='dashed', linewidth=2, 
                    label=f'Mean < {minthreash}: {mean_low:.2f}')
        plt.axvline(mean_high, color='orange', linestyle='dashed', linewidth=2, 
                    label=f'Mean > {maxthreash}: {mean_high:.2f}')
    
        # Perform statistical tests
        # 1. Independent t-test
        t_stat, p_value_t = stats.ttest_ind(subset_low, subset_high, equal_var=False)
    
        # 2. Mann-Whitney U test
        u_stat, p_value_u = stats.mannwhitneyu(subset_low, subset_high, alternative='two-sided')
    
        # Determine significance
        alpha = 0.05
        significance_t = 'Significant' if p_value_t < alpha else 'Not Significant'
        significance_u = 'Significant' if p_value_u < alpha else 'Not Significant'
    
        # Annotate the plot with test results
        plt.text(0.95, 0.15, f'T-test p-value: {p_value_t:.3e} ({significance_t})',
                 horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
                 fontsize=12, color='blue')
    
        plt.text(0.95, 0.10, f'Mann-Whitney U p-value: {p_value_u:.3e} ({significance_u})',
                 horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
                 fontsize=12, color='orange')
    
        # Customize the plot
        plt.xlabel('Connection Strength', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title('Distribution of Connection Strengths for Different Time Differences', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
    
        # Display the plot
        plt.tight_layout()
        plt.show()
        
    def z_score_with_zero_handling(A,dim=0):
        # Calculate mean and standard deviation along the first dimension (row-wise)
        mean_A = np.mean(A, axis=dim)
        std_A = np.std(A, axis=dim)
        # Avoid division by zero: if std_A is zero, set it to 1 to prevent invalid division
        std_A[std_A == 0] = 1
        # Calculate z-score
        z_scores = (A - mean_A) / std_A
        return z_scores    
    
    def extract_average_weights_no_bins(A, e, f, p):
        m, n = A.shape
        B = np.zeros(2 * p + 1)
        counts = np.zeros(2 * p + 1)
        
        # Initialize a list to collect weights for each offset
        offset_weights = [[] for _ in range(2 * p + 1)]
    
        for k in range(n):
            # Compute time differences for column k
            t = e - f[k]  # Shape: (m,)
            # Get indices sorted by absolute time difference
            sorted_indices = np.argsort(np.abs(t))
            # Loop over offsets from 0 to 2p
            for o in range(min(2 * p + 1, m)):
                i = sorted_indices[o]
                weight = A[i, k]
                offset_weights[o].append(weight)
        
        # Compute average weights for each offset
        for o in range(2 * p + 1):
            if offset_weights[o]:
                B[o] = np.mean(offset_weights[o])
                counts[o] = len(offset_weights[o])
            else:
                B[o] = np.nan  # No data for this offset
        
        return B
    
    def extract_average_weights(A, e, f, p):
        m, n = A.shape
        B = np.zeros(2 * p + 1)
        counts = np.zeros(2 * p + 1)  # To keep track of the number of weights in each bin
    
        # Compute the differences in firing times (m x n matrix)
        delta_t = e[:, np.newaxis] - f[np.newaxis, :]  # Shape: (m, n)
    
        # Define bin edges from -p - 0.5 to p + 0.5 to capture all differences
        bin_edges = np.linspace(-p - 0.5, p + 0.5, 2 * p + 2)
    
        # Flatten the differences and corresponding weights
        delta_t_flat = delta_t.flatten()
        weights_flat = A.flatten()
    
        # Assign each difference to a bin
        bin_indices = np.digitize(delta_t_flat, bins=bin_edges) - 1  # Subtract 1 to get bin indices from 0 to 2p
    
        # Filter out differences that fall outside our bins
        valid_indices = np.where((bin_indices >= 0) & (bin_indices < 2 * p + 1))
    
        bin_indices = bin_indices[valid_indices]
        weights_valid = weights_flat[valid_indices]
    
        # Initialize a list to collect weights for each bin
        bin_weights = [[] for _ in range(2 * p + 1)]
    
        # Collect weights into bins
        for idx, bin_idx in enumerate(bin_indices):
            bin_weights[bin_idx].append(weights_valid[idx])
    
        # Compute the average weights for each bin
        for b in range(2 * p + 1):
            if bin_weights[b]:
                B[b] = np.mean(bin_weights[b])
                counts[b] = len(bin_weights[b])
            else:
                B[b] = np.nan  # No data for this bin
    
        return B
    
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)