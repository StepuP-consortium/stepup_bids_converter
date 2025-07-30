"""
Motion capture marker fragment merging with strict temporal non-overlap enforcement.
Customized for coordinate system where:
- X: Back (positive) to front (negative)
- Y: Right (positive) to left (negative)
- Z: Bottom (positive) to up (negative)

Author: JuliusWelzel
Date: 2025-07-03 08:03:49
"""

import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

def merge_marker_fragments(marker_dict, n_expected_markers,
                          merge_distance=20.0, min_fragment_size=5, 
                          spatial_weight=5.0, temporal_weight=0.1):
    """
    Post-process marker data to merge fragments belonging to the same physical marker,
    with strict enforcement of temporal non-overlap and emphasis on spatial similarity.
    
    Args:
        marker_dict (dict): Dictionary with marker IDs as keys, each containing:
                           'data': array of [x, y, z, marker_id]
                           'indices': tuple containing array of time indices
        n_expected_markers (int): Expected number of true markers
        jump_threshold (float): Distance threshold to detect marker jumps (mm)
        merge_distance (float): Distance threshold for merging markers (mm)
        min_fragment_size (int): Minimum size of fragments to consider (filters noise)
        spatial_weight (float): Weight factor for spatial features (higher = more emphasis)
        temporal_weight (float): Weight factor for temporal features (lower = less emphasis)
        
    Returns:
        dict: Merged marker dictionary with same structure
    """
    print(f"Starting marker fragment merging with STRICT temporal non-overlap enforcement.")
    print(f"Expected markers: {n_expected_markers}, Spatial weight: {spatial_weight}, Temporal weight: {temporal_weight}")
    print(f"Using coordinate system: X(back+/front-), Y(right+/left-), Z(bottom+/up-)")
    
    # Check if input dictionary is valid
    if not marker_dict:
        print("Error: Input marker dictionary is empty.")
        return {}
    
    # Get indices array from the tuple
    def get_indices_array(marker_data):
        if isinstance(marker_data['indices'], tuple) and len(marker_data['indices']) > 0:
            return marker_data['indices'][0]  # Extract array from tuple
        return marker_data['indices']  # In case it's already an array
    
    # Display sample data structure
    sample_key = list(marker_dict.keys())[0]
    print(f"Sample data shape: {marker_dict[sample_key]['data'].shape}")
    indices_array = get_indices_array(marker_dict[sample_key])
    print(f"Sample indices shape: {indices_array.shape}")
    
    # Filter out very small fragments (likely noise)
    filtered_dict = {k: v for k, v in marker_dict.items() 
                     if len(get_indices_array(v)) >= min_fragment_size}
    
    if len(filtered_dict) < len(marker_dict):
        print(f"Filtered out {len(marker_dict) - len(filtered_dict)} small fragments")
    
    if not filtered_dict:
        print("Error: No marker fragments left after filtering.")
        return {}
    
    # CRITICAL: Build frame overlap index to quickly check if fragments overlap in time
    # Store the frame indices for each marker as sets for fast intersection checks
    frame_indices = {}
    for marker_id, marker_data in filtered_dict.items():
        indices = get_indices_array(marker_data)
        frame_indices[marker_id] = set(indices)
    
    # Function to check if two fragments have any temporal overlap
    def has_temporal_overlap(marker_id1, marker_id2):
        """Return True if markers have any overlapping frames"""
        return bool(frame_indices[marker_id1].intersection(frame_indices[marker_id2]))
    
    # Extract rich spatial features for each fragment
    fragment_features = {}
    for marker_id, marker_data in filtered_dict.items():
        data = marker_data['data']
        indices = get_indices_array(marker_data)
        
        # Extract XYZ positions
        xyz = data[:, :3]
        
        # Calculate spatial features
        mean_pos = np.mean(xyz, axis=0)
        min_pos = np.min(xyz, axis=0)
        max_pos = np.max(xyz, axis=0)
        range_pos = max_pos - min_pos
        std_pos = np.std(xyz, axis=0)
        
        # Height information (Z-axis in this coordinate system)
        height = mean_pos[2]  # Z is height (bottom to up)
        
        # Calculate lateral position (Y-axis in this coordinate system)
        lateral_position = mean_pos[1]  # Y is lateral (right to left)
        
        # Calculate anterior-posterior position (X-axis)
        ap_position = mean_pos[0]  # X is anterior-posterior (back to front)
        
        # Calculate temporal features
        start_index = indices[0]
        end_index = indices[-1]
        duration = end_index - start_index
        
        # Store all features
        fragment_features[marker_id] = {
            'marker_id': marker_id,
            'mean_position': mean_pos,
            'min_position': min_pos,
            'max_position': max_pos,
            'range': range_pos,
            'std': std_pos,
            'height': height,  # Z-axis
            'lateral_position': lateral_position,  # Y-axis
            'ap_position': ap_position,  # X-axis
            'start_index': start_index,
            'end_index': end_index,
            'duration': duration,
            'start_position': xyz[0],
            'end_position': xyz[-1],
            'data_length': len(indices),
            'indices': indices
        }
    
    # Create initial connectivity graph based on spatial similarity and temporal non-overlap
    connectivity_graph = defaultdict(list)
    
    # Check all pairs of fragments for compatibility
    marker_ids = list(fragment_features.keys())
    n_markers = len(marker_ids)
    
    print(f"Checking temporal overlap between {n_markers} fragments...")
    
    # Track overlap statistics
    total_pairs = 0
    overlapping_pairs = 0
    
    for i in range(n_markers):
        for j in range(i+1, n_markers):
            marker_id1 = marker_ids[i]
            marker_id2 = marker_ids[j]
            
            total_pairs += 1
            
            # STRICT CHECK: If fragments have ANY overlapping frames, they CANNOT be merged
            if has_temporal_overlap(marker_id1, marker_id2):
                overlapping_pairs += 1
                continue
            
            # Calculate spatial similarity only if there's no temporal overlap
            pos1 = fragment_features[marker_id1]['mean_position']
            pos2 = fragment_features[marker_id2]['mean_position']
            
            # Spatial distance between mean positions
            distance = np.linalg.norm(pos1 - pos2)
            
            # If they're spatially close, add connection
            if distance < merge_distance * 2:  # Using a wider threshold for initial connectivity
                connectivity_graph[marker_id1].append((marker_id2, distance))
                connectivity_graph[marker_id2].append((marker_id1, distance))
    
    print(f"Found {overlapping_pairs} overlapping pairs out of {total_pairs} total pairs ({overlapping_pairs/total_pairs*100:.1f}%)")
    
    # Use connected components to form initial clusters
    def find_connected_components(graph):
        visited = set()
        components = []
        
        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor, _ in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for node in graph:
            if node not in visited:
                component = []
                dfs(node, component)
                components.append(component)
        
        return components
    
    initial_clusters = find_connected_components(connectivity_graph)
    print(f"Found {len(initial_clusters)} connected components")
    
    # If we have more components than expected markers, merge based on spatial similarity
    if len(initial_clusters) > n_expected_markers:
        print(f"Too many clusters ({len(initial_clusters)}), merging to {n_expected_markers} using KMeans")
        
        # Prepare data for KMeans - using custom feature vectors with appropriate weighting
        cluster_features = []
        cluster_indices = []
        
        for i, cluster in enumerate(initial_clusters):
            # Calculate feature vector for this cluster
            positions = np.array([fragment_features[mid]['mean_position'] for mid in cluster])
            mean_pos = np.mean(positions, axis=0)
            
            # Height and lateral position are key for distinguishing markers
            heights = np.array([fragment_features[mid]['height'] for mid in cluster])
            mean_height = np.mean(heights)
            
            lateral_positions = np.array([fragment_features[mid]['lateral_position'] for mid in cluster])
            mean_lateral = np.mean(lateral_positions)
            
            ap_positions = np.array([fragment_features[mid]['ap_position'] for mid in cluster])
            mean_ap = np.mean(ap_positions)
            
            # Create weighted feature vector with emphasis on spatial properties
            feature_vector = np.array([
                mean_pos[0] * spatial_weight,      # X - weighted 
                mean_pos[1] * spatial_weight * 2,  # Y - double weight for lateral position
                mean_pos[2] * spatial_weight * 2,  # Z - double weight for height
                mean_height * spatial_weight,      # Additional height emphasis
                mean_lateral * spatial_weight      # Additional lateral emphasis
            ])
            
            cluster_features.append(feature_vector)
            cluster_indices.append(i)
        
        # Apply KMeans to merge clusters
        feature_array = np.array(cluster_features)
        kmeans = KMeans(n_clusters=n_expected_markers, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feature_array)
        
        # Create new merged clusters
        merged_clusters = [[] for _ in range(n_expected_markers)]
        for i, label in enumerate(labels):
            merged_clusters[label].extend(initial_clusters[i])
        
        final_clusters = merged_clusters
    else:
        final_clusters = initial_clusters
    
    # Convert to dictionary format
    clusters = {i: cluster for i, cluster in enumerate(final_clusters)}
    
    # Final verification: Make sure no cluster contains temporally overlapping fragments
    verified_clusters = {}
    
    for cluster_id, marker_ids in clusters.items():
        # Recheck all pairs within each cluster for temporal overlap
        valid_markers = []
        conflict_groups = []
        
        # Group by overlapping frames
        remaining_markers = set(marker_ids)
        while remaining_markers:
            current = remaining_markers.pop()
            current_group = {current}
            
            # Find all markers that overlap with current
            overlapping = set()
            for other in list(remaining_markers):
                if has_temporal_overlap(current, other):
                    overlapping.add(other)
            
            # If we found overlapping markers, we need to form a conflict group
            if overlapping:
                current_group.update(overlapping)
                remaining_markers -= overlapping
                conflict_groups.append(current_group)
            else:
                valid_markers.append(current)
        
        # Handle conflict groups - keep only one marker from each conflict group
        for group in conflict_groups:
            # Select the marker with the most data points
            best_marker = max(group, key=lambda mid: fragment_features[mid]['data_length'])
            valid_markers.append(best_marker)
            print(f"Conflict in cluster {cluster_id}: keeping marker {best_marker} from group {group}")
        
        verified_clusters[cluster_id] = valid_markers
    
    # Efficiently create the merged marker dictionary
    merged_dict = {}
    
    for cluster_id, marker_ids in verified_clusters.items():
        if not marker_ids:
            continue
        
        # Pre-compute sizes to minimize memory allocations
        total_points = sum(len(get_indices_array(filtered_dict[marker_id])) for marker_id in marker_ids)
        
        if total_points == 0:
            continue
            
        # Pre-allocate arrays
        all_data = np.zeros((total_points, filtered_dict[marker_ids[0]]['data'].shape[1]), 
                           dtype=filtered_dict[marker_ids[0]]['data'].dtype)
        all_indices = np.zeros(total_points, dtype=get_indices_array(filtered_dict[marker_ids[0]]).dtype)
        
        # Fill arrays
        offset = 0
        for marker_id in marker_ids:
            data = filtered_dict[marker_id]['data']
            indices = get_indices_array(filtered_dict[marker_id])
            length = len(indices)
            
            all_data[offset:offset+length] = data
            all_indices[offset:offset+length] = indices
            offset += length
        
        # Sort by index in one vectorized operation
        sort_idx = np.argsort(all_indices)
        sorted_data = all_data[sort_idx]
        sorted_indices = all_indices[sort_idx]
        
        # Update marker_id in the data to be the cluster_id (vectorized)
        sorted_data[:, 3] = cluster_id
        
        # Store in the result dictionary with the same structure as input
        merged_dict[cluster_id] = {
            'data': sorted_data,
            'indices': (sorted_indices,)  # Match original tuple format
        }
        
        # Calculate average position for reporting
        avg_pos = np.mean(sorted_data[:, :3], axis=0)
        print(f"Cluster {cluster_id}: merged {len(marker_ids)} fragments, position [X:{avg_pos[0]:.1f}, Y:{avg_pos[1]:.1f}, Z:{avg_pos[2]:.1f}]")
    
    print(f"Final result: {len(merged_dict)} marker trajectories")
    return merged_dict

# Example usage:
# merged_markers = merge_marker_fragments(marker_dict, n_expected_markers=5)