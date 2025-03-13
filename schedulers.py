"""
Packet scheduling algorithms for the network simulation environment.
"""
import random

def round_robin_scheduler(packet, access_points, current_time, total_packets, **kwargs):
    """Simple round-robin packet scheduler."""
    ap_index = total_packets % len(access_points)
    return access_points[ap_index]

def least_loaded_scheduler(packet, access_points, current_time, total_packets, **kwargs):
    """Selects the AP with the lowest current load."""
    return min(access_points, key=lambda ap: ap.current_load)

def random_scheduler(packet, access_points, current_time, total_packets, **kwargs):
    """Randomly selects an AP."""
    return random.choice(access_points)

def fifo_scheduler(packet, access_points, current_time, total_packets, buffer_queues=None, **kwargs):
    """First-In-First-Out scheduler with a global queue.
    
    This is a stateful scheduler that requires keeping track of a queue outside.
    The buffer_queues parameter should be a dictionary mapping AP IDs to their respective queues.
    """
    if buffer_queues is None:
        # If no queues are provided, default to least loaded
        return least_loaded_scheduler(packet, access_points, current_time, total_packets)
    
    # Get the AP with the emptiest queue
    ap_with_shortest_queue = min(access_points, key=lambda ap: len(buffer_queues.get(ap.ap_id, [])))
    
    # Add packet to that AP's queue
    if ap_with_shortest_queue.ap_id not in buffer_queues:
        buffer_queues[ap_with_shortest_queue.ap_id] = []
    
    buffer_queues[ap_with_shortest_queue.ap_id].append(packet)
    
    return ap_with_shortest_queue

def weighted_scheduler(packet, access_points, current_time, total_packets, weights=None, **kwargs):
    """Weighted selection based on provided weights."""
    if weights is None:
        # Default weights proportional to capacity
        weights = [ap.capacity for ap in access_points]
    
    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    # Weighted random selection
    r = random.random()
    cumulative_weight = 0
    
    for i, weight in enumerate(normalized_weights):
        cumulative_weight += weight
        if r <= cumulative_weight:
            return access_points[i]
    
    # Fallback (shouldn't reach here)
    return access_points[-1]

def packet_size_based_scheduler(packet, access_points, current_time, total_packets, **kwargs):
    """Assigns packets based on their size and AP capacity."""
    # Normalize packet size
    max_packet_size = 10  # Assuming max packet size is 10KB
    normalized_size = packet.size / max_packet_size
    
    if normalized_size < 0.3:
        # Small packets go to low capacity APs
        candidate_aps = sorted(access_points, key=lambda ap: ap.capacity)
    elif normalized_size < 0.7:
        # Medium packets go to medium capacity APs
        candidate_aps = sorted(access_points, key=lambda ap: abs(ap.capacity - 75))
    else:
        # Large packets go to high capacity APs
        candidate_aps = sorted(access_points, key=lambda ap: -ap.capacity)
    
    # Among candidates, choose the least loaded
    return min(candidate_aps[:max(1, len(candidate_aps) // 2)], key=lambda ap: ap.current_load)

def pattern_aware_scheduler(packet, access_points, current_time, total_packets, **kwargs):
    """Assigns packets based on their traffic pattern."""
    if packet.pattern_type == 'burst':
        # Burst traffic goes to highest capacity APs
        return max(access_points, key=lambda ap: ap.capacity)
    elif packet.pattern_type == 'periodic':
        # Periodic traffic is distributed based on current load
        return min(access_points, key=lambda ap: ap.current_load)
    else:  # smooth and others
        # Round-robin for smooth traffic
        ap_index = total_packets % len(access_points)
        return access_points[ap_index]