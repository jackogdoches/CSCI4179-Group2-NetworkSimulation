"""
Access Point module for the network simulation environment.
"""
from collections import deque

class AccessPoint:
    """Represents an access point in the network."""
    
    def __init__(self, ap_id, capacity, buffer_size=100):
        self.ap_id = ap_id
        self.capacity = capacity  # in Mbps
        self.current_load = 0
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.processed_packets = 0
        self.dropped_packets = 0
        self.total_delay = 0
        self.current_bandwidth = capacity  # Initially set to max capacity
    
    def is_buffer_full(self):
        """Check if the packet buffer is full."""
        return len(self.buffer) >= self.buffer_size
    
    def enqueue_packet(self, packet, current_time):
        """Add a packet to the AP's buffer."""
        if self.is_buffer_full():
            self.dropped_packets += 1
            packet.dropped = True
            return False
        
        packet.ap_id = self.ap_id
        self.buffer.append(packet)
        return True
    
    def process_packets(self, current_time, time_step):
        """Process packets from the buffer based on current bandwidth."""
        # Calculate how many bytes can be processed in this time step
        bytes_to_process = (self.current_bandwidth * time_step) * 125  # Convert Mbps to KB/s
        
        bytes_processed = 0
        packets_processed = 0
        
        while self.buffer and bytes_processed < bytes_to_process:
            packet = self.buffer[0]
            
            # If we can process the entire packet
            if bytes_processed + packet.size <= bytes_to_process:
                self.buffer.popleft()
                packet.arrival_time = current_time
                packet.processed = True
                
                # Calculate delay for this packet
                delay = packet.arrival_time - packet.creation_time
                self.total_delay += delay
                
                bytes_processed += packet.size
                packets_processed += 1
                self.processed_packets += 1
            else:
                # Cannot process more packets in this time step
                break
        
        # Update current load (as a percentage of capacity)
        self.current_load = len(self.buffer) / self.buffer_size * 100
        
        return packets_processed
    
    def set_bandwidth(self, bandwidth):
        """Set the current bandwidth allocation for this AP."""
        # Ensure bandwidth doesn't exceed capacity
        self.current_bandwidth = min(bandwidth, self.capacity)
    
    def get_stats(self):
        """Return performance statistics for this AP."""
        if self.processed_packets == 0:
            avg_delay = 0
        else:
            avg_delay = self.total_delay / self.processed_packets
        
        total_packets = self.processed_packets + self.dropped_packets
        if total_packets == 0:
            packet_loss_rate = 0
        else:
            packet_loss_rate = self.dropped_packets / total_packets * 100
        
        return {
            'ap_id': self.ap_id,
            'current_load': self.current_load,
            'processed_packets': self.processed_packets,
            'dropped_packets': self.dropped_packets,
            'avg_delay': avg_delay,
            'packet_loss_rate': packet_loss_rate,
            'current_bandwidth': self.current_bandwidth
        }