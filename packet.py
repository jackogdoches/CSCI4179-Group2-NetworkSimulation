"""
Packet module for the network simulation environment.
"""

class Packet:
    """Represents a network packet in the simulation."""
    
    def __init__(self, packet_id, size, creation_time, pattern_type):
        self.packet_id = packet_id
        self.size = size  # in KB
        self.creation_time = creation_time
        self.arrival_time = None
        self.processed = False
        self.dropped = False
        self.ap_id = None
        self.pattern_type = pattern_type  # 'smooth', 'periodic', or 'burst'
    
    def __repr__(self):
        return f"Packet(id={self.packet_id}, size={self.size}KB, pattern={self.pattern_type})"