"""
Traffic Generator module for the network simulation environment.
"""
import numpy as np
import random
from packet import Packet

class TrafficGenerator:
    """Generates network traffic with different patterns."""
    
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Default packet size ranges
        self.packet_size_range = (1, 10)  # KB
        self.next_packet_id = 0
    
    def generate_smooth_traffic(self, current_time, time_step, rate):
        """Generate smooth, consistent traffic.
        
        Args:
            current_time: Current simulation time
            time_step: Simulation time step
            rate: Average packet rate (packets per second)
            
        Returns:
            List of generated packets
        """
        # Calculate number of packets for this time step
        num_packets = int(rate * time_step)
        
        packets = []
        for _ in range(num_packets):
            size = random.uniform(*self.packet_size_range)
            packet = Packet(
                packet_id=self.next_packet_id,
                size=size,
                creation_time=current_time,
                pattern_type='smooth'
            )
            packets.append(packet)
            self.next_packet_id += 1
        
        return packets
    
    def generate_periodic_traffic(self, current_time, time_step, base_rate, period=10, amplitude=5):
        """Generate traffic with periodic highs and lows.
        
        Args:
            current_time: Current simulation time
            time_step: Simulation time step
            base_rate: Base packet rate (packets per second)
            period: Period of the cycle in seconds
            amplitude: Amplitude of the cycle (additional packets)
            
        Returns:
            List of generated packets
        """
        # Calculate current rate based on sine wave
        current_phase = (current_time % period) / period * 2 * np.pi
        rate_modifier = amplitude * np.sin(current_phase)
        current_rate = max(0, base_rate + rate_modifier)
        
        # Calculate number of packets for this time step
        num_packets = int(current_rate * time_step)
        
        packets = []
        for _ in range(num_packets):
            size = random.uniform(*self.packet_size_range)
            packet = Packet(
                packet_id=self.next_packet_id,
                size=size,
                creation_time=current_time,
                pattern_type='periodic'
            )
            packets.append(packet)
            self.next_packet_id += 1
        
        return packets
    
    def generate_burst_traffic(self, current_time, time_step, base_rate, burst_prob=0.05, burst_size=20):
        """Generate traffic with occasional bursts.
        
        Args:
            current_time: Current simulation time
            time_step: Simulation time step
            base_rate: Base packet rate (packets per second)
            burst_prob: Probability of a burst occurring
            burst_size: Size of the burst (in packets)
            
        Returns:
            List of generated packets
        """
        # Calculate base number of packets
        base_packets = int(base_rate * time_step)
        
        # Determine if a burst occurs
        burst_packets = 0
        if random.random() < burst_prob:
            burst_packets = burst_size
        
        total_packets = base_packets + burst_packets
        
        packets = []
        for _ in range(total_packets):
            size = random.uniform(*self.packet_size_range)
            packet = Packet(
                packet_id=self.next_packet_id,
                size=size,
                creation_time=current_time,
                pattern_type='burst' if burst_packets > 0 else 'base'
            )
            packets.append(packet)
            self.next_packet_id += 1
            
            # Decrement burst packets counter
            if burst_packets > 0:
                burst_packets -= 1
        
        return packets