"""
Network Scheduler module for the network simulation environment.
"""
import numpy as np
import matplotlib.pyplot as plt
from traffic_generator import TrafficGenerator

class NetworkScheduler:
    """Manages traffic and distributes packets to access points."""
    
    def __init__(self, access_points):
        self.access_points = access_points
        self.traffic_generator = TrafficGenerator()
        self.current_time = 0
        self.time_step = 0.1  # seconds
        
        # Traffic pattern settings
        self.traffic_patterns = {
            'smooth': {'enabled': False, 'rate': 50},  # packets per second
            'periodic': {'enabled': False, 'base_rate': 30, 'period': 10, 'amplitude': 20},
            'burst': {'enabled': False, 'base_rate': 20, 'burst_prob': 0.05, 'burst_size': 50}
        }
        
        # Performance metrics
        self.total_generated_packets = 0
        self.total_processed_packets = 0
        self.total_dropped_packets = 0
        self.throughput_history = []
        self.delay_history = []
        self.packet_loss_history = []
        
        # For visualization
        self.ap_load_history = {ap.ap_id: [] for ap in access_points}
        self.bandwidth_allocation_history = {ap.ap_id: [] for ap in access_points}
        self.time_history = []
    
    def enable_traffic_pattern(self, pattern_name, **kwargs):
        """Enable a specific traffic pattern with optional parameters."""
        if pattern_name not in self.traffic_patterns:
            raise ValueError(f"Unknown traffic pattern: {pattern_name}")
        
        self.traffic_patterns[pattern_name]['enabled'] = True
        
        # Update parameters if provided
        for key, value in kwargs.items():
            if key in self.traffic_patterns[pattern_name]:
                self.traffic_patterns[pattern_name][key] = value
    
    def disable_traffic_pattern(self, pattern_name):
        """Disable a specific traffic pattern."""
        if pattern_name in self.traffic_patterns:
            self.traffic_patterns[pattern_name]['enabled'] = False
    
    def allocate_bandwidth(self, bandwidth_allocation):
        """Allocate bandwidth across access points.
        
        Args:
            bandwidth_allocation: Dictionary mapping AP IDs to bandwidth values
        """
        for ap in self.access_points:
            if ap.ap_id in bandwidth_allocation:
                ap.set_bandwidth(bandwidth_allocation[ap.ap_id])
    
    def distribute_packet(self, packet, scheduler=None, scheduler_args=None):
        """Distribute a packet to an access point using the provided scheduler.
        
        Args:
            packet: The packet to distribute
            scheduler: A function that selects an access point or None for default behavior
            scheduler_args: Additional arguments to pass to the scheduler function
            
        Returns:
            Boolean indicating if packet was successfully enqueued
        """
        # Use provided scheduler if available
        if scheduler is not None:
            if scheduler_args is None:
                scheduler_args = {}
            
            # Call the provided scheduler function
            selected_ap = scheduler(
                packet=packet, 
                access_points=self.access_points, 
                current_time=self.current_time,
                total_packets=self.total_generated_packets,
                **scheduler_args
            )
        else:
            # Default to least loaded as a fallback
            selected_ap = min(self.access_points, key=lambda ap: ap.current_load)
        
        # Try to enqueue the packet
        return selected_ap.enqueue_packet(packet, self.current_time)
    
    def step(self, scheduler=None, scheduler_args=None):
        """Advance the simulation by one time step.
        
        Args:
            scheduler: A function that selects an access point or None for default behavior
            scheduler_args: Additional arguments to pass to the scheduler function
        """
        # Generate traffic based on enabled patterns
        generated_packets = []
        
        if self.traffic_patterns['smooth']['enabled']:
            smooth_packets = self.traffic_generator.generate_smooth_traffic(
                current_time=self.current_time,
                time_step=self.time_step,
                rate=self.traffic_patterns['smooth']['rate']
            )
            generated_packets.extend(smooth_packets)
        
        if self.traffic_patterns['periodic']['enabled']:
            periodic_packets = self.traffic_generator.generate_periodic_traffic(
                current_time=self.current_time,
                time_step=self.time_step,
                base_rate=self.traffic_patterns['periodic']['base_rate'],
                period=self.traffic_patterns['periodic']['period'],
                amplitude=self.traffic_patterns['periodic']['amplitude']
            )
            generated_packets.extend(periodic_packets)
        
        if self.traffic_patterns['burst']['enabled']:
            burst_packets = self.traffic_generator.generate_burst_traffic(
                current_time=self.current_time,
                time_step=self.time_step,
                base_rate=self.traffic_patterns['burst']['base_rate'],
                burst_prob=self.traffic_patterns['burst']['burst_prob'],
                burst_size=self.traffic_patterns['burst']['burst_size']
            )
            generated_packets.extend(burst_packets)
        
        # Distribute packets to APs
        for packet in generated_packets:
            self.distribute_packet(packet, scheduler, scheduler_args)
        
        # Process packets at each AP
        total_processed_this_step = 0
        for ap in self.access_points:
            processed = ap.process_packets(self.current_time, self.time_step)
            total_processed_this_step += processed
        
        # Update metrics
        self.total_generated_packets += len(generated_packets)
        self.total_processed_packets += total_processed_this_step
        
        # Calculate current throughput (in packets/second)
        current_throughput = total_processed_this_step / self.time_step
        self.throughput_history.append(current_throughput)
        
        # Update other metrics for visualization
        total_delay = sum(ap.total_delay for ap in self.access_points)
        total_processed = sum(ap.processed_packets for ap in self.access_points)
        total_dropped = sum(ap.dropped_packets for ap in self.access_points)
        
        # Calculate average delay
        if total_processed > 0:
            avg_delay = total_delay / total_processed
        else:
            avg_delay = 0
        
        # Calculate packet loss rate
        if self.total_generated_packets > 0:
            packet_loss_rate = total_dropped / self.total_generated_packets * 100
        else:
            packet_loss_rate = 0
        
        self.delay_history.append(avg_delay)
        self.packet_loss_history.append(packet_loss_rate)
        self.time_history.append(self.current_time)
        
        # Record AP loads and bandwidth allocations
        for ap in self.access_points:
            self.ap_load_history[ap.ap_id].append(ap.current_load)
            self.bandwidth_allocation_history[ap.ap_id].append(ap.current_bandwidth)
        
        # Advance time
        self.current_time += self.time_step
        
        return {
            'time': self.current_time,
            'throughput': current_throughput,
            'avg_delay': avg_delay,
            'packet_loss_rate': packet_loss_rate,
            'ap_stats': [ap.get_stats() for ap in self.access_points]
        }
    
    def get_state(self):
        """Return the current state of the network for the RL agent."""
        state = []
        
        # Add AP states
        for ap in self.access_points:
            state.extend([
                ap.current_load / 100,  # Normalize to [0, 1]
                ap.current_bandwidth / ap.capacity,  # Normalize to [0, 1]
                len(ap.buffer) / ap.buffer_size  # Buffer utilization [0, 1]
            ])
        
        return np.array(state)
    
    def get_metrics(self):
        """Get the current performance metrics."""
        total_delay = sum(ap.total_delay for ap in self.access_points)
        total_processed = sum(ap.processed_packets for ap in self.access_points)
        total_dropped = sum(ap.dropped_packets for ap in self.access_points)
        
        # Calculate average delay
        if total_processed > 0:
            avg_delay = total_delay / total_processed
        else:
            avg_delay = 0
        
        # Calculate packet loss rate
        if self.total_generated_packets > 0:
            packet_loss_rate = total_dropped / self.total_generated_packets * 100
        else:
            packet_loss_rate = 0
        
        # Calculate current throughput (packets per second)
        # Using the average of the last 10 steps
        if len(self.throughput_history) > 0:
            current_throughput = sum(self.throughput_history[-10:]) / min(10, len(self.throughput_history))
        else:
            current_throughput = 0
        
        return {
            'throughput': current_throughput,
            'avg_delay': avg_delay,
            'packet_loss_rate': packet_loss_rate,
            'total_processed': total_processed,
            'total_dropped': total_dropped,
            'total_generated': self.total_generated_packets
        }
    
    def visualize_performance(self, save_path=None):
        """Visualize network performance metrics."""
        fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        
        # Throughput over time
        axs[0].plot(self.time_history, self.throughput_history, 'b-')
        axs[0].set_title('Throughput over Time')
        axs[0].set_ylabel('Packets/second')
        axs[0].grid(True)
        
        # Delay over time
        axs[1].plot(self.time_history, self.delay_history, 'r-')
        axs[1].set_title('Average Delay over Time')
        axs[1].set_ylabel('Delay (seconds)')
        axs[1].grid(True)
        
        # Packet loss rate over time
        axs[2].plot(self.time_history, self.packet_loss_history, 'g-')
        axs[2].set_title('Packet Loss Rate over Time')
        axs[2].set_ylabel('Loss Rate (%)')
        axs[2].grid(True)
        
        # AP loads over time
        for ap_id, loads in self.ap_load_history.items():
            axs[3].plot(self.time_history, loads, label=f'AP {ap_id}')
        
        axs[3].set_title('Access Point Loads over Time')
        axs[3].set_xlabel('Time (seconds)')
        axs[3].set_ylabel('Load (%)')
        axs[3].legend()
        axs[3].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def visualize_bandwidth_allocation(self, save_path=None):
        """Visualize bandwidth allocation across access points."""
        plt.figure(figsize=(12, 6))
        
        for ap_id, bandwidths in self.bandwidth_allocation_history.items():
            plt.plot(self.time_history, bandwidths, label=f'AP {ap_id}')
        
        plt.title('Bandwidth Allocation over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Bandwidth (Mbps)')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()