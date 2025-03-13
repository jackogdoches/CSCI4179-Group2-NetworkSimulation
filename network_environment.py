"""
Network Environment module for the network simulation environment.
"""
import numpy as np
import random
import time
from access_point import AccessPoint
from network_scheduler import NetworkScheduler

class NetworkEnvironment:
    """Main simulation environment that can interface with RL agents."""
    
    def __init__(self, num_aps=3, ap_capacities=None, seed=None):
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Set default capacities if not provided
        if ap_capacities is None:
            ap_capacities = [50, 75, 100]  # Mbps
        
        # Ensure we have enough capacities for all APs
        if len(ap_capacities) < num_aps:
            # Extend with random capacities in the same range
            min_cap = min(ap_capacities)
            max_cap = max(ap_capacities)
            additional_caps = [random.uniform(min_cap, max_cap) for _ in range(num_aps - len(ap_capacities))]
            ap_capacities.extend(additional_caps)
        
        # Create access points
        self.access_points = [
            AccessPoint(ap_id=i, capacity=ap_capacities[i])
            for i in range(num_aps)
        ]
        
        # Create network scheduler
        self.scheduler = NetworkScheduler(self.access_points)
        
        # Environment settings
        self.max_steps = 1000
        self.current_step = 0
        
        # Action and observation spaces (for RL)
        self.num_aps = num_aps
        self.action_space_shape = num_aps  # Bandwidth allocation per AP
        self.observation_space_shape = num_aps * 3  # 3 features per AP
    
    def reset(self, traffic_patterns=None):
        """Reset the environment to initial state."""
        # Reset access points
        for ap in self.access_points:
            ap.current_load = 0
            ap.buffer.clear()
            ap.processed_packets = 0
            ap.dropped_packets = 0
            ap.total_delay = 0
            ap.current_bandwidth = ap.capacity  # Reset to full capacity
        
        # Reset scheduler
        self.scheduler = NetworkScheduler(self.access_points)
        
        # Set traffic patterns if provided
        if traffic_patterns:
            for pattern, settings in traffic_patterns.items():
                if settings.get('enabled', False):
                    self.scheduler.enable_traffic_pattern(pattern, **settings)
        
        self.current_step = 0
        
        return self.scheduler.get_state()
    
    def step(self, action=None, scheduler=None, scheduler_args=None):
        """Take a step in the environment.
        
        Args:
            action: Bandwidth allocation across APs (if None, uses current allocation)
            scheduler: A function that selects an access point or None for default behavior
            scheduler_args: Additional arguments to pass to the scheduler function
            
        Returns:
            next_state: The new state after the action
            reward: The reward for this step
            done: Whether the episode is done
            info: Additional information
        """
        # Apply action (bandwidth allocation) if provided
        if action is not None:
            bandwidth_allocation = {ap.ap_id: action[i] for i, ap in enumerate(self.access_points)}
            self.scheduler.allocate_bandwidth(bandwidth_allocation)
        
        # Step the scheduler with the provided packet scheduler
        step_info = self.scheduler.step(scheduler=scheduler, scheduler_args=scheduler_args)
        
        # Get new state
        next_state = self.scheduler.get_state()
        
        # Calculate reward based on performance metrics
        metrics = self.scheduler.get_metrics()
        
        # Simple reward: maximize throughput, minimize delay and packet loss
        throughput_reward = metrics['throughput'] / 100  # Normalize
        delay_penalty = -metrics['avg_delay']
        loss_penalty = -metrics['packet_loss_rate'] / 10  # Normalize
        
        reward = throughput_reward + delay_penalty + loss_penalty
        
        # Check if episode is done
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return next_state, reward, done, step_info
    
    def run_episode(self, traffic_patterns, agent=None, scheduler=None, scheduler_args=None, render=False, render_interval=100):
        """Run a complete episode with specified traffic patterns.
        
        Args:
            traffic_patterns: Dictionary of traffic patterns to enable
            agent: Optional RL agent to make bandwidth allocation decisions
            scheduler: A function that selects an access point or None for default behavior
            scheduler_args: Additional arguments to pass to the scheduler function
            render: Whether to render performance metrics
            render_interval: How often to render (in steps)
            
        Returns:
            Final performance metrics
        """
        # Reset environment and set traffic patterns
        state = self.reset(traffic_patterns)
        
        done = False
        step_counter = 0
        
        while not done:
            # Get action from agent if provided
            if agent:
                action = agent.get_action(state)
            else:
                # Default to equal distribution
                action = [ap.capacity for ap in self.access_points]
            
            # Take a step with the provided scheduler
            state, reward, done, info = self.step(action, scheduler=scheduler, scheduler_args=scheduler_args)
            
            # Update agent if provided
            if agent:
                agent.update(state, reward, done)
            
            # Render if requested
            if render and step_counter % render_interval == 0:
                self.render()
            
            step_counter += 1
        
        # Final render if requested
        if render:
            self.render()
        
        # Return final metrics
        return self.scheduler.get_metrics()
    
    def render(self):
        """Render the current state of the environment."""
        # Clear the screen (for terminal-based rendering)
        print("\033c", end="")
        
        # Print current time and step
        print(f"Time: {self.scheduler.current_time:.2f}s | Step: {self.current_step}/{self.max_steps}")
        
        # Print performance metrics
        metrics = self.scheduler.get_metrics()
        print("\nPerformance Metrics:")
        print(f"  Throughput: {metrics['throughput']:.2f} packets/s")
        print(f"  Average Delay: {metrics['avg_delay']:.4f}s")
        print(f"  Packet Loss Rate: {metrics['packet_loss_rate']:.2f}%")
        
        # Print AP statistics
        print("\nAccess Point Statistics:")
        for ap in self.access_points:
            stats = ap.get_stats()
            print(f"  AP {stats['ap_id']} | Load: {stats['current_load']:.2f}% | " 
                  f"Bandwidth: {stats['current_bandwidth']:.2f} Mbps | "
                  f"Buffer: {len(ap.buffer)}/{ap.buffer_size}")
        
        # Short pause to make output readable
        time.sleep(0.1)
    
    def visualize_results(self):
        """Visualize the results of the simulation."""
        self.scheduler.visualize_performance()
        self.scheduler.visualize_bandwidth_allocation()