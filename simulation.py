"""
Main simulation module that brings together all components.
"""
from network_environment import NetworkEnvironment
from schedulers import *
import numpy as np
import matplotlib.pyplot as plt
import time

def run_basic_simulation():
    """Run a basic simulation with mixed traffic patterns."""
    # Create environment with 3 APs of different capacities
    env = NetworkEnvironment(num_aps=3, ap_capacities=[50, 75, 100], seed=42)
    
    # Configure traffic patterns
    traffic_patterns = {
        'smooth': {'enabled': True, 'rate': 40},
        'periodic': {'enabled': True, 'base_rate': 20, 'period': 5, 'amplitude': 15},
        'burst': {'enabled': True, 'base_rate': 10, 'burst_prob': 0.1, 'burst_size': 30}
    }
    
    # Run a sample episode with least loaded scheduler
    print("Running simulation with least loaded scheduler...")
    final_metrics = env.run_episode(
        traffic_patterns, 
        scheduler=least_loaded_scheduler,
        render=True, 
        render_interval=50
    )
    
    print("\nFinal Metrics:")
    for key, value in final_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Visualize results
    env.visualize_results()


def run_traffic_pattern_experiment(pattern_name, scheduler=None, scheduler_args=None):
    """Run experiment with a specific traffic pattern."""
    # Create environment
    env = NetworkEnvironment(num_aps=3, ap_capacities=[50, 75, 100], seed=42)
    env.max_steps = 200  # Reduce to speed up experiments
    
    # Set traffic patterns (only enable the one we're testing)
    traffic_patterns = {
        'smooth': {'enabled': False, 'rate': 40},
        'periodic': {'enabled': False, 'base_rate': 20, 'period': 5, 'amplitude': 15},
        'burst': {'enabled': False, 'base_rate': 10, 'burst_prob': 0.1, 'burst_size': 30}
    }
    
    # Enable only the pattern we're testing
    traffic_patterns[pattern_name]['enabled'] = True
    
    # Get scheduler name for display
    scheduler_name = scheduler.__name__ if scheduler else "default"
    
    # Run experiment
    print(f"\nRunning experiment with {pattern_name} traffic pattern using {scheduler_name} scheduler...")
    final_metrics = env.run_episode(
        traffic_patterns, 
        scheduler=scheduler, 
        scheduler_args=scheduler_args,
        render=False
    )
    
    print(f"\nFinal Metrics for {pattern_name} traffic with {scheduler_name} scheduler:")
    for key, value in final_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Save visualization with pattern name and scheduler
    env.scheduler.visualize_performance(save_path=f"{pattern_name}_{scheduler_name}_performance.png")
    env.scheduler.visualize_bandwidth_allocation(save_path=f"{pattern_name}_{scheduler_name}_bandwidth.png")
    
    return final_metrics


def run_comparative_experiments():
    """Run experiments with different traffic patterns and compare results."""
    patterns = ['smooth', 'periodic', 'burst']
    results = {}
    
    for pattern in patterns:
        results[pattern] = run_traffic_pattern_experiment(
            pattern, 
            scheduler=round_robin_scheduler
        )
    
    # Create comparison visualization
    metrics_to_compare = ['throughput', 'avg_delay', 'packet_loss_rate']
    
    fig, ax = plt.subplots(len(metrics_to_compare), 1, figsize=(10, 12))
    
    for i, metric in enumerate(metrics_to_compare):
        metric_values = [results[pattern][metric] for pattern in patterns]
        ax[i].bar(patterns, metric_values)
        ax[i].set_title(f'{metric} Comparison')
        ax[i].set_ylabel(metric)
        ax[i].grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("traffic_pattern_comparison.png")
    plt.show()


def custom_bandwidth_allocation_experiment():
    """Experiment with different static bandwidth allocation strategies."""
    # Create environment
    env = NetworkEnvironment(num_aps=3, ap_capacities=[50, 75, 100], seed=42)
    env.max_steps = 300  # Longer experiment to see the effect
    
    # Enable all traffic patterns for a more realistic scenario
    traffic_patterns = {
        'smooth': {'enabled': True, 'rate': 20},
        'periodic': {'enabled': True, 'base_rate': 15, 'period': 5, 'amplitude': 10},
        'burst': {'enabled': True, 'base_rate': 5, 'burst_prob': 0.05, 'burst_size': 20}
    }
    
    # Define different bandwidth allocation strategies
    strategies = {
        'equal': [75, 75, 75],  # Equal allocation
        'proportional': [50, 75, 100],  # Proportional to capacity
        'inverse_load': None  # Will be determined dynamically
    }
    
    results = {}
    
    # Test each strategy
    for strategy_name, allocation in strategies.items():
        print(f"\nTesting {strategy_name} bandwidth allocation strategy...")
        
        # Reset environment
        state = env.reset(traffic_patterns)
        
        # Run simulation
        done = False
        step_metrics = []
        
        while not done:
            # For inverse_load strategy, allocate bandwidth inversely proportional to load
            if strategy_name == 'inverse_load':
                # Get current loads
                loads = [ap.current_load for ap in env.access_points]
                # Avoid division by zero
                adjusted_loads = [max(1.0, load) for load in loads]
                # Inverse - higher load gets less bandwidth
                inverse_loads = [1.0/load for load in adjusted_loads]
                # Normalize to use full capacity
                total_inverse = sum(inverse_loads)
                normalized_inverse = [inv/total_inverse for inv in inverse_loads]
                # Scale by capacity
                action = [normalized_inverse[i] * ap.capacity for i, ap in enumerate(env.access_points)]
            else:
                action = allocation
            
            # Take step
            state, reward, done, info = env.step(action)
            
            # Record metrics
            step_metrics.append(env.scheduler.get_metrics())
            
            # Print progress
            if len(step_metrics) % 50 == 0:
                print(f"  Step {len(step_metrics)}/{env.max_steps}")
        
        # Record final results
        results[strategy_name] = {
            'final_metrics': env.scheduler.get_metrics(),
            'throughput_history': env.scheduler.throughput_history,
            'delay_history': env.scheduler.delay_history,
            'packet_loss_history': env.scheduler.packet_loss_history
        }
        
        # Save visualizations
        env.scheduler.visualize_performance(save_path=f"{strategy_name}_performance.png")
    
    # Compare strategies
    plt.figure(figsize=(12, 8))
    
    # Plot throughput comparison
    plt.subplot(3, 1, 1)
    for strategy, data in results.items():
        plt.plot(data['throughput_history'], label=strategy)
    plt.title('Throughput Comparison')
    plt.ylabel('Packets/second')
    plt.legend()
    plt.grid(True)
    
    # Plot delay comparison
    plt.subplot(3, 1, 2)
    for strategy, data in results.items():
        plt.plot(data['delay_history'], label=strategy)
    plt.title('Delay Comparison')
    plt.ylabel('Seconds')
    plt.legend()
    plt.grid(True)
    
    # Plot packet loss comparison
    plt.subplot(3, 1, 3)
    for strategy, data in results.items():
        plt.plot(data['packet_loss_history'], label=strategy)
    plt.title('Packet Loss Comparison')
    plt.xlabel('Step')
    plt.ylabel('Loss Rate (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("bandwidth_strategy_comparison.png")
    plt.show()
    
    # Print final metrics for each strategy
    print("\nFinal Metrics Comparison:")
    metrics_to_compare = ['throughput', 'avg_delay', 'packet_loss_rate']
    
    for metric in metrics_to_compare:
        print(f"\n{metric}:")
        for strategy, data in results.items():
            value = data['final_metrics'][metric]
            print(f"  {strategy}: {value:.2f}")
    
    return results


def adaptive_bandwidth_experiment():
    """Simulate a simple adaptive bandwidth allocation strategy (as a placeholder for RL)."""
    # Create environment
    env = NetworkEnvironment(num_aps=3, ap_capacities=[50, 75, 100], seed=42)
    env.max_steps = 300
    
    # Enable mixed traffic
    traffic_patterns = {
        'smooth': {'enabled': True, 'rate': 20},
        'periodic': {'enabled': True, 'base_rate': 15, 'period': 5, 'amplitude': 10},
        'burst': {'enabled': True, 'base_rate': 5, 'burst_prob': 0.05, 'burst_size': 20}
    }
    
    # Reset environment
    state = env.reset(traffic_patterns)
    
    # Run simulation with adaptive bandwidth allocation
    print("\nRunning adaptive bandwidth allocation experiment...")
    done = False
    step = 0
    
    # History for visualization
    adaptive_actions = []
    
    while not done:
        # Simple adaptive strategy: allocate bandwidth inversely proportional to buffer fullness
        buffer_fullness = [len(ap.buffer) / ap.buffer_size for ap in env.access_points]
        total_fullness = sum(buffer_fullness) + 0.001  # Avoid division by zero
        
        # Allocate more bandwidth to APs with fuller buffers
        normalized_fullness = [full / total_fullness for full in buffer_fullness]
        total_capacity = sum(ap.capacity for ap in env.access_points)
        
        # Calculate bandwidth allocation
        action = []
        for i, ap in enumerate(env.access_points):
            # Base allocation proportional to capacity
            base_allocation = ap.capacity / total_capacity * 0.5
            
            # Additional allocation based on buffer fullness
            adaptive_allocation = normalized_fullness[i] * 0.5
            
            # Combine and scale by capacity
            allocation = (base_allocation + adaptive_allocation) * ap.capacity
            action.append(min(allocation, ap.capacity))  # Cap at capacity
        
        adaptive_actions.append(action)
        
        # Take step with adaptive action
        state, reward, done, info = env.step(action)
        
        step += 1
        if step % 50 == 0:
            print(f"  Step {step}/{env.max_steps}")
    
    # Visualize results
    env.scheduler.visualize_performance(save_path="adaptive_performance.png")
    
    # Plot the adaptive actions
    plt.figure(figsize=(10, 6))
    adaptive_actions = np.array(adaptive_actions)
    for i in range(env.num_aps):
        plt.plot(adaptive_actions[:, i], label=f'AP {i}')
    
    plt.title('Adaptive Bandwidth Allocation')
    plt.xlabel('Step')
    plt.ylabel('Bandwidth (Mbps)')
    plt.legend()
    plt.grid(True)
    plt.savefig("adaptive_bandwidth.png")
    plt.show()
    
    # Print final metrics
    final_metrics = env.scheduler.get_metrics()
    print("\nFinal Metrics for Adaptive Allocation:")
    for key, value in final_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    return final_metrics


def compare_scheduling_methods():
    """Compare different packet scheduling methods across traffic patterns."""
    # Define the schedulers to compare
    schedulers = [
        round_robin_scheduler,
        least_loaded_scheduler,
        random_scheduler,
        pattern_aware_scheduler,
    ]
    
    # Define the traffic patterns to test
    patterns = ['smooth', 'periodic', 'burst']
    
    # Run experiments for each combination
    results = {}
    
    for pattern in patterns:
        results[pattern] = {}
        for scheduler in schedulers:
            scheduler_name = scheduler.__name__
            print(f"\nTesting {scheduler_name} with {pattern} traffic...")
            results[pattern][scheduler_name] = run_traffic_pattern_experiment(
                pattern, 
                scheduler=scheduler
            )
    
    # Create comparison visualizations for each metric
    metrics_to_compare = ['throughput', 'avg_delay', 'packet_loss_rate']
    
    for metric in metrics_to_compare:
        plt.figure(figsize=(14, 8))
        
        # Bar positions
        n_schedulers = len(schedulers)
        bar_width = 0.8 / n_schedulers
        index = np.arange(len(patterns))
        
        # Plot bars for each scheduler
        for i, scheduler in enumerate(schedulers):
            scheduler_name = scheduler.__name__
            values = [results[pattern][scheduler_name][metric] for pattern in patterns]
            offset = (i - n_schedulers / 2 + 0.5) * bar_width
            plt.bar(index + offset, values, bar_width, label=scheduler_name)
        
        plt.xlabel('Traffic Pattern')
        plt.ylabel(metric)
        plt.title(f'{metric} by Scheduler and Traffic Pattern')
        plt.xticks(index, patterns)
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"scheduler_comparison_{metric}.png")
        plt.show()
    
    # Print detailed summary
    print("\nDetailed Performance Summary:")
    for metric in metrics_to_compare:
        print(f"\n{metric.upper()}:")
        for pattern in patterns:
            print(f"  {pattern} traffic:")
            for scheduler in schedulers:
                scheduler_name = scheduler.__name__
                value = results[pattern][scheduler_name][metric]
                print(f"    {scheduler_name}: {value:.2f}")
    
    return results


if __name__ == "__main__":
    print("Network Environment Simulation")
    print("=============================")
    
    while True:
        print("\nSelect an experiment to run:")
        print("1. Basic Simulation")
        print("2. Compare Traffic Patterns (Smooth, Periodic, Burst)")
        print("3. Compare Bandwidth Allocation Strategies")
        print("4. Run Adaptive Bandwidth Allocation Experiment")
        print("5. Compare Scheduling Methods")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == "1":
            run_basic_simulation()
        elif choice == "2":
            run_comparative_experiments()
        elif choice == "3":
            custom_bandwidth_allocation_experiment()
        elif choice == "4":
            adaptive_bandwidth_experiment()
        elif choice == "5":
            compare_scheduling_methods()
        elif choice == "6":
            print("Exiting simulation.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")