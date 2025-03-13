# Network Environment Simulation

A Python-based network environment simulation for load balancing research using reinforcement learning. This package provides tools for simulating different network traffic patterns and testing various bandwidth allocation strategies across multiple access points.

## Project Structure

```
network_simulation/
├── __init__.py           # Package initialization
├── access_point.py       # Access point implementation
├── network_environment.py # Main environment class
├── network_scheduler.py  # Network scheduler class
├── packet.py             # Packet class
├── schedulers.py         # Packet scheduling algorithms
├── simulation.py         # Simulation experiments
├── traffic_generator.py  # Traffic pattern generation
└── main.py               # Main entry point
```

## Features

- **Network Scheduler**: Generates network traffic with different characteristics (smooth, periodic, and burst patterns).
- **Access Point Pool**: Simulated access points with configurable capacities.
- **Traffic Patterns**:
  - **Smooth**: Consistent and steady data flow.
  - **Periodic**: Alternating high and low traffic phases.
  - **Burst**: Sudden spikes in network demand.
- **Performance Metrics**:
  - **Throughput**: Total amount of data successfully transmitted.
  - **Delay**: Average time for packets to reach their destination.
  - **Packet Loss Rate**: Percentage of packets that fail to reach their destination.
- **Scheduling Algorithms**:
  - **Round Robin**: Rotates through APs sequentially.
  - **Least Loaded**: Selects the AP with the lowest current load.
  - **FIFO**: First-In-First-Out scheduling with global queues.
  - **Pattern-Aware**: Different strategies based on traffic pattern.
- **Visualization Tools**: Built-in functions to visualize network performance.
- **Reinforcement Learning (RL) Interface**: Designed for easy integration with RL agents.

## Installation

### Prerequisites

- Python 3.7+
- NumPy
- Matplotlib

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/network-simulation.git
   cd network-simulation
   ```

2. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```

## Usage

### Quick Start

Run the main script to launch the interactive simulation:

```bash
python main.py
```

Then select from the available experiments:
1. Basic Simulation
2. Compare Traffic Patterns
3. Compare Bandwidth Allocation Strategies
4. Run Adaptive Bandwidth Allocation Experiment
5. Compare Scheduling Methods

### Basic Usage in Your Own Code

```python
from network_environment import NetworkEnvironment
from schedulers import round_robin_scheduler

# Create environment with 3 APs of different capacities
env = NetworkEnvironment(num_aps=3, ap_capacities=[50, 75, 100])

# Configure traffic patterns
traffic_patterns = {
    'smooth': {'enabled': True, 'rate': 40},
    'periodic': {'enabled': True, 'base_rate': 20, 'period': 5, 'amplitude': 15},
    'burst': {'enabled': True, 'base_rate': 10, 'burst_prob': 0.1, 'burst_size': 30}
}

# Run a simulation with round-robin packet scheduling
final_metrics = env.run_episode(
    traffic_patterns,
    scheduler=round_robin_scheduler,
    render=True
)

# Visualize results
env.visualize_results()
```

### Creating Custom Schedulers

You can easily create custom packet schedulers:

```python
def my_custom_scheduler(packet, access_points, current_time, total_packets, **kwargs):
    """Custom packet scheduling algorithm."""
    # Your scheduling logic here
    # Example: Select the AP with the most available buffer space
    return max(access_points, key=lambda ap: ap.buffer_size - len(ap.buffer))

# Use your custom scheduler
env.run_episode(traffic_patterns, scheduler=my_custom_scheduler)
```

### Integrating with RL Agents

The environment is designed to work with reinforcement learning agents:

```python
# Create your RL agent
class SimpleRLAgent:
    def get_action(self, state):
        # Your action selection logic
        return [50, 75, 100]  # Example bandwidth allocation
        
    def update(self, state, reward, done):
        # Your learning update
        pass

# Use the agent
agent = SimpleRLAgent()
env.run_episode(traffic_patterns, agent=agent)
```

## Customization

### Modifying Traffic Patterns

```python
# Customize smooth traffic
env.scheduler.enable_traffic_pattern('smooth', rate=60)

# Customize periodic traffic
env.scheduler.enable_traffic_pattern('periodic', 
                                    base_rate=30, 
                                    period=8,  # cycle length in seconds
                                    amplitude=20)  # variation amount

# Customize burst traffic
env.scheduler.enable_traffic_pattern('burst', 
                                    base_rate=15, 
                                    burst_prob=0.08,  # probability of burst
                                    burst_size=40)  # packets per burst
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.