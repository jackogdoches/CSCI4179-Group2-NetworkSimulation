"""
Network Environment Simulation Package
--------------------------------------

A Python-based network environment simulation for load balancing research using
reinforcement learning.

This package provides tools for simulating different network traffic patterns
and testing various bandwidth allocation strategies across multiple access points.
"""

# Import main components for easier access
from .packet import Packet
from .access_point import AccessPoint
from .traffic_generator import TrafficGenerator
from .network_scheduler import NetworkScheduler
from .network_environment import NetworkEnvironment

# Import scheduler algorithms
from .schedulers import (
    round_robin_scheduler,
    least_loaded_scheduler,
    random_scheduler,
    fifo_scheduler,
    weighted_scheduler,
    packet_size_based_scheduler,
    pattern_aware_scheduler
)

# Version information
__version__ = '0.1.0'