#!/usr/bin/env python3
"""
Main entry point for running the network environment simulation.
"""

import sys
from simulation import (
    run_basic_simulation,
    run_comparative_experiments,
    custom_bandwidth_allocation_experiment,
    adaptive_bandwidth_experiment,
    compare_scheduling_methods
)

def main():
    """Main function for running the simulation."""
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
        
        try:
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
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Continuing...")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())