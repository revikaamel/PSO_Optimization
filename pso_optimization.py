import numpy as np
import matplotlib.pyplot as plt

# Define the objective function f(x) = x^2 + 2x + 1
def objective_function(x):
    return x**2 + 2*x + 1

# PSO parameters
num_particles = 10
max_iterations = 50
w = 0.5  # Inertia weight
c1 = 1.5  # Cognitive coefficient
c2 = 1.5  # Social coefficient
bounds = [-10, 10]  # Search bounds

# Initialize particles
np.random.seed(42)  # For reproducibility
positions = np.random.uniform(bounds[0], bounds[1], num_particles)
velocities = np.random.uniform(-1, 1, num_particles)
pbest_positions = positions.copy()
pbest_values = np.array([objective_function(x) for x in positions])
gbest_idx = np.argmin(pbest_values)
gbest_position = pbest_positions[gbest_idx]
gbest_value = pbest_values[gbest_idx]

# Store best fitness per iteration for plotting
best_fitness_history = []

# PSO main loop
for iteration in range(max_iterations):
    for i in range(num_particles):
        # Update velocity
        r1, r2 = np.random.rand(), np.random.rand()
        velocities[i] = (w * velocities[i] +
                         c1 * r1 * (pbest_positions[i] - positions[i]) +
                         c2 * r2 * (gbest_position - positions[i]))
        
        # Update position
        positions[i] += velocities[i]
        
        # Constrain position to bounds
        positions[i] = np.clip(positions[i], bounds[0], bounds[1])
        
        # Evaluate fitness
        fitness = objective_function(positions[i])
        
        # Update personal best
        if fitness < pbest_values[i]:
            pbest_positions[i] = positions[i]
            pbest_values[i] = fitness
    
    # Update global best
    gbest_idx = np.argmin(pbest_values)
    if pbest_values[gbest_idx] < gbest_value:
        gbest_position = pbest_positions[gbest_idx]
        gbest_value = pbest_values[gbest_idx]
    
    # Store best fitness for this iteration
    best_fitness_history.append(gbest_value)

# Print results
print(f"Minimum value found: {gbest_value}")
print(f"Best position (x): {gbest_position}")

# Plot best fitness per iteration
plt.figure(figsize=(8, 6))
plt.plot(range(max_iterations), best_fitness_history, 'b-', label='Best Fitness')
plt.title('Best Fitness Value per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Fitness Value')
plt.grid(True)
plt.legend()
plt.savefig('pso_fitness_plot.png')
