#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize Poinsot ellipsoids')
parser.add_argument('--omega1', type=float, help='Angular velocity omega1')
parser.add_argument('--omega2', type=float, help='Angular velocity omega2')
parser.add_argument('--omega3', type=float, help='Angular velocity omega3')
args = parser.parse_args()

# If no arguments given at all, default to omega2 = 1
if len(sys.argv) == 1:
    omega1, omega2, omega3 = 0.0, 1.0, 0.0
else:
    omega1 = args.omega1 if args.omega1 is not None else 0.0
    omega2 = args.omega2 if args.omega2 is not None else 0.0
    omega3 = args.omega3 if args.omega3 is not None else 0.0

# Moments of inertia (assume I1 > I2 > I3)
I1, I2, I3 = 5.0, 3.0, 2.0

# Calculate conserved quantities
L_squared = I1**2 * omega1**2 + I2**2 * omega2**2 + I3**2 * omega3**2
T = 0.5 * (I1 * omega1**2 + I2 * omega2**2 + I3 * omega3**2)

# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate mesh for one octant (positive octant)
u = np.linspace(0, np.pi/2, 50)
v = np.linspace(0, np.pi/2, 50)
U, V = np.meshgrid(u, v)

# Inertia ellipsoid: omega1^2/a1^2 + omega2^2/a2^2 + omega3^2/a3^2 = 1
# where a_i = sqrt(L^2/I_i^2)
a1 = np.sqrt(L_squared) / I1
a2 = np.sqrt(L_squared) / I2
a3 = np.sqrt(L_squared) / I3

X_inertia = a1 * np.sin(U) * np.cos(V)
Y_inertia = a2 * np.sin(U) * np.sin(V)
Z_inertia = a3 * np.cos(U)

# Energy ellipsoid: I1*omega1^2 + I2*omega2^2 + I3*omega3^2 = 2T
# or omega1^2/b1^2 + omega2^2/b2^2 + omega3^2/b3^2 = 1
# where b_i = sqrt(2T/I_i)
b1 = np.sqrt(2 * T / I1)
b2 = np.sqrt(2 * T / I2)
b3 = np.sqrt(2 * T / I3)

X_energy = b1 * np.sin(U) * np.cos(V)
Y_energy = b2 * np.sin(U) * np.sin(V)
Z_energy = b3 * np.cos(U)

# Plot the inertia ellipsoid (blue)
ax.plot_surface(X_inertia, Y_inertia, Z_inertia, alpha=0.3, color='blue', 
                label='Inertia Ellipsoid')

# Plot the energy ellipsoid (red)
ax.plot_surface(X_energy, Y_energy, Z_energy, alpha=0.3, color='red',
                label='Energy Ellipsoid')

# Calculate intersection curve
# Both ellipsoids pass through the point (omega1, omega2, omega3)
# Intersection is where both constraints are satisfied simultaneously
theta = np.linspace(0, 2*np.pi, 200)

# Parametric curve for intersection (using Lagrange multiplier solution)
# The intersection lies on the polhode
# For visualization, sample points on a circular path scaled appropriately
omega_mag = np.sqrt(omega1**2 + omega2**2 + omega3**2)
scale = 0.8

# Generate intersection curve by finding points satisfying both constraints
intersection_points = []
for t in theta:
    # Use spherical coordinates and scale to find approximate intersection
    for phi in np.linspace(0, np.pi/2, 20):
        w1 = omega_mag * np.sin(phi) * np.cos(t) * scale
        w2 = omega_mag * np.sin(phi) * np.sin(t) * scale  
        w3 = omega_mag * np.cos(phi) * scale
        
        # Check if point is close to both surfaces
        inertia_val = (w1*I1)**2 + (w2*I2)**2 + (w3*I3)**2
        energy_val = I1*w1**2 + I2*w2**2 + I3*w3**2
        
        if abs(inertia_val - L_squared) < 0.1 * L_squared and \
           abs(energy_val - 2*T) < 0.1 * 2*T and \
           w1 >= 0 and w2 >= 0 and w3 >= 0:
            intersection_points.append([w1, w2, w3])

if intersection_points:
    intersection_points = np.array(intersection_points)
    ax.plot(intersection_points[:, 0], intersection_points[:, 1], 
            intersection_points[:, 2], 'g--', linewidth=2, label='Intersection')

# Mark the given angular velocity point
ax.scatter([omega1], [omega2], [omega3], color='black', s=100, marker='o',
           label=f'ω = ({omega1}, {omega2}, {omega3})')

# Draw and label axes
axis_length = max(a1, a2, a3) * 1.2
ax.plot([0, axis_length], [0, 0], [0, 0], 'k-', linewidth=1)
ax.plot([0, 0], [0, axis_length], [0, 0], 'k-', linewidth=1)
ax.plot([0, 0], [0, 0], [0, axis_length], 'k-', linewidth=1)

ax.text(axis_length, 0, 0, r'$\omega_1$', fontsize=14)
ax.text(0, axis_length, 0, r'$\omega_2$', fontsize=14)
ax.text(0, 0, axis_length, r'$\omega_3$', fontsize=14)

# Set labels and title
ax.set_xlabel(r'$\omega_1$', fontsize=12)
ax.set_ylabel(r'$\omega_2$', fontsize=12)
ax.set_zlabel(r'$\omega_3$', fontsize=12)
ax.set_title('Poinsot Ellipsoids\n(Blue: Inertia, Red: Energy)', fontsize=14)

# Set equal aspect ratio
max_range = max(a1, a2, a3)
ax.set_xlim([0, max_range])
ax.set_ylim([0, max_range])
ax.set_zlim([0, max_range])

ax.legend(loc='upper left')
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('poinsot_ellipsoids.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Moments of inertia: I1={I1}, I2={I2}, I3={I3}")
print(f"Angular velocity: ω=({omega1}, {omega2}, {omega3})")
print(f"Angular momentum squared: L²={L_squared:.3f}")
print(f"Rotational kinetic energy: T={T:.3f}")
