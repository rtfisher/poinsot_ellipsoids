#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys


def parse_arguments(args=None):
    """Parse command line arguments for angular velocity components.

    Args:
        args: List of command line arguments (defaults to sys.argv if None)

    Returns:
        tuple: (omega1, omega2, omega3) angular velocity components
    """
    parser = argparse.ArgumentParser(description='Visualize Poinsot ellipsoids')
    parser.add_argument('--omega1', type=float, help='Angular velocity omega1')
    parser.add_argument('--omega2', type=float, help='Angular velocity omega2')
    parser.add_argument('--omega3', type=float, help='Angular velocity omega3')

    parsed_args = parser.parse_args(args)

    # If no arguments given at all, default to omega2 = 1
    if args is None and len(sys.argv) == 1:
        return 0.0, 1.0, 0.0
    elif args is not None and len(args) == 0:
        return 0.0, 1.0, 0.0
    else:
        omega1 = parsed_args.omega1 if parsed_args.omega1 is not None else 0.0
        omega2 = parsed_args.omega2 if parsed_args.omega2 is not None else 0.0
        omega3 = parsed_args.omega3 if parsed_args.omega3 is not None else 0.0
        return omega1, omega2, omega3


def calculate_conserved_quantities(omega1, omega2, omega3, I1, I2, I3):
    """Calculate angular momentum squared and kinetic energy.

    Args:
        omega1, omega2, omega3: Angular velocity components
        I1, I2, I3: Moments of inertia

    Returns:
        tuple: (L_squared, T) - angular momentum squared and kinetic energy
    """
    L_squared = I1**2 * omega1**2 + I2**2 * omega2**2 + I3**2 * omega3**2
    T = 0.5 * (I1 * omega1**2 + I2 * omega2**2 + I3 * omega3**2)
    return L_squared, T


def calculate_ellipsoid_parameters(L_squared, T, I1, I2, I3):
    """Calculate semi-axes for inertia and energy ellipsoids.

    Args:
        L_squared: Angular momentum squared
        T: Kinetic energy
        I1, I2, I3: Moments of inertia

    Returns:
        tuple: (a1, a2, a3, b1, b2, b3) where a_i are inertia ellipsoid semi-axes
               and b_i are energy ellipsoid semi-axes
    """
    # Inertia ellipsoid semi-axes
    a1 = np.sqrt(L_squared) / I1 if L_squared > 0 else 0.0
    a2 = np.sqrt(L_squared) / I2 if L_squared > 0 else 0.0
    a3 = np.sqrt(L_squared) / I3 if L_squared > 0 else 0.0

    # Energy ellipsoid semi-axes
    b1 = np.sqrt(2 * T / I1) if T > 0 else 0.0
    b2 = np.sqrt(2 * T / I2) if T > 0 else 0.0
    b3 = np.sqrt(2 * T / I3) if T > 0 else 0.0

    return a1, a2, a3, b1, b2, b3


def generate_ellipsoid_surface(a1, a2, a3, n_points=50):
    """Generate mesh points for an ellipsoid surface in the positive octant.

    Args:
        a1, a2, a3: Semi-axes of the ellipsoid
        n_points: Number of points along each dimension

    Returns:
        tuple: (X, Y, Z) arrays of surface coordinates
    """
    u = np.linspace(0, np.pi/2, n_points)
    v = np.linspace(0, np.pi/2, n_points)
    U, V = np.meshgrid(u, v)

    X = a1 * np.sin(U) * np.cos(V)
    Y = a2 * np.sin(U) * np.sin(V)
    Z = a3 * np.cos(U)

    return X, Y, Z


def find_intersection_points(omega1, omega2, omega3, L_squared, T, I1, I2, I3,
                            n_theta=200, n_phi=20, tolerance=0.1):
    """Find approximate intersection points between inertia and energy ellipsoids.

    Args:
        omega1, omega2, omega3: Angular velocity components
        L_squared: Angular momentum squared
        T: Kinetic energy
        I1, I2, I3: Moments of inertia
        n_theta: Number of points in theta direction
        n_phi: Number of points in phi direction
        tolerance: Relative tolerance for intersection

    Returns:
        numpy.ndarray: Array of intersection points with shape (n, 3)
    """
    theta = np.linspace(0, 2*np.pi, n_theta)
    omega_mag = np.sqrt(omega1**2 + omega2**2 + omega3**2)
    scale = 0.8

    intersection_points = []
    for t in theta:
        for phi in np.linspace(0, np.pi/2, n_phi):
            w1 = omega_mag * np.sin(phi) * np.cos(t) * scale
            w2 = omega_mag * np.sin(phi) * np.sin(t) * scale
            w3 = omega_mag * np.cos(phi) * scale

            # Check if point is close to both surfaces
            inertia_val = (w1*I1)**2 + (w2*I2)**2 + (w3*I3)**2
            energy_val = I1*w1**2 + I2*w2**2 + I3*w3**2

            if abs(inertia_val - L_squared) < tolerance * L_squared and \
               abs(energy_val - 2*T) < tolerance * 2*T and \
               w1 >= 0 and w2 >= 0 and w3 >= 0:
                intersection_points.append([w1, w2, w3])

    if intersection_points:
        return np.array(intersection_points)
    else:
        return np.array([])


def create_visualization(omega1, omega2, omega3, I1, I2, I3,
                        output_file='poinsot_ellipsoids.png', show_plot=True):
    """Create and save Poinsot ellipsoid visualization.

    Args:
        omega1, omega2, omega3: Angular velocity components
        I1, I2, I3: Moments of inertia
        output_file: Path to save the figure
        show_plot: Whether to display the plot interactively

    Returns:
        tuple: (L_squared, T) - conserved quantities
    """
    # Calculate conserved quantities
    L_squared, T = calculate_conserved_quantities(omega1, omega2, omega3, I1, I2, I3)

    # Calculate ellipsoid parameters
    a1, a2, a3, b1, b2, b3 = calculate_ellipsoid_parameters(L_squared, T, I1, I2, I3)

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generate ellipsoid surfaces
    X_inertia, Y_inertia, Z_inertia = generate_ellipsoid_surface(a1, a2, a3)
    X_energy, Y_energy, Z_energy = generate_ellipsoid_surface(b1, b2, b3)

    # Plot the inertia ellipsoid (blue)
    ax.plot_surface(X_inertia, Y_inertia, Z_inertia, alpha=0.3, color='blue',
                    label='Inertia Ellipsoid')

    # Plot the energy ellipsoid (red)
    ax.plot_surface(X_energy, Y_energy, Z_energy, alpha=0.3, color='red',
                    label='Energy Ellipsoid')

    # Calculate and plot intersection curve
    intersection_points = find_intersection_points(omega1, omega2, omega3,
                                                   L_squared, T, I1, I2, I3)

    if len(intersection_points) > 0:
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
    plt.savefig(output_file, dpi=150, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()

    return L_squared, T


def main():
    """Main function to run the Poinsot ellipsoid visualization."""
    # Parse command line arguments
    omega1, omega2, omega3 = parse_arguments()

    # Moments of inertia (assume I1 > I2 > I3)
    I1, I2, I3 = 5.0, 3.0, 2.0

    # Create visualization
    L_squared, T = create_visualization(omega1, omega2, omega3, I1, I2, I3)

    # Print results
    print(f"Moments of inertia: I1={I1}, I2={I2}, I3={I3}")
    print(f"Angular velocity: ω=({omega1}, {omega2}, {omega3})")
    print(f"Angular momentum squared: L²={L_squared:.3f}")
    print(f"Rotational kinetic energy: T={T:.3f}")


if __name__ == '__main__':
    main()
