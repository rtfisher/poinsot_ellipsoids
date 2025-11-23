#!/usr/bin/env python3
"""
Comprehensive test suite for poinsot_ellipsoid.py

Tests cover:
- Argument parsing
- Physics calculations (conserved quantities)
- Ellipsoid parameter calculations
- Surface generation
- Intersection finding
- Visualization creation
- Edge cases and error handling
"""

import pytest
import numpy as np
import os
import tempfile
from poinsot_ellipsoid import (
    parse_arguments,
    calculate_conserved_quantities,
    calculate_ellipsoid_parameters,
    generate_ellipsoid_surface,
    find_intersection_points,
    create_visualization
)


class TestArgumentParsing:
    """Test command-line argument parsing."""

    def test_no_arguments_defaults_to_omega2_equals_one(self):
        """Test that no arguments defaults to ω = (0, 1, 0)."""
        omega1, omega2, omega3 = parse_arguments([])
        assert omega1 == 0.0
        assert omega2 == 1.0
        assert omega3 == 0.0

    def test_single_omega1_argument(self):
        """Test parsing single omega1 argument."""
        omega1, omega2, omega3 = parse_arguments(['--omega1', '0.5'])
        assert omega1 == 0.5
        assert omega2 == 0.0
        assert omega3 == 0.0

    def test_single_omega2_argument(self):
        """Test parsing single omega2 argument."""
        omega1, omega2, omega3 = parse_arguments(['--omega2', '0.8'])
        assert omega1 == 0.0
        assert omega2 == 0.8
        assert omega3 == 0.0

    def test_single_omega3_argument(self):
        """Test parsing single omega3 argument."""
        omega1, omega2, omega3 = parse_arguments(['--omega3', '0.3'])
        assert omega1 == 0.0
        assert omega2 == 0.0
        assert omega3 == 0.3

    def test_all_three_components(self):
        """Test parsing all three angular velocity components."""
        omega1, omega2, omega3 = parse_arguments(['--omega1', '0.2', '--omega2', '0.8', '--omega3', '0.1'])
        assert omega1 == 0.2
        assert omega2 == 0.8
        assert omega3 == 0.1

    def test_negative_values(self):
        """Test that negative angular velocities are accepted."""
        omega1, omega2, omega3 = parse_arguments(['--omega1', '-0.5', '--omega2', '-0.3'])
        assert omega1 == -0.5
        assert omega2 == -0.3
        assert omega3 == 0.0

    def test_zero_values(self):
        """Test explicit zero values."""
        omega1, omega2, omega3 = parse_arguments(['--omega1', '0', '--omega2', '0', '--omega3', '0'])
        assert omega1 == 0.0
        assert omega2 == 0.0
        assert omega3 == 0.0

    def test_floating_point_values(self):
        """Test various floating point formats."""
        omega1, omega2, omega3 = parse_arguments(['--omega1', '1.234', '--omega2', '0.001', '--omega3', '100.0'])
        assert pytest.approx(omega1) == 1.234
        assert pytest.approx(omega2) == 0.001
        assert pytest.approx(omega3) == 100.0


class TestConservedQuantities:
    """Test physics calculations for conserved quantities."""

    def test_zero_angular_velocity(self):
        """Test that zero angular velocity gives zero conserved quantities."""
        I1, I2, I3 = 5.0, 3.0, 2.0
        L_squared, T = calculate_conserved_quantities(0.0, 0.0, 0.0, I1, I2, I3)
        assert L_squared == 0.0
        assert T == 0.0

    def test_rotation_about_axis_1(self):
        """Test rotation purely about axis 1."""
        I1, I2, I3 = 5.0, 3.0, 2.0
        omega1 = 1.0
        L_squared, T = calculate_conserved_quantities(omega1, 0.0, 0.0, I1, I2, I3)
        assert pytest.approx(L_squared) == I1**2 * omega1**2
        assert pytest.approx(T) == 0.5 * I1 * omega1**2

    def test_rotation_about_axis_2(self):
        """Test rotation purely about axis 2."""
        I1, I2, I3 = 5.0, 3.0, 2.0
        omega2 = 1.0
        L_squared, T = calculate_conserved_quantities(0.0, omega2, 0.0, I1, I2, I3)
        assert pytest.approx(L_squared) == I2**2 * omega2**2
        assert pytest.approx(T) == 0.5 * I2 * omega2**2

    def test_rotation_about_axis_3(self):
        """Test rotation purely about axis 3."""
        I1, I2, I3 = 5.0, 3.0, 2.0
        omega3 = 1.0
        L_squared, T = calculate_conserved_quantities(0.0, 0.0, omega3, I1, I2, I3)
        assert pytest.approx(L_squared) == I3**2 * omega3**2
        assert pytest.approx(T) == 0.5 * I3 * omega3**2

    def test_general_rotation(self):
        """Test general rotation with all components."""
        I1, I2, I3 = 5.0, 3.0, 2.0
        omega1, omega2, omega3 = 0.2, 0.8, 0.1
        L_squared, T = calculate_conserved_quantities(omega1, omega2, omega3, I1, I2, I3)

        expected_L_squared = (I1*omega1)**2 + (I2*omega2)**2 + (I3*omega3)**2
        expected_T = 0.5 * (I1*omega1**2 + I2*omega2**2 + I3*omega3**2)

        assert pytest.approx(L_squared) == expected_L_squared
        assert pytest.approx(T) == expected_T

    def test_conservation_under_scaling(self):
        """Test that doubling angular velocity quadruples both L² and T."""
        I1, I2, I3 = 5.0, 3.0, 2.0
        omega1, omega2, omega3 = 0.2, 0.8, 0.1

        L1, T1 = calculate_conserved_quantities(omega1, omega2, omega3, I1, I2, I3)
        L2, T2 = calculate_conserved_quantities(2*omega1, 2*omega2, 2*omega3, I1, I2, I3)

        assert pytest.approx(L2) == 4 * L1
        assert pytest.approx(T2) == 4 * T1

    def test_different_inertia_values(self):
        """Test with different moment of inertia values."""
        I1, I2, I3 = 10.0, 5.0, 1.0
        omega1, omega2, omega3 = 0.1, 0.2, 0.3
        L_squared, T = calculate_conserved_quantities(omega1, omega2, omega3, I1, I2, I3)

        assert L_squared > 0
        assert T > 0
        assert pytest.approx(L_squared) == (10*0.1)**2 + (5*0.2)**2 + (1*0.3)**2
        assert pytest.approx(T) == 0.5 * (10*0.1**2 + 5*0.2**2 + 1*0.3**2)


class TestEllipsoidParameters:
    """Test ellipsoid semi-axes calculations."""

    def test_zero_angular_velocity_gives_zero_parameters(self):
        """Test that zero angular velocity gives zero ellipsoid parameters."""
        I1, I2, I3 = 5.0, 3.0, 2.0
        L_squared, T = 0.0, 0.0
        a1, a2, a3, b1, b2, b3 = calculate_ellipsoid_parameters(L_squared, T, I1, I2, I3)

        assert a1 == 0.0
        assert a2 == 0.0
        assert a3 == 0.0
        assert b1 == 0.0
        assert b2 == 0.0
        assert b3 == 0.0

    def test_inertia_ellipsoid_axes_ordering(self):
        """Test that inertia ellipsoid semi-axes have correct ordering (a1 < a2 < a3 for I1 > I2 > I3)."""
        I1, I2, I3 = 5.0, 3.0, 2.0
        L_squared = 10.0
        T = 1.0
        a1, a2, a3, b1, b2, b3 = calculate_ellipsoid_parameters(L_squared, T, I1, I2, I3)

        # For I1 > I2 > I3, we have a1 < a2 < a3 (inverse relationship)
        assert a1 < a2 < a3

    def test_energy_ellipsoid_axes_ordering(self):
        """Test that energy ellipsoid semi-axes have correct ordering (b1 < b2 < b3 for I1 > I2 > I3)."""
        I1, I2, I3 = 5.0, 3.0, 2.0
        L_squared = 10.0
        T = 1.0
        a1, a2, a3, b1, b2, b3 = calculate_ellipsoid_parameters(L_squared, T, I1, I2, I3)

        # For I1 > I2 > I3, we have b1 < b2 < b3 (inverse relationship)
        assert b1 < b2 < b3

    def test_specific_values(self):
        """Test ellipsoid parameters with specific known values."""
        I1, I2, I3 = 5.0, 3.0, 2.0
        omega1, omega2, omega3 = 0.0, 1.0, 0.0
        L_squared, T = calculate_conserved_quantities(omega1, omega2, omega3, I1, I2, I3)
        a1, a2, a3, b1, b2, b3 = calculate_ellipsoid_parameters(L_squared, T, I1, I2, I3)

        # Inertia ellipsoid semi-axes
        expected_a1 = np.sqrt(L_squared) / I1
        expected_a2 = np.sqrt(L_squared) / I2
        expected_a3 = np.sqrt(L_squared) / I3

        assert pytest.approx(a1) == expected_a1
        assert pytest.approx(a2) == expected_a2
        assert pytest.approx(a3) == expected_a3

        # Energy ellipsoid semi-axes
        expected_b1 = np.sqrt(2 * T / I1)
        expected_b2 = np.sqrt(2 * T / I2)
        expected_b3 = np.sqrt(2 * T / I3)

        assert pytest.approx(b1) == expected_b1
        assert pytest.approx(b2) == expected_b2
        assert pytest.approx(b3) == expected_b3

    def test_all_positive_parameters(self):
        """Test that all parameters are positive for non-zero angular velocity."""
        I1, I2, I3 = 5.0, 3.0, 2.0
        L_squared, T = 10.0, 2.0
        a1, a2, a3, b1, b2, b3 = calculate_ellipsoid_parameters(L_squared, T, I1, I2, I3)

        assert a1 > 0
        assert a2 > 0
        assert a3 > 0
        assert b1 > 0
        assert b2 > 0
        assert b3 > 0


class TestEllipsoidSurface:
    """Test ellipsoid surface generation."""

    def test_surface_shape(self):
        """Test that generated surface has correct shape."""
        a1, a2, a3 = 1.0, 2.0, 3.0
        n_points = 10
        X, Y, Z = generate_ellipsoid_surface(a1, a2, a3, n_points)

        assert X.shape == (n_points, n_points)
        assert Y.shape == (n_points, n_points)
        assert Z.shape == (n_points, n_points)

    def test_surface_at_poles(self):
        """Test surface coordinates at poles."""
        a1, a2, a3 = 1.0, 2.0, 3.0
        n_points = 50
        X, Y, Z = generate_ellipsoid_surface(a1, a2, a3, n_points)

        # At u=0 (north pole), should have x=0, y=0, z=a3
        assert pytest.approx(X[0, 0], abs=1e-10) == 0.0
        assert pytest.approx(Y[0, 0], abs=1e-10) == 0.0
        assert pytest.approx(Z[0, 0]) == a3

    def test_surface_at_equator(self):
        """Test surface coordinates at equator."""
        a1, a2, a3 = 1.0, 2.0, 3.0
        n_points = 50
        X, Y, Z = generate_ellipsoid_surface(a1, a2, a3, n_points)

        # At u=0, v=π/2 (on Y-axis), should have x≈0, y=a2, z≈0
        # At mesh point [0, -1]
        assert pytest.approx(X[0, -1]) == a1
        assert pytest.approx(Y[0, -1], abs=1e-10) == 0.0
        assert pytest.approx(Z[0, -1], rel=0.1) == 0.0

    def test_surface_all_positive(self):
        """Test that all surface points are in positive octant."""
        a1, a2, a3 = 1.0, 2.0, 3.0
        X, Y, Z = generate_ellipsoid_surface(a1, a2, a3)

        assert np.all(X >= 0)
        assert np.all(Y >= 0)
        assert np.all(Z >= 0)

    def test_surface_satisfies_ellipsoid_equation(self):
        """Test that generated points satisfy ellipsoid equation."""
        a1, a2, a3 = 1.0, 2.0, 3.0
        X, Y, Z = generate_ellipsoid_surface(a1, a2, a3)

        # All points should satisfy (X/a1)² + (Y/a2)² + (Z/a3)² = 1
        ellipsoid_values = (X/a1)**2 + (Y/a2)**2 + (Z/a3)**2
        assert np.allclose(ellipsoid_values, 1.0, rtol=1e-10)

    def test_zero_semi_axes(self):
        """Test surface generation with zero semi-axes."""
        X, Y, Z = generate_ellipsoid_surface(0.0, 0.0, 0.0)
        assert np.all(X == 0.0)
        assert np.all(Y == 0.0)
        assert np.all(Z == 0.0)


class TestIntersectionPoints:
    """Test intersection curve finding."""

    def test_zero_angular_velocity_no_intersection(self):
        """Test that zero angular velocity gives no intersection points."""
        I1, I2, I3 = 5.0, 3.0, 2.0
        omega1, omega2, omega3 = 0.0, 0.0, 0.0
        L_squared, T = calculate_conserved_quantities(omega1, omega2, omega3, I1, I2, I3)
        points = find_intersection_points(omega1, omega2, omega3, L_squared, T, I1, I2, I3)

        assert len(points) == 0

    def test_intersection_returns_array(self):
        """Test that intersection finder returns numpy array."""
        I1, I2, I3 = 5.0, 3.0, 2.0
        omega1, omega2, omega3 = 0.0, 1.0, 0.0
        L_squared, T = calculate_conserved_quantities(omega1, omega2, omega3, I1, I2, I3)
        points = find_intersection_points(omega1, omega2, omega3, L_squared, T, I1, I2, I3)

        assert isinstance(points, np.ndarray)

    def test_intersection_points_in_positive_octant(self):
        """Test that all intersection points are in positive octant."""
        I1, I2, I3 = 5.0, 3.0, 2.0
        omega1, omega2, omega3 = 0.2, 0.8, 0.1
        L_squared, T = calculate_conserved_quantities(omega1, omega2, omega3, I1, I2, I3)
        points = find_intersection_points(omega1, omega2, omega3, L_squared, T, I1, I2, I3)

        if len(points) > 0:
            assert np.all(points[:, 0] >= 0)
            assert np.all(points[:, 1] >= 0)
            assert np.all(points[:, 2] >= 0)

    def test_intersection_points_shape(self):
        """Test that intersection points have correct shape."""
        I1, I2, I3 = 5.0, 3.0, 2.0
        omega1, omega2, omega3 = 0.2, 0.8, 0.1
        L_squared, T = calculate_conserved_quantities(omega1, omega2, omega3, I1, I2, I3)
        points = find_intersection_points(omega1, omega2, omega3, L_squared, T, I1, I2, I3)

        if len(points) > 0:
            assert points.shape[1] == 3  # Each point has 3 coordinates

    def test_intersection_with_different_tolerance(self):
        """Test intersection finding with different tolerance values."""
        I1, I2, I3 = 5.0, 3.0, 2.0
        omega1, omega2, omega3 = 0.2, 0.8, 0.1
        L_squared, T = calculate_conserved_quantities(omega1, omega2, omega3, I1, I2, I3)

        points_strict = find_intersection_points(omega1, omega2, omega3, L_squared, T,
                                                I1, I2, I3, tolerance=0.01)
        points_loose = find_intersection_points(omega1, omega2, omega3, L_squared, T,
                                               I1, I2, I3, tolerance=0.5)

        # Looser tolerance should give at least as many points
        assert len(points_loose) >= len(points_strict)


class TestVisualization:
    """Test visualization creation."""

    def test_visualization_creates_output_file(self):
        """Test that visualization creates the output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_output.png')
            I1, I2, I3 = 5.0, 3.0, 2.0
            omega1, omega2, omega3 = 0.0, 1.0, 0.0

            create_visualization(omega1, omega2, omega3, I1, I2, I3,
                               output_file=output_file, show_plot=False)

            assert os.path.exists(output_file)
            assert os.path.getsize(output_file) > 0

    def test_visualization_returns_conserved_quantities(self):
        """Test that visualization returns correct conserved quantities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_output.png')
            I1, I2, I3 = 5.0, 3.0, 2.0
            omega1, omega2, omega3 = 0.2, 0.8, 0.1

            L_squared, T = create_visualization(omega1, omega2, omega3, I1, I2, I3,
                                               output_file=output_file, show_plot=False)

            expected_L, expected_T = calculate_conserved_quantities(omega1, omega2, omega3, I1, I2, I3)
            assert pytest.approx(L_squared) == expected_L
            assert pytest.approx(T) == expected_T

    def test_visualization_with_zero_angular_velocity(self):
        """Test visualization with zero angular velocity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_zero.png')
            I1, I2, I3 = 5.0, 3.0, 2.0
            omega1, omega2, omega3 = 0.0, 0.0, 0.0

            L_squared, T = create_visualization(omega1, omega2, omega3, I1, I2, I3,
                                               output_file=output_file, show_plot=False)

            assert L_squared == 0.0
            assert T == 0.0
            assert os.path.exists(output_file)

    def test_visualization_with_all_components(self):
        """Test visualization with all angular velocity components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_all.png')
            I1, I2, I3 = 5.0, 3.0, 2.0
            omega1, omega2, omega3 = 0.5, 0.8, 0.3

            L_squared, T = create_visualization(omega1, omega2, omega3, I1, I2, I3,
                                               output_file=output_file, show_plot=False)

            assert L_squared > 0
            assert T > 0
            assert os.path.exists(output_file)

    def test_visualization_with_different_inertias(self):
        """Test visualization with different moment of inertia values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_inertias.png')
            I1, I2, I3 = 10.0, 5.0, 1.0
            omega1, omega2, omega3 = 0.1, 0.2, 0.3

            L_squared, T = create_visualization(omega1, omega2, omega3, I1, I2, I3,
                                               output_file=output_file, show_plot=False)

            assert L_squared > 0
            assert T > 0
            assert os.path.exists(output_file)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_angular_velocity(self):
        """Test with very small but non-zero angular velocity."""
        I1, I2, I3 = 5.0, 3.0, 2.0
        omega1, omega2, omega3 = 1e-10, 1e-10, 1e-10
        L_squared, T = calculate_conserved_quantities(omega1, omega2, omega3, I1, I2, I3)

        assert L_squared > 0
        assert T > 0
        assert L_squared < 1e-15
        assert T < 1e-15

    def test_very_large_angular_velocity(self):
        """Test with very large angular velocity."""
        I1, I2, I3 = 5.0, 3.0, 2.0
        omega1, omega2, omega3 = 1e6, 1e6, 1e6
        L_squared, T = calculate_conserved_quantities(omega1, omega2, omega3, I1, I2, I3)

        assert L_squared > 0
        assert T > 0
        assert np.isfinite(L_squared)
        assert np.isfinite(T)

    def test_equal_moments_of_inertia(self):
        """Test with equal moments of inertia (spherical top)."""
        I1 = I2 = I3 = 3.0
        omega1, omega2, omega3 = 0.5, 0.5, 0.5
        L_squared, T = calculate_conserved_quantities(omega1, omega2, omega3, I1, I2, I3)
        a1, a2, a3, b1, b2, b3 = calculate_ellipsoid_parameters(L_squared, T, I1, I2, I3)

        # For spherical top, all semi-axes should be equal
        assert pytest.approx(a1) == a2
        assert pytest.approx(a2) == a3
        assert pytest.approx(b1) == b2
        assert pytest.approx(b2) == b3

    def test_two_equal_moments_of_inertia(self):
        """Test with two equal moments of inertia (symmetric top)."""
        I1 = I2 = 5.0
        I3 = 2.0
        omega1, omega2, omega3 = 0.5, 0.5, 0.3
        L_squared, T = calculate_conserved_quantities(omega1, omega2, omega3, I1, I2, I3)
        a1, a2, a3, b1, b2, b3 = calculate_ellipsoid_parameters(L_squared, T, I1, I2, I3)

        # For symmetric top, a1 should equal a2 (and b1 should equal b2)
        assert pytest.approx(a1) == a2
        assert pytest.approx(b1) == b2
        assert a3 != a1  # But a3 should be different

    def test_single_axis_rotation_variants(self):
        """Test rotation about each principal axis."""
        I1, I2, I3 = 5.0, 3.0, 2.0

        # Test each axis
        for axis in range(3):
            omega = [0.0, 0.0, 0.0]
            omega[axis] = 1.0
            L_squared, T = calculate_conserved_quantities(omega[0], omega[1], omega[2], I1, I2, I3)
            assert L_squared > 0
            assert T > 0


class TestPhysicsConsistency:
    """Test physics consistency and relationships."""

    def test_angular_momentum_and_energy_relationship(self):
        """Test the relationship between L² and T for different angular velocities."""
        I1, I2, I3 = 5.0, 3.0, 2.0

        # For rotation about principal axis 1 with omega1 = 1.0:
        # L² = I1² * omega1² = I1²
        # T = 0.5 * I1 * omega1² = 0.5 * I1
        # So L²/T = I1² / (0.5 * I1) = 2 * I1
        omega1 = 1.0
        L1, T1 = calculate_conserved_quantities(omega1, 0.0, 0.0, I1, I2, I3)
        ratio1 = L1 / T1 if T1 > 0 else 0
        expected_ratio1 = 2 * I1
        assert pytest.approx(ratio1) == expected_ratio1

    def test_ellipsoid_intersection_consistency(self):
        """Test that angular velocity point lies on both ellipsoids."""
        I1, I2, I3 = 5.0, 3.0, 2.0
        omega1, omega2, omega3 = 0.2, 0.8, 0.1

        L_squared, T = calculate_conserved_quantities(omega1, omega2, omega3, I1, I2, I3)
        a1, a2, a3, b1, b2, b3 = calculate_ellipsoid_parameters(L_squared, T, I1, I2, I3)

        # Check that (omega1, omega2, omega3) lies on inertia ellipsoid
        inertia_check = (omega1/a1)**2 + (omega2/a2)**2 + (omega3/a3)**2
        assert pytest.approx(inertia_check, rel=1e-10) == 1.0

        # Check that (omega1, omega2, omega3) lies on energy ellipsoid
        energy_check = (omega1/b1)**2 + (omega2/b2)**2 + (omega3/b3)**2
        assert pytest.approx(energy_check, rel=1e-10) == 1.0

    def test_energy_always_positive(self):
        """Test that kinetic energy is always non-negative."""
        I1, I2, I3 = 5.0, 3.0, 2.0

        test_cases = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 0.0),
            (0.5, -0.5, 0.5)
        ]

        for omega1, omega2, omega3 in test_cases:
            L_squared, T = calculate_conserved_quantities(omega1, omega2, omega3, I1, I2, I3)
            assert T >= 0, f"Energy is negative for ω=({omega1}, {omega2}, {omega3})"

    def test_angular_momentum_squared_always_positive(self):
        """Test that L² is always non-negative."""
        I1, I2, I3 = 5.0, 3.0, 2.0

        test_cases = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 0.0),
            (0.5, -0.5, 0.5)
        ]

        for omega1, omega2, omega3 in test_cases:
            L_squared, T = calculate_conserved_quantities(omega1, omega2, omega3, I1, I2, I3)
            assert L_squared >= 0, f"L² is negative for ω=({omega1}, {omega2}, {omega3})"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
