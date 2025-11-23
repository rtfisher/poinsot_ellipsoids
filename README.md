# Poinsot Ellipsoid Visualization

A Python visualization tool for Poinsot ellipsoids, which describe the geometry of rigid body rotation in classical mechanics.

## Overview

This script visualizes the Poinsot construction, a geometric representation of the rotational motion of a rigid body. It plots two key ellipsoids:

- **Inertia Ellipsoid** (blue): Represents the constraint from conservation of angular momentum
- **Energy Ellipsoid** (red): Represents the constraint from conservation of kinetic energy

The intersection of these two ellipsoids forms the polhode, the curve traced by the angular velocity vector in the body frame.

![Poinsot Ellipsoids Visualization](poinsot_ellipsoids.png)

## Physics Background

For a rotating rigid body with moments of inertia I₁, I₂, I₃ and angular velocity components ω₁, ω₂, ω₃:

- **Angular momentum squared** (conserved): L² = (I₁ω₁)² + (I₂ω₂)² + (I₃ω₃)²
- **Rotational kinetic energy** (conserved): T = ½(I₁ω₁² + I₂ω₂² + I₃ω₃²)

These conservation laws define two ellipsoidal surfaces in angular velocity space. The angular velocity vector must lie on both surfaces simultaneously, constraining its motion to the intersection curve (polhode).

## Requirements

- Python 3.x
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install numpy matplotlib
```

## Usage

### Basic usage (default configuration)
```bash
python poinsot_ellipsoid.py
```
Default: ω = (0, 1, 0)

### Specify custom angular velocity components
```bash
python poinsot_ellipsoid.py --omega1 0.5 --omega2 0.8 --omega3 0.3
```

### Specify individual components
```bash
python poinsot_ellipsoid.py --omega2 1.0
python poinsot_ellipsoid.py --omega1 0.5 --omega3 0.3
```

Unspecified components default to 0.

## Output

The script generates:
- An interactive 3D visualization window
- A saved PNG image: `poinsot_ellipsoids.png`
- Console output with calculated values:
  - Moments of inertia
  - Angular velocity components
  - Angular momentum squared (L²)
  - Rotational kinetic energy (T)

## Configuration

The moments of inertia are hardcoded in the script (poinsot_ellipsoid.py:24):
```python
I1, I2, I3 = 5.0, 3.0, 2.0  # Assumes I1 > I2 > I3
```

Modify these values to explore different rigid body configurations.

## Visualization Details

The plot shows:
- Blue semi-ellipsoid: Inertia ellipsoid surface
- Red semi-ellipsoid: Energy ellipsoid surface
- Green dashed curve: Approximate intersection (polhode)
- Black point: Given angular velocity vector ω
- Coordinate axes: ω₁, ω₂, ω₃

Only the positive octant is displayed for clarity.

## Example

```bash
python poinsot_ellipsoid.py --omega1 0.2 --omega2 0.8 --omega3 0.1
```

Output:
```
Moments of inertia: I1=5.0, I2=3.0, I3=2.0
Angular velocity: ω=(0.2, 0.8, 0.1)
Angular momentum squared: L²=6.770
Rotational kinetic energy: T=1.030
```
