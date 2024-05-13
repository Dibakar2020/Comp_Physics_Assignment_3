import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the Gaussian function
def gaussian(x, y):
    return np.exp(-(x**2 + y**2))

# Define analytical solution
def gaussian_ft(u, v):
    return (1 / 2) * np.exp(-(u**2 + v**2) / 4)

# Define range and number of points
n = 64
xmin = -5
xmax = 5
ymin = -5
ymax = 5
dx = (xmax - xmin) / (n - 1)
dy = (ymax - ymin) / (n - 1)
x = np.linspace(xmin, xmax, n)
y = np.linspace(ymin, ymax, n)
# Create a grid
X, Y = np.meshgrid(x, y)

# Compute the function values
f_values = gaussian(X, Y)

# Compute the Fourier transform
F_values = np.fft.fft2(f_values, norm = "ortho")
F_values_shifted = np.fft.fftshift(F_values)  # Shift the zero frequency component to the center

# Create figure and subplots
fig = plt.figure(figsize=(10, 5))

#
U = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(n, dx))
V = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(n, dy))
U, V = np.meshgrid(U, V)

int_factor = (dx * dy * n / (2 * np.pi)) * np.exp(- 1j * (U * xmin + V * ymin))

F_values_shifted = F_values_shifted * int_factor
# Plot the numerical result
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(U, V, np.abs(F_values_shifted), cmap='viridis')
ax1.set_title('Numerical Solution')
ax1.set_xlabel('Frequency (u)')
ax1.set_ylabel('Frequency (v)')
ax1.set_zlabel('Magnitude')

# Plot the analytical solution

analytical_values = gaussian_ft(U, V)
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(U, V, analytical_values, cmap='plasma', alpha=0.5)
ax2.set_title('Analytical Solution')
ax2.set_xlabel('Frequency (u)')
ax2.set_ylabel('Frequency (v)')
ax2.set_zlabel('Magnitude')
plt.tight_layout()
plt.show()
