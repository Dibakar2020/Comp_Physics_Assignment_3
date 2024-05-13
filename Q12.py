import numpy as np
import matplotlib.pyplot as plt

# Define the functions g(x) and h(x)
def g(x):
    return np.exp(-x**2)

def h(x):
    return np.exp(-4*x**2)

# Analytical convolution
def analytical_convolution(x):
    return np.sqrt(np.pi/5) * np.exp(-(4/5)*x**2)

# Generate x values
x = np.linspace(-5, 5, 1000)
dx = x[1] - x[0]

# Compute the convolution analytically
conv_analytical = analytical_convolution(x)

# Compute the convolution via DFT
conv_dft = np.convolve(g(x), h(x), mode='same') * dx

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, conv_dft, label='DFT Convolution')
plt.plot(x, conv_analytical, label='Analytical Convolution', linestyle = "--")
plt.xlabel('x')
plt.ylabel('Convolution')
plt.title('Analytical vs. DFT Convolution')
plt.legend()
plt.grid(True)
plt.show()
