import numpy as np
import matplotlib.pyplot as plt

# Define the original function
def f(x):
    return np.ones_like(x) * 2

# Define the range of x values and the number of sample points
xmin = -50
xmax = 50
n = 10

# Calculate the spacing between sample points
dx = (xmax - xmin) / (n - 1)

# Generate sample points
xx = np.linspace(xmin, xmax, n)
sample_value = f(xx)

# Perform Fourier Transform on the sample points
fft_sample = np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.array(sample_value)), norm="ortho"))

# Generate k values for the Fourier Transform
kk = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(n, dx))

# Calculate the integration factor
int_factor = dx * np.sqrt(n / (2 * np.pi) * np.exp(- 1j * kk * xmin))

# Apply the integration factor to the Fourier Transform
fft_sample = fft_sample * int_factor

# Plot the original function and sampled points
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(xx, f(xx), label='Original Function')
plt.scatter(xx, sample_value, label='Sampled Points', color='red')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Original Function and Sampled Points')
plt.legend()
plt.grid()

# Plot the magnitude of the Fourier Transform
plt.subplot(1, 2, 2)
plt.plot(kk, np.abs(fft_sample), label='Fourier Transform (Magnitude)', marker="o")
plt.xlabel('k')
plt.ylabel('|F(k)|')
plt.title('Magnitude of Fourier Transform')
plt.legend()
plt.grid()

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
