import numpy as np
import matplotlib.pyplot as plt

# Define the given function 
def f(x):
    return np.sin(x) / x

# Define the range and number of points for x
xmin = -50
xmax = 50
n = 256

# Calculate the spacing between points
dx = (xmax - xmin) / (n - 1)

# Generate the array of x values
xx = np.linspace(xmin, xmax, n)

# Calculate the function values at the sample points
sample_value = f(xx)

# Perform numerical Fourier transformation
fft_sample = np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.array(sample_value)), norm="ortho"))

# Generate the array of k values
kk = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(n, dx))

# Calculate the integration factor
int_factor = dx * np.sqrt(n / (2 * np.pi) * np.exp(- 1j * kk * xmin))

# Apply the integration factor to the Fourier transformed data
fft_sample = fft_sample * int_factor

# Calculate the analytical solution
fk = np.zeros(len(kk))
for i in range(len(kk)):
    if np.abs(kk[i]) <= 1: 
        fk[i] = np.sqrt(np.pi/2)
    else: 
        fk[i] = 0

# Plotting the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(xx, f(xx), label='Original Function')
plt.scatter(xx, sample_value, label='Sampled Points', color='red')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Original Function and Sampled Points')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(kk, np.abs(fft_sample), label='Fourier Transform (Magnitude)')
plt.plot(kk, fk, label='Analytical Solution')
plt.xlabel('k')
plt.ylabel('|F(k)|')
plt.title('Numerical vs Analytical Fourier Transform')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
