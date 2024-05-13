import numpy as np
import time
import matplotlib.pyplot as plt

# Function to compute Discrete Fourier Transform (DFT)
def DFT(f):
    n = len(f)
    dft = np.zeros(n, dtype=np.complex128)

    # Iterate over all frequencies q
    for q in range(n):
        # Iterate over all time points p
        for p in range(n):
            # Compute the DFT using the formula
            dft[q] += (f[p] * np.exp(-2j * np.pi * p * q / n)) / (np.sqrt(n))

    return dft

# Define a range of n values for testing
n_values = range(4, 101)
direct_times = []  # List to store execution times for direct computation
fft_times = []     # List to store execution times for numpy.fft.fft

# Loop through different n values
for n in n_values:
    x = np.random.random(n)  # Generate a random signal of length n

    # Measure execution time for direct computation
    start_time = time.time()
    _ = DFT(x)
    end_time = time.time()
    direct_times.append(end_time - start_time)

    # Measure execution time for numpy.fft.fft
    start_time = time.time()
    _ = np.fft.fft(x)
    end_time = time.time()
    fft_times.append(end_time - start_time)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(n_values, direct_times, label='Direct Computation')
plt.plot(n_values, fft_times, label='numpy.fft.fft')
plt.legend()
plt.xlabel('n')
plt.ylabel('Time (seconds)')
plt.title('Time taken by DFT methods as a function of n')
plt.grid(True)
plt.show()
