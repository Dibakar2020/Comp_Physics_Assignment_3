import numpy as np
import time
import matplotlib.pyplot as plt

def dft(x):
    N = len(x)
    dft_result = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        for n in range(N):
            dft_result[k] += x[n] * np.exp(-2j * np.pi * k * n / N) / (np.sqrt(N))

    return dft_result

# Range of values for n
n_values = range(4, 101)

# Lists to store time taken by each method for each value of n
direct_times = []
fft_times = []

for n in n_values:
    # Generating a sample sequence of length n
    sequence = np.random.rand(n)
    
    # Computing DFT using direct computation
    start_time = time.time()
    dft_result_direct = dft(sequence)
    end_time = time.time()
    direct_time = end_time - start_time
    direct_times.append(direct_time)

    # Computing DFT using numpy.fft.fft
    start_time = time.time()
    dft_result_fft = np.fft.fft(sequence)
    end_time = time.time()
    fft_time = end_time - start_time
    fft_times.append(fft_time)

# Plotting the results
plt.plot(n_values, direct_times, label='Direct Computation')
plt.plot(n_values, fft_times, label='numpy.fft.fft')
plt.xlabel('Number of elements (n)')
plt.ylabel('Time taken (seconds)')
plt.title('Time taken by DFT methods as a function of n')
plt.legend()
plt.grid(True)
plt.show()
