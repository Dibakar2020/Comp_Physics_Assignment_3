import numpy as np
import time

def dft(x):
    N = len(x)
    dft_result = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        for n in range(N):
            dft_result[k] += x[n] * np.exp(-2j * np.pi * k * n / N) / (np.sqrt(N))
    return dft_result

# Number of elements in the sequence
n = 1000

# Generating a sample sequence
sequence = np.random.rand(n)

# Computing DFT using direct computation
start_time = time.time()
dft_result_direct = dft(sequence)
end_time = time.time()
direct_time = end_time - start_time

# Computing DFT using numpy.fft.fft
start_time = time.time()
dft_result_fft = np.fft.fft(sequence)
end_time = time.time()
fft_time = end_time - start_time

# Printing results and time taken
print("DFT using direct computation:", dft_result_direct)
print("DFT using numpy.fft.fft:", dft_result_fft)
print("Time taken by direct computation method:", direct_time, "seconds")
print("Time taken by numpy.fft.fft method:", fft_time, "seconds")
