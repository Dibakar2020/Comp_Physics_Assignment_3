import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
import requests

# Download data from URL
url = "http://theory.tifr.res.in/~kulkarni/noise.txt"
response = requests.get(url)
data = np.fromstring(response.text, sep='\n')

# Plot the measurements
plt.figure(figsize=(10, 5))
plt.plot(data, marker = "o")
plt.title("Measurements")
plt.xlabel("Different measuremens")
plt.ylabel("Value")
plt.grid(True)
plt.show()

# Compute the Discrete Fourier Transform (DFT)
dft = np.fft.fft(data)
freq = np.fft.fftfreq(len(data))

# Plot the Fourier Transform
plt.figure(figsize=(10, 5))
plt.plot(freq, np.abs(dft))
plt.title("Fourier Transform")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Compute the Power Spectrum
frequencies, power_spectrum = periodogram(dft)

# Plot the Power Spectrum
plt.figure(figsize=(10, 5))
plt.plot(frequencies, power_spectrum)
plt.title("Power Spectrum")
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.grid(True)
plt.show()

# Bin the Power Spectrum
num_bins = 10
bins = np.linspace(frequencies.min(), frequencies.max(), num_bins + 1)
bin_indices = np.digitize(frequencies, bins)
binned_power_spectrum = np.zeros(num_bins)
for i in range(1, num_bins + 1):
    indices_in_bin = np.where(bin_indices == i)[0]
    binned_power_spectrum[i - 1] = np.mean(power_spectrum[indices_in_bin])

# Plot the Binned Power Spectrum as a bar plot
plt.figure(figsize=(10, 5))
plt.bar(bins[:-1], binned_power_spectrum, width=np.diff(bins), align='edge')
plt.title("Binned Power Spectrum")
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.grid(True)
plt.show()


