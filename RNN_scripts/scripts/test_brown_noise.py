# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 15:35:22 2025

@author: RHIRAsimulation
"""
import numpy as np
import matplotlib.pyplot as plt

def generate_2d_brown_noise(size, scale=1.0):
    """
    Generate 2D Brownian noise.

    Parameters:
        size (tuple): Dimensions of the 2D noise (height, width).
        scale (float): Scaling factor for the noise intensity.

    Returns:
        numpy.ndarray: 2D Brownian noise.
    """
    # Create white noise (Gaussian random values)
    white_noise = np.random.normal(size=size)

    # Compute the 2D Fourier transform of the white noise
    fft_noise = np.fft.fft2(white_noise)

    # Create frequency grids
    freq_y = np.fft.fftfreq(size[0])
    freq_x = np.fft.fftfreq(size[1])
    freq_x, freq_y = np.meshgrid(freq_x, freq_y)

    # Compute the radial frequency
    radial_freq = np.sqrt(freq_x**2 + freq_y**2)

    # Avoid division by zero for the DC component
    radial_freq[radial_freq == 0] = np.min(radial_freq[radial_freq > 0])

    # Scale the Fourier transform by 1/frequency to get Brownian noise
    scaled_fft = fft_noise / radial_freq

    # Transform back to the spatial domain
    brown_noise = np.fft.ifft2(scaled_fft).real

    # Normalize and scale the noise
    brown_noise = scale * (brown_noise - brown_noise.min()) / (brown_noise.max() - brown_noise.min())

    return brown_noise

# Parameters
size = (512, 2500)  # Dimensions of the 2D noise
scale = 1.0        # Intensity scaling factor

# Generate 2D Brownian noise
brown_noise = generate_2d_brown_noise(size, scale)

# Plot the noise
plt.figure(figsize=(6, 6))
plt.imshow(brown_noise, cmap="gray", origin="upper")
plt.colorbar(label="Intensity")
plt.title("2D Brownian Noise")
plt.axis("off")
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt

def generate_2d_brown_noise(size, scale=1.0, freq_scale_x=1.0, freq_scale_y=1.0):
    """
    Generate 2D Brownian noise with adjustable frequency scaling.

    Parameters:
        size (tuple): Dimensions of the 2D noise (height, width).
        scale (float): Scaling factor for the noise intensity.
        freq_scale_x (float): Scaling factor for frequencies along the x-axis.
        freq_scale_y (float): Scaling factor for frequencies along the y-axis.

    Returns:
        numpy.ndarray: 2D Brownian noise.
    """
    # Create white noise (Gaussian random values)
    white_noise = np.random.normal(size=size)

    # Compute the 2D Fourier transform of the white noise
    fft_noise = np.fft.fft2(white_noise)

    # Create frequency grids
    freq_y = np.fft.fftfreq(size[0]) * freq_scale_y
    freq_x = np.fft.fftfreq(size[1]) * freq_scale_x
    freq_x, freq_y = np.meshgrid(freq_x, freq_y)

    # Compute the radial frequency
    radial_freq = np.sqrt(freq_x**2 + freq_y**2)

    # Avoid division by zero for the DC component
    radial_freq[radial_freq == 0] = np.min(radial_freq[radial_freq > 0])

    # Scale the Fourier transform by 1/frequency to get Brownian noise
    scaled_fft = fft_noise / np.power(radial_freq, 2.0)

    # Transform back to the spatial domain
    brown_noise = np.fft.ifft2(scaled_fft).real

    # Normalize and scale the noise
    #brown_noise = scale * (brown_noise - brown_noise.min()) / (brown_noise.max() - brown_noise.min())

    return brown_noise

# Parameters
size = (512, 512*5)  # Dimensions of the 2D noise
scale = 1.0        # Intensity scaling factor
freq_scale_x = 5 # Frequency scaling along the x-axis
freq_scale_y = 1 # Frequency scaling along the y-axis

# Generate 2D Brownian noise
brown_noise = generate_2d_brown_noise(size, scale, freq_scale_x, freq_scale_y)

# Plot the noise
plt.figure(figsize=(6, 6))
plt.imshow(brown_noise, cmap="gray", origin="upper")
plt.colorbar(label="Intensity")
plt.title("2D Brownian Noise")
plt.axis("off")
plt.show()
