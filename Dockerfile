# Use TensorFlow 2.1.0 base image with GPU support (CUDA 10.1)
FROM tensorflow/tensorflow:2.1.0-gpu-py3

# Set up a working directory inside the container
WORKDIR /workspace

# Install required Python libraries
RUN pip install keras==2.3.1 \
    numpy \
    scipy \
    matplotlib \
    scikit-learn \
    jupyter \
    ipykernel

# Expose Jupyter port (only needed when using Jupyter)
EXPOSE 8888


