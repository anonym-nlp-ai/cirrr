#!/bin/bash

# Default value for Fabric Manager installation
INSTALL_FABRIC_MANAGER=${INSTALL_FABRIC_MANAGER:-"false"}

# Verify NVCC version
echo "$(date +"%Y-%m-%d %H:%M:%S") - Checking NVCC version..."
nvcc --version

# Verify NVIDIA-SMI
echo "$(date +"%Y-%m-%d %H:%M:%S") - Checking NVIDIA-SMI..."
nvidia-smi

# Update package lists
echo "$(date +"%Y-%m-%d %H:%M:%S") - Updating package lists..."
sudo apt update -y
echo "$(date +"%Y-%m-%d %H:%M:%S") - Package lists updated."

# Install NVIDIA CUDA Toolkit
echo "$(date +"%Y-%m-%d %H:%M:%S") - Installing NVIDIA CUDA Toolkit..."
sudo apt install -y nvidia-cuda-toolkit
echo "$(date +"%Y-%m-%d %H:%M:%S") - NVIDIA CUDA Toolkit installed."

# Verify NVCC version
echo "$(date +"%Y-%m-%d %H:%M:%S") - Checking NVCC version..."
nvcc --version

# Verify NVIDIA-SMI
echo "$(date +"%Y-%m-%d %H:%M:%S") - Checking NVIDIA-SMI..."
nvidia-smi

# Install NVIDIA Fabric Manager if requested
if [ "$INSTALL_FABRIC_MANAGER" == "true" ]; then
    echo "$(date +"%Y-%m-%d %H:%M:%S") - Installing NVIDIA Fabric Manager..."
    apt search nvidia-fabricmanager
    # TODO: match fabric manager version with nvidia driver version
    sudo apt-get install -V nvidia-fabricmanager-535
    
    # Check the Fabric Manager Status
    sudo systemctl status nvidia-fabricmanager
    
    # Start and enable Fabric Manager
    sudo systemctl start nvidia-fabricmanager
    sudo systemctl enable nvidia-fabricmanager
    
    echo "$(date +"%Y-%m-%d %H:%M:%S") - NVIDIA Fabric Manager installation and setup complete."

    # Check the Fabric Manager Status
    sudo systemctl status nvidia-fabricmanager    
else
    echo "$(date +"%Y-%m-%d %H:%M:%S") - Skipping NVIDIA Fabric Manager installation."
fi

# Install nvtop
echo "$(date +"%Y-%m-%d %H:%M:%S") - Install nvtop"
sudo snap install nvtop

echo "$(date +"%Y-%m-%d %H:%M:%S") - OS setup complete."