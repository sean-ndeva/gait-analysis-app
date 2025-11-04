#!/bin/bash
# setup.sh - Installation script for Streamlit Community Cloud

# Update package list and install system dependencies
apt-get update
apt-get install -y pkg-config python3-dev build-essential \
    libavdevice-dev libavformat-dev libavcodec-dev \
    libavutil-dev libswscale-dev libswresample-dev

# Install Python packages
pip install -r requirements.txt