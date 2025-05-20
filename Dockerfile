# 1. Use an official Python runtime as a parent image
FROM python:3.11-slim-bullseye

# 2. Set environment variables to prevent Python from writing .pyc files to disc and for unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory in the container
WORKDIR /app

# 4. Install system dependencies and build TA-Lib from source
RUN apt-get update && \
    # Install core build tools from Bullseye first
    apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        build-essential \
        make && \
    # Now, set up Bookworm for specific newer tools
    echo 'deb http://deb.debian.org/debian bookworm main' > /etc/apt/sources.list.d/bookworm.list && \
    echo 'Package: *\nPin: release n=bullseye\nPin-Priority: 900\n\nPackage: *\nPin: release n=bookworm\nPin-Priority: 100\n\nPackage: automake\nPin: release n=bookworm\nPin-Priority: 990\n\nPackage: autoconf\nPin: release n=bookworm\nPin-Priority: 990\n\nPackage: libtool\nPin: release n=bookworm\nPin-Priority: 990' > /etc/apt/preferences.d/bookworm.pref && \
    apt-get update && \
    # Install pinned tools from Bookworm
    apt-get install -y --no-install-recommends \
        automake \
        autoconf \
        libtool && \
    # Download TA-Lib C library source
    wget https://github.com/TA-Lib/ta-lib/archive/refs/tags/v0.6.4.tar.gz -O ta-lib-0.6.4.tar.gz && \
    # Extract source
    tar -xzf ta-lib-0.6.4.tar.gz && \
    # Navigate into source directory
    cd ta-lib-0.6.4 && \
    # Ensure autogen.sh is executable, then generate configure script, then configure, compile, and install
    chmod +x autogen.sh && \
    ./autogen.sh && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    # Clean up TA-Lib source files
    cd .. && \
    rm -rf ta-lib-0.6.4.tar.gz ta-lib-0.6.4 && \
    # Clean up APT pinning and caches
    rm /etc/apt/sources.list.d/bookworm.list && \
    rm /etc/apt/preferences.d/bookworm.pref && \
    apt-get autoremove -y && \
    apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 5. Copy the requirements.txt file into the container at /app
COPY requirements.txt .

# 6. Upgrade pip and then install any needed packages specified in requirements.txt
# Using --no-cache-dir to keep the image size down
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of the application's code into the container at /app
# This includes main.py, config.yaml, and the src/ directory.
COPY . .

# 8. Specify the command to run when the container starts
CMD ["python", "main.py"]
