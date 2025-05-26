#!/bin/bash
set -e

echo "========== Docker Diagnostic Tool =========="
echo "Checking Docker installation..."

# Check Docker version
echo "Docker version:"
if docker --version; then
    echo "✅ Docker is installed"
else
    echo "❌ Docker is not installed or not in PATH"
    exit 1
fi

# Check if Docker daemon is running
echo -e "\nChecking if Docker daemon is running..."
if docker info &>/dev/null; then
    echo "✅ Docker daemon is running"
else
    echo "❌ Docker daemon is not running"
    echo "Please start Docker Desktop or the Docker service"
    exit 1
fi

# Check Docker Compose
echo -e "\nChecking Docker Compose..."
if docker-compose --version; then
    echo "✅ Docker Compose is installed"
elif docker compose version; then
    echo "✅ Docker Compose plugin is installed"
else
    echo "❌ Docker Compose is not installed or not in PATH"
fi

# Check Docker login status
echo -e "\nChecking Docker Hub authentication..."
if docker info | grep -q "Username"; then
    echo "✅ Logged in to Docker Hub"
else
    echo "❌ Not logged in to Docker Hub"
    echo "Try running: docker login"
fi

# Test pulling a small image
echo -e "\nTesting image pull... (alpine is a tiny image)"
if docker pull alpine:latest; then
    echo "✅ Successfully pulled test image"
    # Clean up
    docker rmi alpine:latest &>/dev/null
else
    echo "❌ Failed to pull test image"
    echo "This indicates network or authentication issues"
fi

# Check credentials helper
echo -e "\nChecking Docker credential helpers..."
CREDENTIAL_HELPERS=$(find /usr/local/bin /usr/bin /bin -name "docker-credential-*" 2>/dev/null || echo "None found")
echo "Found credential helpers: $CREDENTIAL_HELPERS"

# Check config.json
echo -e "\nChecking Docker config.json..."
CONFIG_FILE="$HOME/.docker/config.json"
if [ -f "$CONFIG_FILE" ]; then
    echo "✅ Config file exists at: $CONFIG_FILE"
    echo "Credential helpers configured:"
    grep "credHelpers" "$CONFIG_FILE" || echo "No credential helpers found in config"
else
    echo "❌ Config file not found at: $CONFIG_FILE"
fi

echo -e "\n========== Diagnostic Complete =========="
echo "If you're still having issues, consider:"
echo "1. Reinstalling Docker Desktop"
echo "2. Manually logging in: docker login"
echo "3. Using a direct image reference with tag"
echo "4. Checking your network connectivity to Docker Hub" 