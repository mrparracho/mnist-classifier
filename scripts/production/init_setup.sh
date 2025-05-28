#!/bin/bash

echo "Starting system setup..."

echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "Installing Docker and Docker Compose..."
sudo apt install -y docker.io docker-compose

echo "Adding user to docker group..."
sudo usermod -aG docker $USER

echo "Installing Git..."
sudo apt install -y git

echo "Installing Python 3.11..."
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

echo "Cloning repository..."
git clone https://github.com/mrparracho/mnist-classifier.git
cd mnist-classifier

echo "Setting up SSL certificates..."

# 1. Create SSL directory if it doesn't exist
echo "Creating SSL directory..."
mkdir -p infrastructure/nginx/ssl

# 2. Generate self-signed certificate
echo "Generating self-signed certificate..."
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout infrastructure/nginx/ssl/key.pem \
  -out infrastructure/nginx/ssl/cert.pem \
  -subj "/CN=$(hostname -I | awk '{print $1}')"

# 3. Set proper permissions
echo "Setting SSL certificate permissions..."
sudo chown -R $USER:$USER infrastructure/nginx/ssl
chmod 600 infrastructure/nginx/ssl/key.pem
chmod 644 infrastructure/nginx/ssl/cert.pem

echo "SSL certificates have been generated and permissions set correctly."
echo "Certificate location: infrastructure/nginx/ssl/cert.pem"
echo "Key location: infrastructure/nginx/ssl/key.pem" 

echo "Installing monitoring tools..."
sudo apt install -y htop netdata

echo "Enabling and starting netdata service..."
sudo systemctl enable netdata
sudo systemctl start netdata

echo "Setting up firewall rules..."

echo "Opening required ports..."
# Allow SSH (port 22)
sudo ufw allow 22/tcp

# Allow HTTP (port 80)
sudo ufw allow 80/tcp

# Allow HTTPS (port 443)
sudo ufw allow 443/tcp

echo "Enabling firewall..."
sudo ufw enable

echo "Firewall status:"
sudo ufw status verbose

echo "Setup completed successfully!"
echo "The following services are now configured:"
echo "- SSH (port 22)"
echo "- HTTP (port 80)"
echo "- HTTPS (port 443)"
echo "- Docker and Docker Compose"
echo "- Python 3.11"
echo "- Monitoring tools (htop, netdata)" 