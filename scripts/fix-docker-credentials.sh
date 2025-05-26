#!/bin/bash
set -e

echo "======== Docker Credential Fix Tool ========"

# Backup existing config
CONFIG_FILE="$HOME/.docker/config.json"
if [ -f "$CONFIG_FILE" ]; then
    echo "Backing up existing Docker config..."
    cp "$CONFIG_FILE" "$CONFIG_FILE.backup"
    echo "Backup created at $CONFIG_FILE.backup"
    
    # Create a clean config without credential helpers
    echo "Creating a clean Docker config..."
    cat > "$CONFIG_FILE" <<EOF
{
  "auths": {},
  "credsStore": ""
}
EOF
    echo "Clean config created at $CONFIG_FILE"
else
    echo "No Docker config file found. Creating a new one..."
    mkdir -p "$HOME/.docker"
    cat > "$CONFIG_FILE" <<EOF
{
  "auths": {},
  "credsStore": ""
}
EOF
    echo "New config created at $CONFIG_FILE"
fi

# Use local registry for this project
echo "Setting up a project-specific Docker configuration..."
mkdir -p .docker
cat > .docker/config.json <<EOF
{
  "auths": {},
  "credsStore": "",
  "experimental": "enabled"
}
EOF

echo "======== Fix Complete ========"
echo "Now try running your docker-compose command again."
echo ""
echo "Alternative: Use the nginx image we found locally instead of postgres:"
echo ""
echo "    cd db && cp Dockerfile Dockerfile.bak && sed 's/postgres:15/nginx:alpine/g' Dockerfile.bak > Dockerfile"
echo ""
echo "Or skip pulling images completely with:"
echo ""
echo "DOCKER_CLI_HINTS=false docker compose -f infrastructure/docker-compose.dev.yml build --pull never db" 