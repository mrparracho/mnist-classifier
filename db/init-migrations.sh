#!/bin/bash
set -eo pipefail

MAX_RETRIES=30
RETRY_INTERVAL=2

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
for i in $(seq 1 $MAX_RETRIES); do
    if pg_isready -U postgres; then
        break
    fi
    
    if [ $i -eq $MAX_RETRIES ]; then
        echo "Timeout waiting for PostgreSQL to start"
        exit 1
    fi
    
    echo "Attempt $i of $MAX_RETRIES. Retrying in ${RETRY_INTERVAL}s..."
    sleep $RETRY_INTERVAL
done

# Run migrations
echo "Running database migrations..."
cd /docker-entrypoint-initdb.d
python3 migrate.py

echo "Database initialization complete!" 