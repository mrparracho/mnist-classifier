#!/bin/bash

# Script to run the database migration for multi-model support

set -e

echo "Running database migration for multi-model support..."

# Check if we're in a Docker environment
if [ -n "$DB_HOST" ]; then
    echo "Running migration in Docker environment..."
    
    # Wait for database to be ready
    echo "Waiting for database to be ready..."
    until pg_isready -h $DB_HOST -p $DB_PORT -U $DB_USER; do
        echo "Database is not ready yet. Waiting..."
        sleep 2
    done
    
    # Run migration
    echo "Running migration..."
    psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f /app/db/migrations/002_add_model_name_column.sql
    
else
    echo "Running migration in local environment..."
    
    # Check if psql is available
    if ! command -v psql &> /dev/null; then
        echo "Error: psql is not installed. Please install PostgreSQL client."
        exit 1
    fi
    
    # Set default values for local environment
    DB_HOST=${DB_HOST:-localhost}
    DB_PORT=${DB_PORT:-5432}
    DB_USER=${DB_USER:-postgres}
    DB_NAME=${DB_NAME:-mnist}
    
    echo "Using database: $DB_HOST:$DB_PORT/$DB_NAME (user: $DB_USER)"
    
    # Run migration
    psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f db/migrations/002_add_model_name_column.sql
fi

echo "Migration completed successfully!"
echo "The database now supports multiple models."
echo ""
echo "Available models:"
echo "- cnn_mnist (CNN MNIST)"
echo "- transformer1_mnist (Transformer1 MNIST)"
echo "- transformer2_mnist (Transformer2 MNIST)" 