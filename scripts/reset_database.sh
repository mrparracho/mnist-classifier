#!/bin/bash

# Database Reset Script for MNIST Classifier
# This script clears all prediction and feedback data while preserving the database schema

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default database connection parameters
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-mnist}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-postgres}"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if running in Docker environment
check_docker_env() {
    if command -v docker-compose &> /dev/null; then
        if docker-compose -f infrastructure/docker-compose.dev.yml ps db | grep -q "Up"; then
            return 0  # Docker environment detected and running
        fi
    fi
    return 1  # Not in Docker environment or not running
}

# Function to execute SQL command
execute_sql() {
    local sql_command="$1"
    local description="$2"
    
    print_status "$description"
    
    if check_docker_env; then
        # Use docker-compose to execute in container
        docker-compose -f infrastructure/docker-compose.dev.yml exec -T db psql -U "$DB_USER" -d "$DB_NAME" -c "$sql_command"
    else
        # Use local psql connection
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "$sql_command"
    fi
    
    if [ $? -eq 0 ]; then
        print_success "$description completed"
    else
        print_error "$description failed"
        exit 1
    fi
}

# Function to get record counts
get_record_counts() {
    print_status "Getting current record counts..."
    
    if check_docker_env; then
        docker-compose -f infrastructure/docker-compose.dev.yml exec -T db psql -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 
            'predictions' as table_name, 
            COUNT(*) as record_count,
            COUNT(CASE WHEN true_label IS NOT NULL THEN 1 END) as with_feedback
        FROM predictions
        UNION ALL
        SELECT 
            'feedback_history' as table_name, 
            COUNT(*) as record_count,
            COUNT(*) as with_feedback
        FROM feedback_history;"
    else
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 
            'predictions' as table_name, 
            COUNT(*) as record_count,
            COUNT(CASE WHEN true_label IS NOT NULL THEN 1 END) as with_feedback
        FROM predictions
        UNION ALL
        SELECT 
            'feedback_history' as table_name, 
            COUNT(*) as record_count,
            COUNT(*) as with_feedback
        FROM feedback_history;"
    fi
}

# Function to show help
show_help() {
    echo "Database Reset Script for MNIST Classifier"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -y, --yes               Skip confirmation prompt"
    echo "  -s, --stats-only        Only show current database statistics"
    echo "  --feedback-only         Reset only feedback data (keep predictions)"
    echo "  --predictions-only      Reset only prediction data"
    echo "  --all                   Reset all data (default)"
    echo ""
    echo "Environment Variables:"
    echo "  DB_HOST                 Database host (default: localhost)"
    echo "  DB_PORT                 Database port (default: 5432)"
    echo "  DB_NAME                 Database name (default: mnist)"
    echo "  DB_USER                 Database user (default: postgres)"
    echo "  DB_PASSWORD             Database password (default: postgres)"
    echo ""
    echo "Examples:"
    echo "  $0                      # Reset all data with confirmation"
    echo "  $0 -y                   # Reset all data without confirmation"
    echo "  $0 --feedback-only      # Reset only feedback data"
    echo "  $0 -s                   # Show statistics only"
}

# Parse command line arguments
SKIP_CONFIRMATION=false
STATS_ONLY=false
RESET_MODE="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -y|--yes)
            SKIP_CONFIRMATION=true
            shift
            ;;
        -s|--stats-only)
            STATS_ONLY=true
            shift
            ;;
        --feedback-only)
            RESET_MODE="feedback"
            shift
            ;;
        --predictions-only)
            RESET_MODE="predictions"
            shift
            ;;
        --all)
            RESET_MODE="all"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main script execution
print_status "MNIST Classifier Database Reset Script"
print_status "======================================"

# Check database connection
print_status "Checking database connection..."
if check_docker_env; then
    print_status "Detected Docker environment"
    if ! docker-compose -f infrastructure/docker-compose.dev.yml exec -T db psql -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" > /dev/null 2>&1; then
        print_error "Cannot connect to database via Docker"
        exit 1
    fi
else
    print_status "Using direct database connection"
    if ! PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" > /dev/null 2>&1; then
        print_error "Cannot connect to database at $DB_HOST:$DB_PORT"
        print_warning "Make sure the database is running and connection parameters are correct"
        exit 1
    fi
fi
print_success "Database connection successful"

# Show current statistics
echo ""
print_status "Current Database Statistics:"
get_record_counts

# If stats only, exit here
if [ "$STATS_ONLY" = true ]; then
    exit 0
fi

echo ""

# Confirmation prompt
if [ "$SKIP_CONFIRMATION" = false ]; then
    case $RESET_MODE in
        "all")
            print_warning "This will DELETE ALL prediction and feedback data!"
            ;;
        "feedback")
            print_warning "This will DELETE ALL feedback data (predictions will be kept but feedback reset)!"
            ;;
        "predictions")
            print_warning "This will DELETE ALL prediction data!"
            ;;
    esac
    
    echo -n "Are you sure you want to continue? (y/N): "
    read -r response
    
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        print_status "Operation cancelled by user"
        exit 0
    fi
fi

echo ""
print_status "Starting database reset (mode: $RESET_MODE)..."

# Execute reset based on mode
case $RESET_MODE in
    "all")
        execute_sql "DELETE FROM feedback_history;" "Clearing feedback history"
        execute_sql "DELETE FROM predictions;" "Clearing all predictions"
        execute_sql "SELECT setval(pg_get_serial_sequence('predictions', 'id'), 1, false);" "Resetting sequence"
        ;;
    "feedback")
        execute_sql "DELETE FROM feedback_history;" "Clearing feedback history"
        execute_sql "UPDATE predictions SET true_label = NULL, is_correct = NULL, feedback_submitted = false, updated_at = NULL;" "Resetting feedback data in predictions"
        ;;
    "predictions")
        execute_sql "DELETE FROM feedback_history;" "Clearing feedback history"
        execute_sql "DELETE FROM predictions;" "Clearing all predictions"
        execute_sql "SELECT setval(pg_get_serial_sequence('predictions', 'id'), 1, false);" "Resetting sequence"
        ;;
esac

# Show final statistics
echo ""
print_status "Final Database Statistics:"
get_record_counts

echo ""
print_success "Database reset completed successfully!"
print_status "You can now start fresh with new predictions." 