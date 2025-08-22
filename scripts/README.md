# Scripts Directory

This directory contains utility scripts for managing the MNIST Classifier project.

## Available Scripts

### `reset_database.sh`

A comprehensive database reset script that can clear prediction and feedback data while preserving the database schema.

#### Usage

```bash
# Show help
./scripts/reset_database.sh --help

# Show current database statistics only
./scripts/reset_database.sh -s

# Reset all data with confirmation prompt
./scripts/reset_database.sh

# Reset all data without confirmation
./scripts/reset_database.sh -y

# Reset only feedback data (keep predictions)
./scripts/reset_database.sh --feedback-only

# Reset only prediction data
./scripts/reset_database.sh --predictions-only
```

#### Features

- **Smart Environment Detection**: Automatically detects if running in Docker environment
- **Multiple Reset Modes**:
  - `--all`: Reset all prediction and feedback data (default)
  - `--feedback-only`: Reset only feedback data, keep predictions
  - `--predictions-only`: Reset prediction data and associated feedback
- **Safety Features**:
  - Confirmation prompts (can be skipped with `-y`)
  - Connection validation before operations
  - Detailed status reporting
- **Statistics Display**: Shows record counts before and after operations
- **Colored Output**: Easy-to-read status messages with color coding

#### Environment Variables

The script uses the following environment variables (with defaults):

```bash
DB_HOST=localhost        # Database host
DB_PORT=5432            # Database port  
DB_NAME=mnist           # Database name
DB_USER=postgres        # Database user
DB_PASSWORD=postgres    # Database password
```

#### Examples

**Development workflow:**
```bash
# Check current state
./scripts/reset_database.sh -s

# Reset only feedback to test feedback system
./scripts/reset_database.sh --feedback-only -y

# Complete reset for fresh start
./scripts/reset_database.sh -y
```

**Production-like testing:**
```bash
# Reset in a controlled way
./scripts/reset_database.sh --all
# Confirms before proceeding
```

### Other Scripts

- `docker-diagnostic.sh`: Docker environment diagnostics
- `fix-docker-credentials.sh`: Docker credentials fixing
- `run_migration.sh`: Database migration runner
- `setup_checkpoints.sh`: Model checkpoint setup
- `setup_venvs.sh`: Virtual environment setup

## Script Permissions

Make sure scripts are executable:

```bash
chmod +x scripts/*.sh
```

## Safety Notes

⚠️ **Important**: The database reset script will permanently delete data. Always:

1. Use `--stats-only` first to see what will be affected
2. Consider using `--feedback-only` to preserve prediction data
3. Take backups if needed before running in production environments
4. Test the script in development environment first

## Troubleshooting

**Connection Issues:**
- Ensure database is running: `docker-compose ps db`
- Check environment variables are set correctly
- Verify network connectivity if not using Docker

**Permission Issues:**
- Make scripts executable: `chmod +x scripts/reset_database.sh`
- Check database user permissions

**Docker Issues:**
- Ensure docker-compose.yml file path is correct
- Verify containers are running: `docker-compose ps` 