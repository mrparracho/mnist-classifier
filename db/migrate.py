import os
import sys
import logging
from typing import Set, List
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseMigrator:
    def __init__(self):
        self.db_config = {
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
            'database': os.getenv('POSTGRES_DB', 'mnist'),
            'host': 'localhost',
            'port': '5432'
        }
        self.conn = None

    def get_connection(self, database: str = None) -> psycopg2.extensions.connection:
        """Create a connection to the specified database."""
        config = self.db_config.copy()
        if database:
            config['database'] = database

        try:
            if self.conn and not self.conn.closed:
                return self.conn

            self.conn = psycopg2.connect(**config)
            self.conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            return self.conn
        except psycopg2.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def ensure_database_exists(self):
        """Create the database if it doesn't exist."""
        try:
            # Connect to default postgres database
            conn = self.get_connection('postgres')
            
            with conn.cursor() as cur:
                # Check if database exists
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (self.db_config['database'],)
                )
                exists = cur.fetchone()
                
                if not exists:
                    logger.info(f"Creating database {self.db_config['database']}")
                    cur.execute(f"CREATE DATABASE {self.db_config['database']}")
                    logger.info("Database created successfully")
                else:
                    logger.info(f"Database {self.db_config['database']} already exists")
        
        except psycopg2.Error as e:
            logger.error(f"Error creating database: {e}")
            raise
        finally:
            if conn and not conn.closed:
                conn.close()

    def get_applied_migrations(self) -> Set[str]:
        """Get list of already applied migrations."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Create migrations table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS migrations (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        checksum VARCHAR(64)
                    )
                """)
                
                # Get applied migrations
                cur.execute("SELECT name FROM migrations")
                return {row[0] for row in cur.fetchall()}
        except psycopg2.Error as e:
            logger.error(f"Error getting applied migrations: {e}")
            raise

    def get_pending_migrations(self) -> List[str]:
        """Get list of pending migrations."""
        migration_dir = os.path.join(os.path.dirname(__file__), 'migrations')
        migration_files = sorted([
            f for f in os.listdir(migration_dir)
            if f.endswith('.sql')
        ])
        
        applied_migrations = self.get_applied_migrations()
        return [f for f in migration_files if f not in applied_migrations]

    def apply_migration(self, migration_file: str):
        """Apply a single migration."""
        conn = self.get_connection()
        migration_path = os.path.join(
            os.path.dirname(__file__),
            'migrations',
            migration_file
        )
        
        try:
            # Read migration file
            with open(migration_path, 'r') as f:
                migration_sql = f.read()
            
            with conn.cursor() as cur:
                # Start transaction
                cur.execute("BEGIN")
                
                try:
                    # Apply migration
                    cur.execute(migration_sql)
                    
                    # Record migration
                    cur.execute(
                        "INSERT INTO migrations (name) VALUES (%s)",
                        (migration_file,)
                    )
                    
                    # Commit transaction
                    cur.execute("COMMIT")
                    logger.info(f"Successfully applied migration: {migration_file}")
                
                except Exception as e:
                    cur.execute("ROLLBACK")
                    logger.error(f"Error applying migration {migration_file}: {e}")
                    raise
        
        except Exception as e:
            logger.error(f"Error processing migration {migration_file}: {e}")
            raise

    def run_migrations(self):
        """Run all pending migrations."""
        try:
            pending_migrations = self.get_pending_migrations()
            
            if not pending_migrations:
                logger.info("No pending migrations")
                return
            
            logger.info(f"Found {len(pending_migrations)} pending migrations")
            
            for migration_file in pending_migrations:
                logger.info(f"Applying migration: {migration_file}")
                self.apply_migration(migration_file)
            
            logger.info("All migrations applied successfully")
        
        except Exception as e:
            logger.error(f"Error running migrations: {e}")
            raise

def main():
    """Main function to handle database setup and migrations."""
    try:
        logger.info("Starting database setup and migration")
        migrator = DatabaseMigrator()
        
        # Ensure database exists
        migrator.ensure_database_exists()
        
        # Run migrations
        migrator.run_migrations()
        
        logger.info("Database setup and migration completed successfully")
    
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 