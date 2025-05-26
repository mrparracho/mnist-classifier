-- Create extensions first
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create initial user roles if needed
DO $$
BEGIN
    -- Create a read-only user if it doesn't exist
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'mnist_readonly') THEN
        CREATE ROLE mnist_readonly;
        GRANT CONNECT ON DATABASE mnist TO mnist_readonly;
        GRANT USAGE ON SCHEMA public TO mnist_readonly;
        GRANT SELECT ON ALL TABLES IN SCHEMA public TO mnist_readonly;
        GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO mnist_readonly;
        ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO mnist_readonly;
    END IF;
    
    -- Create a read-write user if it doesn't exist
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'mnist_readwrite') THEN
        CREATE ROLE mnist_readwrite;
        GRANT CONNECT ON DATABASE mnist TO mnist_readwrite;
        GRANT USAGE ON SCHEMA public TO mnist_readwrite;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO mnist_readwrite;
        GRANT SELECT, USAGE ON ALL SEQUENCES IN SCHEMA public TO mnist_readwrite;
        ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO mnist_readwrite;
        ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, USAGE ON SEQUENCES TO mnist_readwrite;
    END IF;
END
$$;

-- Set search path
SET search_path TO public;
