FROM python:3.11-slim as builder
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM postgres:15

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Add health check
HEALTHCHECK --interval=5s --timeout=5s --retries=3 \
    CMD pg_isready -U postgres || exit 1

# Copy initialization scripts
COPY init.sql /docker-entrypoint-initdb.d/01-init.sql
COPY migrations/ /docker-entrypoint-initdb.d/migrations/
COPY migrate.py /docker-entrypoint-initdb.d/
COPY init-migrations.sh /docker-entrypoint-initdb.d/
RUN chmod +x /docker-entrypoint-initdb.d/init-migrations.sh

# Set environment variables
ENV POSTGRES_DB=mnist
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres

# Expose PostgreSQL port
EXPOSE 5432
