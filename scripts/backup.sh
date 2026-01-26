#!/bin/bash
# Database Backup Script for BO-MCP-UI
#
# Creates a compressed PostgreSQL backup with timestamp.
# Backups are stored in ./backups/ by default.
#
# Usage:
#   ./scripts/backup.sh                    # Use defaults
#   BACKUP_DIR=/path/to/dir ./backup.sh    # Custom backup directory
#   CONTAINER=my-db-container ./backup.sh  # Custom container name
#
# Reference: https://www.postgresql.org/docs/current/app-pgdump.html

set -euo pipefail

# Configuration with sensible defaults
BACKUP_DIR="${BACKUP_DIR:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONTAINER="${CONTAINER:-bo-mcp-ui-db-1}"
DB_USER="${POSTGRES_USER:-bo_user}"
DB_NAME="${POSTGRES_DB:-bo_mcp}"

# Ensure backup directory exists
mkdir -p "$BACKUP_DIR"

# Create backup filename
BACKUP_FILE="$BACKUP_DIR/backup_${TIMESTAMP}.sql.gz"

echo "Starting backup..."
echo "  Container: $CONTAINER"
echo "  Database:  $DB_NAME"
echo "  Output:    $BACKUP_FILE"

# Create compressed backup
# --clean: Include DROP statements before CREATE
# --if-exists: Avoid errors if objects don't exist during restore
# --no-owner: Omit ownership commands (useful for different target users)
docker exec -t "$CONTAINER" pg_dump \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    --clean \
    --if-exists \
    --no-owner \
    | gzip > "$BACKUP_FILE"

# Verify backup was created and has content
if [[ -s "$BACKUP_FILE" ]]; then
    BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    echo "Backup created successfully: $BACKUP_FILE ($BACKUP_SIZE)"
else
    echo "ERROR: Backup file is empty or was not created" >&2
    rm -f "$BACKUP_FILE"
    exit 1
fi

# Optional: Clean up old backups (keep last 7 days)
if [[ "${CLEANUP_OLD_BACKUPS:-false}" == "true" ]]; then
    echo "Cleaning up backups older than 7 days..."
    find "$BACKUP_DIR" -name "backup_*.sql.gz" -mtime +7 -delete
fi

echo "Backup complete!"
