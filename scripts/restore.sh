#!/bin/bash
# Database Restore Script for BO-MCP-UI
#
# Restores a PostgreSQL database from a backup file.
# Supports both compressed (.gz) and uncompressed (.sql) backups.
#
# Usage:
#   ./scripts/restore.sh backup_20250116_120000.sql.gz
#   ./scripts/restore.sh /path/to/backup.sql
#   CONTAINER=my-db-container ./restore.sh backup.sql.gz
#
# WARNING: This will DROP existing tables and recreate them from the backup.
#
# Reference: https://www.postgresql.org/docs/current/app-psql.html

set -euo pipefail

# Validate arguments
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <backup_file>" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  $0 backups/backup_20250116_120000.sql.gz" >&2
    echo "  $0 /path/to/backup.sql" >&2
    exit 1
fi

BACKUP_FILE="${1}"
CONTAINER="${CONTAINER:-bo-mcp-ui-db-1}"
DB_USER="${POSTGRES_USER:-bo_user}"
DB_NAME="${POSTGRES_DB:-bo_mcp}"

# Verify backup file exists
if [[ ! -f "$BACKUP_FILE" ]]; then
    echo "ERROR: Backup file not found: $BACKUP_FILE" >&2
    exit 1
fi

echo "Starting restore..."
echo "  Container: $CONTAINER"
echo "  Database:  $DB_NAME"
echo "  Source:    $BACKUP_FILE"
echo ""

# Confirmation prompt (skip in non-interactive mode)
if [[ -t 0 ]]; then
    read -p "WARNING: This will overwrite the existing database. Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Restore cancelled."
        exit 0
    fi
fi

# Restore based on file type
if [[ "$BACKUP_FILE" == *.gz ]]; then
    echo "Decompressing and restoring..."
    gunzip -c "$BACKUP_FILE" | docker exec -i "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" --quiet
else
    echo "Restoring..."
    docker exec -i "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" --quiet < "$BACKUP_FILE"
fi

# Verify restore succeeded by checking table count
TABLE_COUNT=$(docker exec -t "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -tAc \
    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")

echo ""
echo "Restore complete!"
echo "  Tables restored: $TABLE_COUNT"
