"""Migration script to add created_at column to conversation_histories table."""
import asyncio
import sys
from datetime import timezone

from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

from app.db import engine


async def migrate() -> None:
    """Add created_at column to conversation_histories if it doesn't exist."""
    async with engine.begin() as conn:
        # Check if column exists
        result = await conn.execute(
            text(
                """
                SELECT COUNT(*) as count
                FROM pragma_table_info('conversation_histories')
                WHERE name = 'created_at'
                """
            )
        )
        row = result.fetchone()
        has_column = row[0] > 0 if row else False

        if not has_column:
            print("Adding created_at column to conversation_histories...")
            # SQLite doesn't allow non-constant defaults, so add column first
            await conn.execute(
                text(
                    """
                    ALTER TABLE conversation_histories
                    ADD COLUMN created_at DATETIME
                    """
                )
            )
            # Update existing rows with current timestamp
            await conn.execute(
                text(
                    """
                    UPDATE conversation_histories
                    SET created_at = datetime('now')
                    WHERE created_at IS NULL
                    """
                )
            )
            print("✓ Added created_at column")
        else:
            print("✓ created_at column already exists")

        # Check if updated_at needs timezone-aware update
        # SQLite doesn't support timezone-aware datetimes natively,
        # but we can ensure the column exists
        result = await conn.execute(
            text(
                """
                SELECT COUNT(*) as count
                FROM pragma_table_info('conversation_histories')
                WHERE name = 'updated_at'
                """
            )
        )
        row = result.fetchone()
        has_updated_at = row[0] > 0 if row else False

        if not has_updated_at:
            print("Adding updated_at column to conversation_histories...")
            await conn.execute(
                text(
                    """
                    ALTER TABLE conversation_histories
                    ADD COLUMN updated_at DATETIME
                    """
                )
            )
            # Update existing rows with current timestamp
            await conn.execute(
                text(
                    """
                    UPDATE conversation_histories
                    SET updated_at = datetime('now')
                    WHERE updated_at IS NULL
                    """
                )
            )
            print("✓ Added updated_at column")
        else:
            print("✓ updated_at column already exists")

        # Also check workspaces and documents tables
        for table in ["workspaces", "documents"]:
            result = await conn.execute(
                text(
                    f"""
                    SELECT COUNT(*) as count
                    FROM pragma_table_info('{table}')
                    WHERE name = 'created_at'
                    """
                )
            )
            row = result.fetchone()
            has_created_at = row[0] > 0 if row else False

            if not has_created_at:
                print(f"Adding created_at column to {table}...")
                await conn.execute(
                    text(
                        f"""
                        ALTER TABLE {table}
                        ADD COLUMN created_at DATETIME
                        """
                    )
                )
                # Update existing rows
                await conn.execute(
                    text(
                        f"""
                        UPDATE {table}
                        SET created_at = datetime('now')
                        WHERE created_at IS NULL
                        """
                    )
                )
                print(f"✓ Added created_at column to {table}")
            else:
                print(f"✓ {table}.created_at column already exists")

    print("\n✓ Migration complete!")


if __name__ == "__main__":
    asyncio.run(migrate())

