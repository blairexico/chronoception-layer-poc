"""Persistent temporal fact storage using SQLite."""

import logging
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from .config import DatabaseConfig
from .types import EPOCH, DatabaseError, TemporalFact

logger = logging.getLogger(__name__)


class TemporalDatabase:
    """Manages persistent temporal facts in SQLite."""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        try:
            self.conn = sqlite3.connect(self.config.db_path)
            self.conn.row_factory = sqlite3.Row
            self._initialize_schema()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to connect to database at {self.config.db_path}") from e

    def _initialize_schema(self) -> None:
        try:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS temporal_facts (
                    fact_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    fact_type TEXT NOT NULL,
                    base_date TEXT NOT NULL,
                    position_absolute INTEGER NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')

            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS fact_aliases (
                    alias_id TEXT PRIMARY KEY,
                    fact_id TEXT NOT NULL,
                    alias_text TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 1,
                    FOREIGN KEY (fact_id) REFERENCES temporal_facts(fact_id)
                )
            ''')

            self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_position
                ON temporal_facts(user_id, position_absolute)
            ''')

            self.conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError("Failed to initialize database schema") from e

    def add_fact(
        self,
        user_id: str,
        canonical_name: str,
        fact_type: str,
        base_date: datetime,
        aliases: Optional[List[str]] = None,
        confidence: float = 1.0,
    ) -> str:
        """Store a new temporal fact. Returns the fact_id."""
        fact_id = str(uuid.uuid4())

        if isinstance(base_date, str):
            base_date = datetime.fromisoformat(base_date)

        position_abs = (base_date - EPOCH).days
        now = datetime.now().isoformat()

        try:
            self.conn.execute('''
                INSERT INTO temporal_facts
                (fact_id, user_id, canonical_name, fact_type, base_date,
                 position_absolute, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (fact_id, user_id, canonical_name, fact_type,
                  base_date.isoformat(), position_abs, confidence, now, now))

            if aliases:
                for alias in aliases:
                    self.conn.execute('''
                        INSERT INTO fact_aliases (alias_id, fact_id, alias_text)
                        VALUES (?, ?, ?)
                    ''', (str(uuid.uuid4()), fact_id, alias.lower()))

            self.conn.commit()
            logger.info("Stored fact: %s (%s) for user %s", canonical_name, fact_type, user_id)
            return fact_id

        except sqlite3.Error as e:
            self.conn.rollback()
            raise DatabaseError(f"Failed to store fact '{canonical_name}'") from e

    def get_facts_for_user(self, user_id: str) -> List[TemporalFact]:
        """Get all facts for a user as TemporalFact objects."""
        try:
            rows = self.conn.execute(
                "SELECT * FROM temporal_facts WHERE user_id = ?",
                [user_id]
            ).fetchall()

            facts = []
            for row in rows:
                row_dict = dict(row)
                # Fetch aliases for this fact
                alias_rows = self.conn.execute(
                    "SELECT alias_text FROM fact_aliases WHERE fact_id = ?",
                    [row_dict['fact_id']]
                ).fetchall()
                aliases = [a['alias_text'] for a in alias_rows]

                facts.append(TemporalFact(
                    fact_id=row_dict['fact_id'],
                    user_id=row_dict['user_id'],
                    canonical_name=row_dict['canonical_name'],
                    fact_type=row_dict['fact_type'],
                    base_date=datetime.fromisoformat(row_dict['base_date']),
                    position_absolute=row_dict['position_absolute'],
                    confidence=row_dict['confidence'],
                    created_at=datetime.fromisoformat(row_dict['created_at']),
                    updated_at=datetime.fromisoformat(row_dict['updated_at']),
                    aliases=aliases,
                ))
            return facts

        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to retrieve facts for user '{user_id}'") from e

    def find_fact_by_alias(self, user_id: str, alias_text: str) -> Optional[TemporalFact]:
        """Resolve alias to a fact."""
        try:
            row = self.conn.execute('''
                SELECT f.* FROM temporal_facts f
                JOIN fact_aliases a ON f.fact_id = a.fact_id
                WHERE f.user_id = ? AND LOWER(a.alias_text) = LOWER(?)
            ''', (user_id, alias_text)).fetchone()

            if row is None:
                return None

            row_dict = dict(row)
            return TemporalFact(
                fact_id=row_dict['fact_id'],
                user_id=row_dict['user_id'],
                canonical_name=row_dict['canonical_name'],
                fact_type=row_dict['fact_type'],
                base_date=datetime.fromisoformat(row_dict['base_date']),
                position_absolute=row_dict['position_absolute'],
                confidence=row_dict['confidence'],
                created_at=datetime.fromisoformat(row_dict['created_at']),
                updated_at=datetime.fromisoformat(row_dict['updated_at']),
            )

        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to resolve alias '{alias_text}'") from e

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
