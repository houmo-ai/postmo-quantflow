"""Durable, exclusive sample claims for independent GPU policy engines."""

import sqlite3
import time
from contextlib import contextmanager


class EvaluationQueue:
    def __init__(self, path):
        self.path = str(path)

    @contextmanager
    def connection(self):
        database = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        try:
            yield database
        finally:
            database.close()

    def initialize(self, count):
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS jobs (stage TEXT, sample INTEGER, status TEXT DEFAULT 'pending', "
                "owner TEXT, attempts INTEGER DEFAULT 0, started REAL, error TEXT, PRIMARY KEY(stage,sample))"
            )
            db.executemany(
                "INSERT OR IGNORE INTO jobs(stage,sample) VALUES (?,?)",
                [(stage, i) for stage in ("bf16", "w8") for i in range(count)],
            )

    def claim(self, stage, owner):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT sample FROM jobs WHERE stage=? AND status='pending' AND attempts<3 ORDER BY sample LIMIT 1",
                (stage,),
            ).fetchone()
            if row:
                db.execute(
                    "UPDATE jobs SET status='running',owner=?,started=?,attempts=attempts+1 WHERE stage=? AND sample=?",
                    (owner, time.time(), stage, row[0]),
                )
            db.execute("COMMIT")
        return row[0] if row else None

    def finish(self, stage, sample, owner):
        with self.connection() as db:
            result = db.execute(
                "UPDATE jobs SET status='done' WHERE stage=? AND sample=? AND status='running' AND owner=?",
                (stage, sample, owner),
            )
            if result.rowcount != 1:
                raise ValueError("sample claim no longer belongs to this engine")

    def mark_reused(self, stage, sample):
        with self.connection() as db:
            db.execute(
                "UPDATE jobs SET status='done',owner='validated_reuse' WHERE stage=? AND sample=?", (stage, sample)
            )

    def release_owner(self, owner):
        with self.connection() as db:
            db.execute(
                "UPDATE jobs SET status=CASE WHEN attempts<3 THEN 'pending' ELSE 'failed' END,owner=NULL "
                "WHERE status='running' AND owner=?",
                (owner,),
            )

    def recover(self):
        with self.connection() as db:
            db.execute(
                "UPDATE jobs SET status=CASE WHEN attempts<3 THEN 'pending' ELSE 'failed' END,owner=NULL "
                "WHERE status='running'"
            )

    def fail(self, stage, sample, owner, error):
        with self.connection() as db:
            db.execute(
                "UPDATE jobs SET status=CASE WHEN attempts<3 THEN 'pending' ELSE 'failed' END,error=?,owner=NULL "
                "WHERE stage=? AND sample=? AND owner=?",
                (str(error), stage, sample, owner),
            )

    def counts(self, stage):
        with self.connection() as db:
            return dict(
                db.execute("SELECT status,COUNT(*) FROM jobs WHERE stage=? GROUP BY status", (stage,)).fetchall()
            )
