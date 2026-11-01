# count remote model calls per visitor (IP address) per day
# so 1 visitor or all visitors cannot burn our token credits

import sqlite3
import threading
from datetime import date

from src.config.settings import REMOTE_LIMIT_PER_IP, REMOTE_LIMIT_TOTAL, USAGE_DB_PATH

_lock = threading.Lock()

def _connect() -> sqlite3.Connection:
    # open the database file, create it and the table the first time
    USAGE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(USAGE_DB_PATH)
    db.execute(
        "CREATE TABLE IF NOT EXISTS usage("
        " day TEXT NOT NULL,"
        " ip TEXT NOT NULL,"
        " count INTEGER NOT NULL,"
        " PRIMARY KEY (day, ip))"
    )
    return db

def try_use_remote(ip: str) -> bool:
    # count one remote call for this IP today
    # return False (and count nothing) if a daily limit reached
    today = date.today().isoformat()
    with _lock:
        db = _connect()
        try: 
            row = db.execute(
                "SELECT count FROM usage WHERE day = ? AND ip = ?", (today, ip)
            ).fetchone()
            used_by_ip = row[0] if row else 0
            used_total = db.execute(
                "SELECT COALESCE(SUM(count), 0) FROM usage WHERE day = ?", (today,)
            ).fetchone()[0]

            if used_by_ip >= REMOTE_LIMIT_PER_IP or used_total >= REMOTE_LIMIT_TOTAL:
                return False

            # add a row with count 1, or add 1 to existing row
            db.execute(
                "INSERT INTO usage (day, ip, count) VALUES (?, ?, 1)"
                " ON CONFLICT (day, ip) DO UPDATE SET count = count + 1",
                (today, ip),
            )
            db.commit()
            return True
        finally:
            db.close()
