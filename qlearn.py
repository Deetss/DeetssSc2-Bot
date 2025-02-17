import os
import sqlite3
import numpy as np
import pandas as pd
import time

class QLearningTable:
    def __init__(self, actions, learning_rate=0.02, reward_decay=0.9, 
                 e_greedy=0.9, db_filename="q_table.db", db_dir="./db",history_length=1,):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.db_dir = db_dir
        if self.db_dir:
            os.makedirs(self.db_dir, exist_ok=True)
            self.db_filename = os.path.join(self.db_dir, db_filename)
        else:
            self.db_filename = db_filename

        self.history_length = history_length
        self.conn = self.create_connection()
        self.create_table()
        # In-memory cache of (state, action) combinations
        self.state_action_cache = set()
        self.q_table = self.load_q_table()

    def create_connection(self):
        conn = sqlite3.connect(self.db_filename, timeout=60, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def create_table(self):
        with self.conn:
            self.conn.execute("DROP TABLE IF EXISTS q_table")
            self.conn.execute("""
                CREATE TABLE q_table (
                    state TEXT NOT NULL,
                    action TEXT NOT NULL,
                    q_value REAL,
                    UNIQUE(state, action) ON CONFLICT REPLACE
                )
            """)
            # Create an index for faster lookup
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_state_action ON q_table(state, action)")

    def load_q_table(self):
        print(f"Loading Q-table from {self.db_filename}...")
        df = pd.read_sql_query("SELECT * FROM q_table", self.conn)
        if not df.empty:
            # Pivot, assuming 'state' and 'action' columns exist
            df = df.pivot(index='state', columns='action', values='q_value').fillna(0)
            print(f"Successfully loaded Q-table from {self.db_filename}.")
            return df
        else:
            print(f"No existing Q-table found in {self.db_filename}, initializing new table.")
            return pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        observation = str(observation)
        # ensure each possible action row exists for this state
        for act in self.actions:
            self.check_state_exist(observation, act)
        if np.random.uniform() < self.epsilon:
            best_action = None
            best_value = float("-inf")
            for act in self.actions:
                val = self.get_q_value(observation, act)
                if val > best_value:
                    best_value = val
                    best_action = act
            return best_action
        else:
            return np.random.choice(self.actions)

    def learn(self, s, a, r, s_):
        s = str(s)
        s_ = str(s_)
        for act in self.actions:
            self.check_state_exist(s, act)
            self.check_state_exist(s_, act)
        q_predict = self.get_q_value(s, a)
        q_target = r + self.gamma * self.get_max_q_value(s_)
        new_q = q_predict + self.lr * (q_target - q_predict)
        self.update_q_value(s, a, new_q)

    def check_state_exist(self, state, action):
        key = (state, action)
        if key in self.state_action_cache:
            return
        cursor = self.retry_db_operation(
            lambda: self.conn.execute(
                "SELECT 1 FROM q_table WHERE state = ? AND action = ?",
                (state, action)
            )
        )
        if not cursor.fetchone():
            self.retry_db_operation(
                lambda: self.conn.execute(
                    "INSERT OR IGNORE INTO q_table(state, action, q_value) VALUES(?,?,0)",
                    (state, action)
                )
            )
        # Mark this state-action as seen to avoid duplicate checks
        self.state_action_cache.add(key)

    def get_q_value(self, state, action):
        cursor = self.retry_db_operation(
            lambda: self.conn.execute(
                "SELECT q_value FROM q_table WHERE state = ? AND action = ?",
                (state, action)
            )
        )
        row = cursor.fetchone()
        return row[0] if row else 0

    def get_max_q_value(self, state):
        cursor = self.retry_db_operation(
            lambda: self.conn.execute(
                "SELECT MAX(q_value) FROM q_table WHERE state = ?",
                (state,)
            )
        )
        row = cursor.fetchone()
        return row[0] if row[0] else 0

    def update_q_value(self, state, action, value, lock=None):
        if lock:
            with lock:
                self.retry_db_operation(
                    lambda: self.conn.execute(
                        "INSERT OR REPLACE INTO q_table(state, action, q_value) VALUES(?,?,?)",
                        (state, action, value)
                    )
                )
        else:
            self.retry_db_operation(
                lambda: self.conn.execute(
                    "INSERT OR REPLACE INTO q_table(state, action, q_value) VALUES(?,?,?)",
                    (state, action, value)
                )
            )

    def save_model(self, filename=None):
        if filename:
            self.conn.commit()  # Ensure all transactions are committed
            # Prepend the db_dir if provided
            if self.db_dir:
                filename = os.path.join(self.db_dir, filename)
            # Remove existing file before running VACUUM INTO
            if os.path.exists(filename):
                os.remove(filename)
            self.retry_db_operation(
                lambda: self.conn.execute(f"VACUUM INTO '{filename}'")
            )

    def load_model(self):
        pass  # No need to load the model as SQLite handles it

    def retry_db_operation(self, operation, retries=5, delay=0.1):
        for attempt in range(retries):
            try:
                return operation()
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise
        raise sqlite3.OperationalError("Database is locked after multiple retries")

@staticmethod
def merge_q_tables_sql(num_workers, db_filename, db_dir=None):
    # Determine path for the base (merged) DB
    base_db_filename = os.path.join(db_dir, db_filename) if db_dir else db_filename
    conn = sqlite3.connect(base_db_filename)
    cur = conn.cursor()

    # Create a temporary table to collect all rows, tracking count per (state, action)
    cur.execute("DROP TABLE IF EXISTS merged_temp")
    cur.execute("""
        CREATE TABLE merged_temp (
            state TEXT NOT NULL,
            action TEXT NOT NULL,
            q_sum REAL NOT NULL,
            count INTEGER NOT NULL,
            PRIMARY KEY(state, action)
        )
    """)
    
    # Insert existing rows from the base DB, if any
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='q_table'")
    if cur.fetchone():
        cur.execute("""
            INSERT INTO merged_temp (state, action, q_sum, count)
            SELECT state, action, q_value, 1 FROM q_table
            ON CONFLICT(state, action) DO UPDATE SET 
                q_sum = merged_temp.q_sum + excluded.q_sum,
                count = merged_temp.count + 1
        """)
    
    # For each worker DB, attach it and merge data in a single SQL transaction
    for worker_id in range(num_workers):
        worker_file = f"q_table_worker_{worker_id}.db"
        worker_db_filename = os.path.join(db_dir, worker_file) if db_dir else worker_file
        if os.path.exists(worker_db_filename):
            attach_alias = f"worker_{worker_id}"
            cur.execute(f"ATTACH DATABASE '{worker_db_filename}' AS {attach_alias}")
            cur.execute(f"""
                INSERT INTO merged_temp (state, action, q_sum, count)
                SELECT state, action, q_value, 1 FROM {attach_alias}.q_table
                ON CONFLICT(state, action) DO UPDATE SET 
                    q_sum = merged_temp.q_sum + excluded.q_sum,
                    count = merged_temp.count + 1
            """)
            cur.execute(f"DETACH DATABASE {attach_alias}")
        else:
            print(f"No DB found for worker {worker_id} at {worker_db_filename}")
    
    # Now compute the averages and replace (or create) q_table in the base DB.
    cur.execute("DROP TABLE IF EXISTS q_table")
    cur.execute("""
        CREATE TABLE q_table AS
        SELECT state, action,
               q_sum * 1.0 / count as q_value
        FROM merged_temp
    """)
    
    conn.commit()
    conn.close()
    
    # Cleanup worker db files after a successful merge
    for worker_id in range(num_workers):
        worker_file = f"q_table_worker_{worker_id}.db"
        worker_db_filename = os.path.join(db_dir, worker_file) if db_dir else worker_file
        if os.path.exists(worker_db_filename):
            os.remove(worker_db_filename)
            print(f"Removed temporary file: {worker_db_filename}")
    print(f"Merged Q-table saved to {base_db_filename}")