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
        self.q_table = self.load_q_table()

    def create_connection(self):
        conn = sqlite3.connect(self.db_filename, timeout=60, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def create_table(self):
        with self.conn:
            columns_info = self.conn.execute("PRAGMA table_info(q_table)").fetchall()
            existing_columns = [col[1] for col in columns_info]
            if 'action' not in existing_columns:
                self.conn.execute("DROP TABLE IF EXISTS q_table")
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS q_table (
                state TEXT NOT NULL,
                action TEXT NOT NULL,
                q_value REAL,
                UNIQUE(state, action) ON CONFLICT REPLACE
            )
            """)

    def load_q_table(self):
        cursor = self.conn.execute("SELECT * FROM q_table")
        rows = cursor.fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=['state', 'action', 'q_value'])
            df = df.pivot(index='state', columns='action', values='q_value').fillna(0)
            return df
        else:
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
    def merge_q_tables(num_workers, db_filename, db_dir=None):
        merged_q_table = None
        count_table = None

        # Merge each worker's persistent database.
        for worker_id in range(num_workers):
            worker_file = f"q_table_worker_{worker_id}.db"
            if db_dir:
                worker_db_filename = os.path.join(db_dir, worker_file)
            else:
                worker_db_filename = worker_file
            if os.path.exists(worker_db_filename):
                print(f"Found worker DB for worker {worker_id}: {worker_db_filename}")
                conn = sqlite3.connect(worker_db_filename)
                worker_q_table = pd.read_sql_query("SELECT * FROM q_table", conn, index_col="state")
                conn.close()
                if merged_q_table is None:
                    merged_q_table = worker_q_table
                    count_table = worker_q_table.notnull().astype(int)
                else:
                    merged_q_table = merged_q_table.add(worker_q_table, fill_value=0)
                    count_table = count_table.add(worker_q_table.notnull().astype(int), fill_value=0)
            else:
                print(f"No DB found for worker {worker_id} at {worker_db_filename}")

        # If a shared DB already exists, load and merge it too.
        if db_dir:
            base_db_filename = os.path.join(db_dir, db_filename)
        else:
            base_db_filename = db_filename
        if os.path.exists(base_db_filename):
            print(f"Found base DB: {base_db_filename}")
            conn = sqlite3.connect(base_db_filename)
            base_q_table = pd.read_sql_query("SELECT * FROM q_table", conn, index_col="state")
            conn.close()
            if merged_q_table is None:
                merged_q_table = base_q_table
                count_table = base_q_table.notnull().astype(int)
            else:
                merged_q_table = merged_q_table.add(base_q_table, fill_value=0)
                count_table = count_table.add(base_q_table.notnull().astype(int), fill_value=0)
        else:
            print(f"No base DB found at {base_db_filename}")

        if merged_q_table is not None:
            merged_q_table = merged_q_table.div(count_table)
            conn = sqlite3.connect(base_db_filename)
            merged_q_table.reset_index().to_sql('q_table', conn, if_exists='replace', index=False)
            conn.close()
            print(f"Merged Q-table saved to {base_db_filename}")

            # Cleanup worker files after successful merge.
            for worker_id in range(num_workers):
                worker_file = f"q_table_worker_{worker_id}.db"
                if db_dir:
                    worker_db_filename = os.path.join(db_dir, worker_file)
                else:
                    worker_db_filename = worker_file
                if os.path.exists(worker_db_filename):
                    os.remove(worker_db_filename)
                    print(f"Removed temporary file: {worker_db_filename}")
        else:
            print("No Q-table data to merge.")