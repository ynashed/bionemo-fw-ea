# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sqlite3


class EmbeddingStore:
    """sqlite3 database to store protein embedding from ESM2 model.
    The database table has name of pdb id and chain id, and the protein embedding
    """

    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(
            '''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seq TEXT NOT NULL,
                embedding TEXT NOT NULL,
                UNIQUE(seq)
            );
            '''
        )

    def __len__(self):
        return self.conn.execute('SELECT count(*) FROM embeddings ').fetchone()[0]

    def search(self, key):
        return self.conn.execute(f"SELECT seq, embedding FROM embeddings WHERE seq like '{key}%'").fetchall()

    def __getitem__(self, key):
        return self.conn.execute('SELECT embedding FROM embeddings WHERE seq = ?', [key]).fetchone()

    def insert(self, key, value):
        try:
            self.conn.execute('INSERT INTO embeddings(seq, embedding) VALUES(?, ?)', [key, sqlite3.Binary(value)])
        except sqlite3.IntegrityError:
            pass

    def commit(self):
        self.conn.commit()

    @DeprecationWarning
    def __delitem__(self, key):
        embedding = self.conn.execute('SELECT seq FROM embeddings WHERE seq = ?', [key]).fetchone()
        self.conn.execute('DELETE FROM embeddings WHERE seq = ?', [key])
        return embedding

    def keys(self):
        return self.conn.execute('SELECT seq FROM embeddings').fetchall()
