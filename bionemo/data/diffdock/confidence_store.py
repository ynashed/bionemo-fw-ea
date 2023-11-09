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

import pickle
import sqlite3


class ConfidenceStore:
    """sqlite3 database to store ligand poses generated from reverse diffusion for confidence training.
    The database table has ligand names, ligand postions, and the RMSDs w.r.t the reference ligands
    """

    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS ligand_confidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ligand TEXT NOT NULL,
                ligand_position TEXT NOT NULL,
                rmsd TEXT NOT NULL
            );
            """
        )

    def __del__(self):
        self.conn.close()

    def __len__(self):
        return self.conn.execute("SELECT count(*) FROM ligand_confidence").fetchone()[0]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start = 0
            stop = len(self)
            if idx.start:
                start = idx.start
            if idx.stop:
                stop = idx.stop
            records = self.conn.execute(
                "SELECT ligand, ligand_position, rmsd FROM ligand_confidence LIMIT ?, ?",
                [start, stop - start],
            ).fetchall()
            ligand_positions = []
            ligand_names = []
            ligand_rmsds = []
            for ligand, ligand_position, rmsd in records:
                ligand = pickle.loads(ligand)
                ligand_position = pickle.loads(ligand_position)
                ligand_rmsd = pickle.loads(rmsd)
                ligand_names.append(ligand)
                ligand_positions.append(ligand_position)
                ligand_rmsds.append(ligand_rmsd)
            return ligand_names, ligand_positions, ligand_rmsds
        else:
            ligand, ligand_position, ligand_rmsd = self.conn.execute(
                "SELECT ligand, ligand_position, rmsd FROM ligand_confidence LIMIT ?, 1",
                [idx],
            ).fetchone()
            lig = pickle.loads(ligand)
            ligand_position = pickle.loads(ligand_position)
            rmsd = pickle.loads(ligand_rmsd)
            return lig, ligand_position, rmsd

    def insert(self, lig, ligand_position, rmsd):
        data = (
            sqlite3.Binary(pickle.dumps(lig)),
            sqlite3.Binary(pickle.dumps(ligand_position)),
            sqlite3.Binary(pickle.dumps(rmsd)),
        )
        self.conn.execute(
            "INSERT INTO ligand_confidence(ligand, ligand_position, rmsd) VALUES(?, ?, ?)",
            data,
        )

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()
