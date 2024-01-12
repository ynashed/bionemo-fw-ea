# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pickle
import sqlite3


class HeterographStore:
    """sqlite3 database to store processed complex graphs from model training.
    The database table has ligand, and the complex graph.
    """

    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(
            '''
            CREATE TABLE IF NOT EXISTS heterograph (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ligand TEXT NOT NULL,
                complex_graph TEXT NOT NULL,
                rec_type int NULL
            );


            '''
        )

    def __len__(self):
        return self.conn.execute('SELECT count(*) FROM heterograph').fetchone()[0]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start = 0
            stop = len(self)

            if idx.start:
                start = idx.start
            if idx.stop:
                stop = idx.stop

            records = self.conn.execute(
                'SELECT ligand, complex_graph FROM heterograph LIMIT ?, ?', [start, stop - start]
            ).fetchall()

            complex_graphs = []
            for ligand, complex_graph in records:
                ligand = pickle.loads(ligand)
                complex_graph = pickle.loads(complex_graph)
                if ligand:
                    complex_graph.mol = ligand
                complex_graphs.append(complex_graph)
            return complex_graphs
        else:
            ligand, complex_graph = self.conn.execute(
                'SELECT ligand, complex_graph FROM heterograph LIMIT ?, 1', [idx]
            ).fetchone()
            ligand = pickle.loads(ligand)
            complex_graph = pickle.loads(complex_graph)
            if ligand:
                complex_graph.mol = ligand
            return complex_graph

    def insert(self, lig, complex_graph):
        self.conn.execute(
            'INSERT INTO heterograph(ligand, complex_graph) VALUES(?, ?)',
            [sqlite3.Binary(pickle.dumps(lig)), sqlite3.Binary(pickle.dumps(complex_graph))],
        )

    def commit(self):
        self.conn.commit()
