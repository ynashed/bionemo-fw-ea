# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import DataLoader
from tqdm import tqdm

from bionemo.model.molecule.moco.data.data_utils import full_atom_encoder, mol_to_torch_geometric
from bionemo.model.molecule.moco.data.molecule_dataset import full_atom_decoder
from bionemo.model.molecule.moco.metrics.metrics import (
    BasicMolecularMetrics,
    get_molecules,
)
from bionemo.model.molecule.moco.models.module import Graph3DInterpolantModel


def calc_performance_stats(rmsd_array):
    coverage_recall = np.mean(rmsd_array.min(axis=1, keepdims=True) < threshold, axis=0)
    amr_recall = rmsd_array.min(axis=1).mean()
    coverage_precision = np.mean(rmsd_array.min(axis=0, keepdims=True) < np.expand_dims(threshold, 1), axis=1)
    amr_precision = rmsd_array.min(axis=0).mean()

    return coverage_recall, amr_recall, coverage_precision, amr_precision


def clean_confs(smi, confs):
    good_ids = []
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi, sanitize=False), isomericSmiles=False)
    for i, c in enumerate(confs):
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c, sanitize=False), isomericSmiles=False)
        if conf_smi == smi:
            good_ids.append(i)
    return [confs[i] for i in good_ids]


def run_benchmark(model_preds, test_data, true_mols, outpath, n_workers=10, dataset="drugs", only_alignmol=False):
    rdkit_smiles = test_data.smiles.values
    corrected_smiles = test_data.corrected_smiles.values

    num_failures = 0
    results = {}
    jobs = []

    filter_mols = None

    for smi, corrected_smi in tqdm(zip(rdkit_smiles, corrected_smiles)):
        if filter_mols is not None and corrected_smi not in filter_mols:
            continue

        if dataset == 'xl':
            smi = corrected_smi

        if corrected_smi not in model_preds:
            print('model failure', corrected_smi)
            num_failures += 1
            continue

        true_mols[smi] = true_confs = clean_confs(corrected_smi, true_mols[smi])

        if len(true_confs) == 0:
            print(f'poor ground truth conformers: {corrected_smi}')
            continue

        n_true = len(true_confs)
        n_model = len(model_preds[corrected_smi])
        results[(smi, corrected_smi)] = {
            'n_true': n_true,
            'n_model': n_model,
            'rmsd': np.nan * np.ones((n_true, n_model)),
        }
        for i_true in range(n_true):
            jobs.append((smi, corrected_smi, i_true))

    def worker_fn(job):
        smi, correct_smi, i_true = job
        true_confs = true_mols[smi]
        model_confs = model_preds[correct_smi]
        tc = true_confs[i_true]

        rmsds = []
        for mc in model_confs:
            try:
                if only_alignmol:
                    rmsd = AllChem.AlignMol(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
                else:
                    rmsd = AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
                rmsds.append(rmsd)
            except Exception as e:
                print(e)
                print('Additional failure', smi, correct_smi)
                rmsds = [np.nan] * len(model_confs)
                break
        return smi, correct_smi, i_true, rmsds

    def populate_results(res):
        smi, correct_smi, i_true, rmsds = res
        results[(smi, correct_smi)]['rmsd'][i_true] = rmsds

    random.shuffle(jobs)
    if n_workers > 1:
        p = Pool(n_workers)
        map_fn = p.imap_unordered
        p.__enter__()
    else:
        map_fn = map

    for res in tqdm(map_fn(worker_fn, jobs), total=len(jobs)):
        populate_results(res)

    if n_workers > 1:
        p.__exit__(None, None, None)

    stats = []
    for res in results.values():
        stats_ = calc_performance_stats(res['rmsd'])
        cr, mr, cp, mp = stats_
        stats.append(stats_)
    coverage_recall, amr_recall, coverage_precision, amr_precision = zip(*stats)

    for i, thresh in enumerate(threshold_ranges):
        print('threshold', thresh)
        coverage_recall_vals = [stat[i] for stat in coverage_recall] + [0] * num_failures
        coverage_precision_vals = [stat[i] for stat in coverage_precision] + [0] * num_failures
        print(
            f'Recall Coverage: Mean = {np.mean(coverage_recall_vals) * 100:.2f}, Median = {np.median(coverage_recall_vals) * 100:.2f}'
        )
        print(f'Recall AMR: Mean = {np.nanmean(amr_recall):.4f}, Median = {np.nanmedian(amr_recall):.4f}')
        print(
            f'Precision Coverage: Mean = {np.mean(coverage_precision_vals) * 100:.2f}, Median = {np.median(coverage_precision_vals) * 100:.2f}'
        )
        print(f'Precision AMR: Mean = {np.nanmean(amr_precision):.4f}, Median = {np.nanmedian(amr_precision):.4f}')

    print(
        len(results),
        'conformer sets compared',
        num_failures,
        'model failures',
        np.isnan(amr_recall).sum(),
        'additional failures',
    )
    with open(outpath, 'wb') as f:
        pickle.dump(stats, f)


if __name__ == "__main__":
    dataset = 'DRUGS'  # [DRUGS, XL]
    test_csv = f"/workspace/bionemo/data/GEOM/torsional_diffusion/{dataset}/test_smiles.csv"
    true_mols = f"/workspace/bionemo/data/GEOM/torsional_diffusion/{dataset}/test_mols.pkl"
    test_data = pd.read_csv(test_csv)  # this should include the corrected smiles
    with open(true_mols, 'rb') as f:
        true_mols = pickle.load(f)

    # test_data =  enumerate(test_data) #tqdm(enumerate(test_data), total=len(test_data))
    test_mols = []  # [(Chem.MolFromSmiles(smi), count) for (raw, count, smi) in test_data.iloc[1:].iterrows()]
    for index, row in test_data.iterrows():
        smiles = row['smiles']
        n_conformers = row['n_conformers']
        corrected_smiles = row['corrected_smiles']
        if dataset == 'XL':
            smiles = corrected_smiles
        # print(f"Index: {index}, SMILES: {smiles}, n_conformers: {n_conformers}, Corrected SMILES: {corrected_smiles}")
        test_mols.append((smiles, n_conformers, Chem.MolFromSmiles(corrected_smiles)))

    input_data = {
        smiles: mol_to_torch_geometric(mol[0], full_atom_encoder, smiles) for (smiles, mol) in true_mols.items()
    }

    threshold = threshold_ranges = np.arange(0, 2.5, 0.125)

    best_ckpt_path = "/workspace/bionemo/bionemo/model/molecule/moco/ckpt/julian_best_mol_stab.ckpt"
    ckpt_path = "/workspace/bionemo/bionemo/model/molecule/moco/models/results/eqgatdiff/EQGAT_FW_TEST_6_28_charges_live_interpolant/checkpoints/last.ckpt"
    save_path = '/workspace/bionemo/bionemo/model/molecule/moco/models/results/eqgatdiff/julian/'  #! set up saving for infernece and forward

    state_dict = torch.load(best_ckpt_path)["state_dict"]
    state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
    model = Graph3DInterpolantModel.load_from_checkpoint(ckpt_path)
    model.dynamics.load_state_dict(state_dict)
    mol_metrics = BasicMolecularMetrics({"atom_decoder": full_atom_decoder}, device=model.device)
    model.cuda()
    model.eval()

    if not os.path.exists(save_path + f"td_{dataset}_confs_final.pkl"):
        OUTPUT = defaultdict(list)
        batch_size = 50  # drugs 50 is like 50% gpu #XL 25 # XL 50 too big 10 is 25%
        # import ipdb; ipdb.set_trace()
        for smi, n_conf, mol in test_mols:
            total = 2 * n_conf
            data = input_data[smi]
            loader = DataLoader([data] * total, batch_size=batch_size, shuffle=False)
            for batch in loader:
                batch = batch.to(model.device)
                batch.h = batch.x
                batch.x = batch.pos
                batch.pos = None
                out = model.conditional_sample(batch, conditional_variables=['h', 'edge_attr', 'charges'])
                mols = get_molecules(out, {"atom_decoder": full_atom_decoder})
                stab_dict, valid_dict, stat_dict, valid_smi, stable_mols, valid_mols = mol_metrics(mols)
                res = {**stab_dict, **valid_dict, **stat_dict}
                print(smi, res)
                OUTPUT[smi].extend([x.rdkit_mol for x in mols])
                with open(save_path + f"td_{dataset}_confs.pkl", 'wb') as f:
                    pickle.dump(OUTPUT, f)
        os.remove(save_path + f"td_{dataset}_confs.pkl")
        with open(save_path + f"td_{dataset}_confs_final.pkl", 'wb') as f:
            pickle.dump(OUTPUT, f)
    else:
        with open(save_path + f"td_{dataset}_confs_final.pkl", 'rb') as f:
            OUTPUT = pickle.load(f)
    result = run_benchmark(OUTPUT, test_data, true_mols, "results.pkl")
