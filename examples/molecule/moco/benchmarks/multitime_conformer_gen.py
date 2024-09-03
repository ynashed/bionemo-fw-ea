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
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import DataLoader
from tqdm import tqdm

from bionemo.model.molecule.moco.data.data_utils import full_atom_encoder, mol_to_torch_geometric
from bionemo.model.molecule.moco.data.molecule_dataset import full_atom_decoder
from bionemo.model.molecule.moco.metrics.molecule import get_molecules
from bionemo.model.molecule.moco.metrics.molecule_stability_2d import Molecule2DStability
from bionemo.model.molecule.moco.models.multitime_module import (
    MultiTimeGraph3DInterpolantModel as Graph3DInterpolantModel,
)


def calc_performance_stats(rmsd_array, threshold):
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


def worker_fn(job):
    smi, correct_smi, i_true, true_mols, model_preds, only_alignmol = job
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


def populate_results(res, results):
    smi, correct_smi, i_true, rmsds = res
    results[(smi, correct_smi)]['rmsd'][i_true] = rmsds


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
            jobs.append((smi, corrected_smi, i_true, true_mols, model_preds, only_alignmol))

    random.shuffle(jobs)
    if n_workers > 1:
        with Pool(n_workers) as pool:
            for res in tqdm(pool.imap_unordered(worker_fn, jobs), total=len(jobs)):
                populate_results(res, results)
    else:
        for res in tqdm(map(worker_fn, jobs), total=len(jobs)):
            populate_results(res, results)

    stats = []
    threshold = np.arange(0, 2.5, 0.125)
    for res in results.values():
        stats_ = calc_performance_stats(res['rmsd'], threshold)
        cr, mr, cp, mp = stats_
        stats.append(stats_)
    coverage_recall, amr_recall, coverage_precision, amr_precision = zip(*stats)

    for i, thresh in enumerate(threshold):
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
    clean = [
        'C=CCC(CC=C)(NC(=O)c1cccc(C)c1)c1ccccc1',
        'C=CCN(c1ccccc1C(=O)O)S(=O)(=O)/C=C/c1ccccc1',
        'C=CCSc1nnc(Cn2c(=O)sc3ccccc32)n1C',
        'CC(=NCCc1ccccc1)C1=C(O)CSC1=O',
        'CC(=O)N1CCCN(Cc2cn(-c3cccc(C)c3)nc2-c2cccc(F)c2)CC1',
        'CC(=O)N1CC[NH+](CC(=O)Nc2c(Cl)cccc2Cl)CC1',
        'CC(=O)N1N=C(c2ccc(C)o2)CC1c1ccc(Br)cc1',
        'CC(=O)c1ccc(NC(=O)C2(c3ccccc3)CCOCC2)cc1',
        'CC(=O)c1ccc(S(=O)(=O)N(CCCO)Cc2cc3cc(C)ccc3[nH]c2=O)cc1',
        'CC(C)(C)C(=O)OCC(=O)[C@H]1CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@H]3CC[C@]12C',
        'CC(C)(C)OC(=O)CNC(=O)c1[nH]cnc1C(=O)N[C@@H](Cc1ccccc1)C(=O)OCc1ccccc1',
        'CC(C)C(C(=O)O)N1CCS(=O)(=O)CC1',
        'CC(C)CCc1c(O)nc(SCC(C)C)[nH]c1=O',
        'CC(C)c1nc(SCC(=O)c2ccccc2)c2c(=O)n(C)c(=O)n(C)c2n1',
        'CC(CNS(=O)(=O)c1ccc(Br)cc1)CN(C)C',
        'CC1(C)CC(=O)C2=C(C1)N=C1SCCC(=O)N1C2c1ccccc1',
        'CC1(C)CC(C(CCN(Cc2ccccc2)C(=O)c2ccco2)c2ccc(F)cc2)CCO1',
        'CC1=NN(c2ccccc2)C(=O)C1C(=O)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1',
        'CC1CC(C)(C)N(C(=O)CSc2ccc(-c3ccc(F)cc3)nn2)c2ccccc21',
        'CC1CN(CCS(=O)(=O)c2cc(Cl)ccc2Cl)CC(C)O1',
        'CCC(=O)c1ccccc1OCC[NH+]1CCN(Cc2ccccc2)CC1',
        'CCC(C)N(CCNC(=O)CC1Oc2ccccc2NC1=O)C1CCCCC1',
        'CCC(C)NC(=O)Cn1ccc2cc(S(=O)(=O)N3CCCCCC3)ccc21',
        'CCC(Sc1nc(C)cc(C)n1)C(=O)Nc1ccc(Cl)cn1',
        'CCCCC(CCC(=O)O)(C(=O)OCC)c1csc([NH2+]c2ccc(OCC)cc2)n1',
        'CCCCCN(C)CCC(O)(P(=O)([O-])O)P(=O)(O)O',
        'CCCCN1CCCC1CNC(=O)c1cc(C#N)c2ccccc2c1OC',
        'CCCCOC(=O)c1nnc(-c2ccc(Cl)cc2)o1',
        'CCCCOc1ccc(C(=O)Nc2ccc(S(=O)(=O)Nc3onc(C)c3C)cc2)cc1',
        'CCCN1CCN(CCCNC(=O)CN2C(=O)C(CC)Oc3ccccc32)CC1',
        'CCCNC(=O)C(=O)Nc1cc2c3c(c1)CCC(=O)N3CCC2',
        'CCCS(=O)(=O)[NH+]1CCN(S(=O)(=O)c2ccccc2)CC1',
        'CCCc1cc(=O)n2nc(N3CCC(C(=O)NCc4ccc(C)o4)CC3)sc2n1',
        'CCN(CC)Cc1c(O)ccc2c(-c3ccccc3)cc(=O)oc12',
        'CCN(CC)S(=O)(=O)c1cc(C(=O)Nc2cccnc2)ccc1Cl',
        'CCN(CC)S(=O)(=O)c1ccc2oc(SCC3CN=C(c4ccc(OC)cc4)S3)nc2c1',
        'CCN(CC)S(=O)(=O)c1cccc(NC(=O)Nc2cccc(C(F)(F)F)c2)c1',
        'CCN1CCCC1CNC(=O)c1[nH]c2ccc(OC)cc2c1C',
        'CCOC(=O)C(=O)NC(C)CCc1ccccc1',
        'CCOC(=O)C1=C(O)c2ccccc2C(=O)/C1=C\\c1ccc(CC)cc1',
        'CCOC(=O)N1CCC(NC(=O)CN2C(=O)CSc3ncccc32)CC1',
        'CCOC(=O)c1c(-c2ccccc2)nnn1-c1nc(OCC)nc(N2CCCCC2)n1',
        'CCOC(=O)c1c(C)[nH]c(C)c1C(=O)CSc1nc(N)cc(=O)[nH]1',
        'CCOC(=O)c1cc2occc2n1CC(=O)Nc1ccc(OC)cc1OC',
        'CCOC(=O)c1ccc(NC(=O)C2CC(=O)Nc3nc4ccccc4n32)cc1',
        'CCOC(=O)c1ccc(Nc2ncnc3c2cnn3C)cc1',
        'CCOCc1nnc(NC(=O)CCS(=O)(=O)Cc2ccccc2)s1',
        'CCOP(=O)(OCC)c1nc(/C=C/c2ccccc2)oc1NCc1ccccc1',
        'CCOc1cc(C)ccc1OCc1nnc(SCC(=O)c2cccc([N+](=O)[O-])c2)n1C',
        'CCOc1ccc(Br)cc1S(=O)(=O)Nc1cccnc1',
        'CCOc1ccc(C2C(C#N)=C(N)Oc3cc(C)n(CCN4CCOCC4)c(=O)c32)cc1OCC',
        'CCOc1ccc(Cl)cc1CSc1n[nH]c(C)n1',
        'CCOc1ccc(NC(=O)C[n+]2ccccc2)cc1',
        'CCOc1cccc(=O)c2c(C)n(Cc3ccco3)c(C)c12',
        'CCOc1cccc(C(=O)NC(=S)Nc2cccc(NC(=O)c3ccccc3Cl)c2)c1',
        'CCc1ccc2c(COC(=O)c3n[nH]c4ccccc34)cc(=O)oc2c1',
        'CCc1ccccc1NS(=O)(=O)c1cccc2nsnc12',
        'CCn1c(O)c(C=NCc2ccccc2)c(=O)[nH]c1=S',
        'CCn1c(SCc2nc3ccccc3c(=O)[nH]2)nnc1C(C)NC(=O)c1ccccc1',
        'CCn1nc(CC(C)C)cc1C(=O)N1CCCC(Nc2ccc(F)cc2)C1',
        'CN(C)C(=O)CSC1c2ccccc2C(=O)N1c1ccccc1',
        'CN(C)CC(=O)N1c2ccccc2CCc2ccc(NC(=O)c3ccc([NH3+])cc3)cc21',
        'CN(C)/C=C/C(=O)c1ccc(-c2ccc(Cl)cc2Cl)o1',
        'CN(C)c1nc(Oc2ccc(=O)[nH]n2)nc(N2CCCCC2)n1',
        'CN(CC(=O)Nc1nc2ccccc2s1)S(=O)(=O)c1ccc(Cl)cc1',
        'CN1CCN(Cc2nc3ccccc3c(=O)n2Cc2nc(-c3ccc([N+](=O)[O-])cc3)cs2)CC1',
        'CN1N=C2c3ccccc3CCCC2C1c1ccc(F)cc1',
        'CNC(=O)C(=O)NCC1OCCN1S(=O)(=O)c1ccc(C)cc1',
        'COC(=O)C(C)Sc1ccc2nnc(CCNS(=O)(=O)c3ccc(C)cc3)n2n1',
        'COC(=O)C1C2C=CC3(CN(C(C)c4ccc(OC)cc4)C(=O)C13)O2',
        'COC(=O)CC1SC(=O)NC1=Nc1ccc(C)cc1',
        'COC(=O)CCN1CCC(c2cc(-c3cccc(C(F)(F)F)c3)n[nH]2)CC1',
        'COC(=O)Nc1ccc2c(c1)N(C(=O)CC[NH+]1CCOCC1)c1ccccc1S2',
        'COC(=O)c1c(-c2cccs2)csc1NC(=O)c1cccnc1',
        'COC(=O)c1c(NC(=O)CN2CCC3(O)CCCCC3C2c2cc(OC)ccc2OC)sc(C)c1C',
        'COC(=O)c1ccccc1S(=O)(=O)N1CCC(NC(=O)C(Cc2cccc(OC)c2OC)NC(C)=O)CC1',
        'COCC(C)n1c(C)cc(C(=O)CSc2nnc(N)s2)c1C',
        'COc1cc(C[NH2+]Cc2ccccn2)ccc1OCC(=O)NC1CCCCC1',
        'COc1cc(N2CCCC2)ccc1[N+](=O)[O-]',
        'COc1cc(NC(=O)CCc2ccccc2)nc(OC)n1',
        'COc1cc(NC(=O)c2cccnc2SC)cc(OC)c1',
        'COc1cc(OC)cc(N2CCN(S(=O)(=O)c3ccc(-n4cnnn4)cc3)CC2)c1',
        'COc1cc(/C=C/C(=O)c2ccc(F)cc2)cc(OC)c1OC',
        'COc1cc(/C=N\\NC(=O)OC(C)(C)C)ccc1OC(=O)c1ccco1',
        'COc1cc(/C=N/O)ccc1OS(=O)(=O)c1ccc(C)cc1',
        'COc1cc2c(OC(C)=O)c3c(c(-c4ccc5c(c4)OCO5)c2cc1OC)C(=O)OC3',
        'COc1cc2c(cc1OC)C(C)N(C(=O)c1c(-c3ccccc3)noc1C)CC2',
        'COc1cc2c(cc1OC)CN(C(=O)[C@@H](NCc1ccncc1)C(C)(C)C)CC2',
        'COc1cc2ncnc(NCc3ccccc3)c2cc1OC',
        'COc1ccc(-c2cnc3n2CCCCC3)cc1',
        'COc1ccc(C(CCNCc2ccc(OC)c(OC)c2)CCC(C)C)cc1',
        'COc1ccc(C2(O)CCN(CC(=O)Nc3nccs3)CC2)c(OC)c1',
        'COc1ccc(CC(=O)NC(C)c2nnc(SCC(=O)Nc3ccccc3)n2C)cc1',
        'COc1ccc(CCNC(=O)c2ccc(C)s2)cc1OC',
        'COc1ccc(CCNC(=O)c2ccc(CNc3nc(=S)[nH]c4ccccc34)cc2)cc1OC',
        'COc1ccc(F)cc1C(=O)C1CCCN(C(=O)c2cccc(N(C)C)c2)C1',
        'COc1ccc(N(C)C(=O)CCS(=O)(=O)c2cccc3nonc23)c(OC)c1',
        'COc1ccc(N(CC(=O)NC2CCCCCC2)S(=O)(=O)c2ccccc2)cc1Cl',
        'COc1ccc(N2CC(C(=O)Nc3ccccc3C(=O)NCc3ccco3)CC2=O)cc1',
        'COc1ccc(N2CC(NC(=O)N3CCN(c4ccccc4)CC3)CC2=O)cc1',
        'COc1ccc(N2CCN(S(=O)(=O)c3ccc(Br)s3)CC2)cc1',
        'COc1ccc(OC)c(C2=NOC(C(=O)Nc3cc(C)nn3-c3ccccc3)C2)c1',
        'COc1ccc(OC)c(NC(=O)CSc2nnc3ccc(-c4ccncc4)nn23)c1',
        'COc1ccc(S(=O)(=O)N2CCN(c3ccccn3)CC2)c(OC)c1',
        'COc1ccc(/C=N/Nc2nc(-c3ccc(C)cc3)c(NC(C)=O)s2)cc1OC',
        'COc1ccc2nc(NC(=O)C3CCCN(S(=O)(=O)c4cccs4)C3)sc2c1',
        'COc1cccc(C(=O)NCCNC(=O)c2ccc(C)nc2)c1',
        'COc1cccc(C2C(C#N)=C(N)Oc3c2oc(CO)cc3=O)c1',
        'COc1cccc2c1OC1(C)CC2NC(=O)N1',
        'COc1ccccc1NC(=O)COc1ccccc1C(=O)c1cnn(-c2ccccc2)c1',
        'COc1ccccc1NS(=O)(=O)c1ccc2nc(CCl)[nH]c2c1',
        'COc1ccccc1[NH+]1CCN(CC(O)COc2cccc3ccccc23)CC1',
        'CP(=O)(Nc1ccc(SC(F)(F)F)cc1)Oc1ccccc1',
        'CSc1nc2ccc(NC(=O)C3CC(=O)N(Cc4ccco4)C3)cc2s1',
        'C[C@H](NC(=O)[C@H]1N2C(=O)c3ccccc3C2SC1(C)C)C(=O)N1CCC[C@H]1C(=O)O',
        'C/C(=N\\Nc1ccccn1)c1ccc(C(C)C)cc1',
        'Cc1cc(C(=O)Cn2nc(C(=O)O)c3ccccc3c2=O)c(C)n1CC(F)(F)F',
        'Cc1cc(C)cc(NC(=O)c2ccc(NC(=O)C3C4CC5OC(=O)C3C5C4)cc2)c1',
        'Cc1cc(C)n(C(=O)c2ccc(C)o2)n1',
        'Cc1cc(C)n(CC(=O)N2CCCC(N3CCN(c4ccccc4C)CC3)C2)n1',
        'Cc1cc(C)n(CCc2nc(-c3cc4ccccc4o3)cs2)n1',
        'Cc1ccc(-c2csc3ncnc(SCC(=O)c4ccc5c(c4)OCCO5)c23)cc1',
        'Cc1ccc(-n2c(N)c(-c3nc4ccccc4n3C)sc2=S)cc1',
        'Cc1ccc(C(=O)Nc2cccc(C(=O)N/N=C/c3ccccc3F)c2)cc1',
        'Cc1ccc(C)c(NC(=S)NCCCN2CCOCC2)c1',
        'Cc1ccc(C)n1-c1ccccc1N1CCCCC1',
        'Cc1ccc(CNC(=O)CCS(=O)(=O)c2cc3c(cc2C)NC(=O)CO3)cc1',
        'Cc1ccc(N)c2c1OCCC2=O',
        'Cc1ccc(N2C(=O)CC(NNc3nc4ccccc4s3)C2=O)cc1',
        'Cc1ccc(N2CCN(S(=O)(=O)c3ccc4c(c3)n(C)c(=O)c(=O)n4C)CC2C)cc1',
        'Cc1ccc(NC(=O)COC(=O)c2ccccc2-c2nc3ccccc3s2)c(C)c1',
        'Cc1ccc(S(=O)(=O)c2cc3c(=O)n4ccccc4nc3n(C)c2=N)cc1C',
        'Cc1ccc2c(c1)N(CC(=O)NCCCN1CCCC1)C(=O)CO2',
        'Cc1ccc2nc(C)cc(C(=O)OCC(=O)Nc3c(C)n(C)n(-c4ccccc4)c3=O)c2c1',
        'Cc1ccc2oc(=O)cc(CN3CCOCC3)c2c1',
        'Cc1cccc(C(=O)NC2CCN(S(=O)(=O)c3ccccc3)CC2)c1',
        'Cc1cccc(C(=O)n2c(=S)nc3cc(C)ccn32)c1',
        'Cc1cccc(N(C(=O)c2snc(C(N)=O)c2N)C(C(=O)NCC2CCCO2)c2cccs2)c1',
        'Cc1cccc(NC(=O)COC(=O)c2ccc([N+](=O)[O-])o2)c1C',
        'Cc1cccc([NH2+]c2nc(C)nc3c2oc2ccccc23)c1',
        'Cc1ccccc1NC(=O)n1ncc2cc(Cl)ccc21',
        'Cc1cccn2c(=O)c3cc(C(=O)N4CCOCC4)n(C)c3nc12',
        'Cc1ccn2cc(-c3cccc(NC(=O)c4cccs4)c3)nc2c1',
        'Cc1csc(NC(=O)CSc2nnc(-c3ccccc3)o2)n1',
        'Cc1nc2cc(C(=O)OCC(=O)Nc3ccc(C#N)cc3)ccc2n1-c1ccccc1',
        'Cc1nn(-c2ccccc2)c(C)c1/C=N/NC(=O)CSc1nnc(SCc2cccc3ccccc23)s1',
        'Cc1nn(-c2ccccc2)c2[nH]c(C(F)(F)F)cc(=O)c12',
        'Cc1nn2cc(-c3cccc(Cl)c3)cnc2c1Cl',
        'Cc1noc(C(=O)Nc2c(C)nn(Cc3ccccc3)c2C)c1Cl',
        'Cn1c(=O)c2c(Nc3ccc(F)cc3)cc(=O)n(-c3ccccc3)c2n(C)c1=O',
        'Cn1c(C(=O)NCc2cccnc2)cc2sc(Cl)cc21',
        'Cn1c(NC(=O)C[n+]2cc(-c3ccc(Cl)cc3)n3c2CCC3)cc(=O)n(C)c1=O',
        'Cn1cc(C(=O)NS(=O)(=O)Nc2ccc(Br)cc2)c2ccccc21',
        'Cn1cccc1CN1CCN(C(=O)COc2ccccc2)CC1',
        'Cn1nccc1C(=O)Oc1ccc(/C=N/NC(=O)c2ccccc2O)cc1',
        'Fc1ccc(NC(=S)NN=C2CCCCCC2)cc1',
        'N#CC1=C(S)N=C(N)C(c2nc(-c3ccccc3)cs2)C12CCCC2',
        'N#CCSc1nc2ccccc2c(=O)n1CCC(=O)NC1CCCCC1',
        'N#C/C(=C\\c1ccc(N2CCOCC2)o1)C(=O)NC1CCCCC1',
        'N#Cc1ccccc1COc1cccc2c1CCCC2O',
        'NC(=O)N1c2ccccc2CC(=O)c2ccccc21',
        'NC(=O)N=Nc1c2[nH]c3ccccc3nc-2c2ccccc12',
        'N/C(=N\\OC(=O)c1ccccc1Cl)c1cccnc1',
        'Nc1cc(=O)[nH]c(=O)n1CC1CC1',
        'Nc1ccn([C@@H]2C(F)=C(CO)[C@@H](O)[C@H]2O)c(=O)n1',
        'Nc1nn(-c2ccccc2)nc1C(=O)OCc1ccccc1',
        'O=C(C1=C[C@@H](C2CCCCC2)C[C@@H](OCCCCO)O1)N1CCN(Cc2ccccc2)CC1',
        'O=C(CCC(=O)Nc1ccc(F)cc1)N/N=C/c1ccccc1O',
        'O=C(CCNC(=O)c1ccc([N+](=O)[O-])cc1)NCCc1ccccc1',
        'O=C(CNC(=O)c1ccc(-c2ccccc2)cc1)OCc1c(F)cccc1Cl',
        'O=C(CSCC(=O)N1CCCCC1)Nc1nc(-c2ccccc2)c(-c2ccccc2)s1',
        'O=C(CSc1nnc(-c2ccccc2)n1Cc1ccco1)Nc1ccc2c(c1)OCCO2',
        'O=C(Cc1c[nH]c2ccccc12)NNC(=O)c1ccccc1',
        'O=C(NCC(c1ccc(F)cc1)N1CCOCC1)c1cccc(Br)c1',
        'O=C(NCCCN1CCOCC1)c1cc(-c2ccc(O)cc2)on1',
        'O=C(NCc1ccccc1Cl)c1ccc2c(c1)SC1=NS(=O)(=O)CCN12',
        'O=C(NCc1nnc(SCC(=O)N2CCCc3ccccc32)o1)c1c(F)cccc1Cl',
        'O=C(NN=C1CCCCCC1)c1cc2ccccc2cc1O',
        'O=C(NNC(=O)c1ccccc1F)c1cccc(Cl)c1',
        'O=C(Nc1ccc(F)cc1)Nc1nnc(Cc2ccccc2)s1',
        'O=C(Nc1ccccc1N1CCCCC1)c1ccc(F)cc1',
        'O=C(O)CCc1ccc(-c2cccs2)n1NC(=O)c1ccc(O)cc1',
        'O=C(OC[C@@H]1CCC[NH+]2CCCC[C@H]12)c1cc2ccccc2oc1=O',
        'O=C(OCc1ccc(Cl)c(Cl)c1)c1ccc[nH]1',
        'O=C1N(Cc2ccc(F)cc2)CCN1c1ccc(F)cc1',
        'O=C1c2ccccc2C(=O)N1Cc1ccco1',
        'O=S(=O)(Nc1cccnc1Cl)c1ccc2c(c1)OCCO2',
        'O=[n+]1cc(-c2cc3ccccc3o2)n([O-])c2c1CCCC2',
        'O=c1[nH]cnc2c1ncn2Cc1ccc([N+](=O)[O-])cc1',
        'O=c1c(Cl)c(NCCO)cnn1-c1cccc(Cl)c1',
        'O=c1c2ccccc2c2ccccc2c(=O)n1Cc1cccnc1',
        'O=c1oc2ccccc2c(CNc2ccccc2)c1[N+](=O)[O-]',
        'OC(CNCCCCNCC(O)C(F)(F)F)C(F)(F)F',
        'OC(Cn1c2c(c3ccccc31)CCCC2)C[NH+]1CCOCC1',
        'Oc1ccc(CN2CCN(c3ccccc3)CC2)c2cccnc12',
        'S=c1[nH]nc(COc2ccccc2)n1/N=C/C=C/c1ccco1',
        'c1cc2c3c(cccc3c1)C(N1CCOCC1)=N2',
    ]
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

    # best_ckpt_path = "/workspace/bionemo/bionemo/model/molecule/moco/ckpt/julian_best_mol_stab.ckpt"
    # ckpt_path = "/workspace/bionemo/bionemo/model/molecule/moco/models/results/eqgatdiff/EQGAT_FW_TEST_6_28_charges_live_interpolant/checkpoints/last.ckpt"
    ckpt_path = "/workspace/bionemo/examples/molecule/moco/checkpoints/best-epoch=89-step=88025--mol_stable=0.954.ckpt"
    save_path = '/workspace/bionemo/bionemo/model/molecule/moco/models/results/multitime/'  #! set up saving for infernece and forward
    # bionemo/examples/molecule/moco/checkpoints/best-epoch=89-step=88025--mol_stable=0.954.ckpt
    # state_dict = torch.load(best_ckpt_path)["state_dict"]
    # state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
    model = Graph3DInterpolantModel.load_from_checkpoint(ckpt_path)
    # model.dynamics.load_state_dict(state_dict)
    # mol_metrics = BasicMolecularMetrics({"atom_decoder": full_atom_decoder}, device=model.device)
    model.cuda()
    model.eval()
    count = 0
    if not os.path.exists(save_path + f"mt_{dataset}_confs_clean.pkl"):
        OUTPUT = defaultdict(list)
    else:
        with open(save_path + f"mt_{dataset}_confs_clean.pkl", 'rb') as f:
            OUTPUT = pickle.load(f)
        print(OUTPUT.keys())
    batch_size = 100  # drugs 50 is like 50% gpu #XL 25 # XL 50 too big 10 is 25%
    # import ipdb; ipdb.set_trace()
    for smi, n_conf, mol in test_mols:
        if count > 100:
            break
        # else:
        #     count += 1
        if smi not in clean:
            continue
        if smi in OUTPUT:
            continue
        print(f"Count {2*n_conf} SMILES {smi}")
        if len(smi) > 80:
            batch_size = 50
            count -= 1
            continue
        else:
            batch_size = 150
        total = 2 * n_conf
        data = input_data[smi]
        loader = DataLoader([data] * total, batch_size=batch_size, shuffle=False)
        for batch in loader:
            mol_2d_stability = Molecule2DStability({"atom_decoder": full_atom_decoder}, device=model.device)
            batch = batch.to(model.device)
            batch.h = batch.x
            batch.x = batch.pos
            batch.pos = None
            out = model.conditional_sample(batch, conditional_variables=['h', 'edge_attr', 'charges'])
            mols = get_molecules(out, {"atom_decoder": full_atom_decoder})
            stability_res, valid_smiles, valid_molecules, stable_molecules, info_2d = mol_2d_stability(mols)
            print(smi, stability_res)
            OUTPUT[smi].extend([x.rdkit_mol for x in mols])
            with open(save_path + f"mt_{dataset}_confs_clean.pkl", 'wb') as f:
                pickle.dump(OUTPUT, f)
    # os.remove(save_path + f"mt_{dataset}_confs.pkl")
    with open(save_path + f"mt_{dataset}_confs_final.pkl_clean", 'wb') as f:
        pickle.dump(OUTPUT, f)

    result = run_benchmark(OUTPUT, test_data, true_mols, "mt_results_clean.pkl")
#! TODO need to filter out non clean molecules from failters ie pass in a filtermol for coverage to be accurate
