# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

alphabet = 'ARNDCQEGHILKMFPSTWYV'

cluster_mapping = {}
with open('Fake50.fasta', 'w') as fd:
    for length in range(50, 1000, 50):
        for base in alphabet:
            fd.write(f">UniRef50_{base}_{length}\n")
            fd.write(base * length)
            fd.write("\n")
            cluster_mapping[f"UniRef50_{base}_{length}"] = []


with open('Fake90.fasta', 'w') as fd:
    num_clusters = 10
    for length in range(50, 1000, 50):
        for base in alphabet:
            for i in range(num_clusters):
                fd.write(f">UniRef90_{base}_{length}-{i}\n")
                fd.write(base * (length + i))
                fd.write("\n")
                cluster_mapping[f"UniRef50_{base}_{length}"].append(f"UniRef90_{base}_{length}-{i}")
            num_clusters += 1


with open("mapping.tsv", 'w') as fd:
    fd.write("dumbheader\n")
    for key, values in cluster_mapping.items():
        str_ = key + "\t" + ",".join(values)
        fd.write(f"{str_}\n")
