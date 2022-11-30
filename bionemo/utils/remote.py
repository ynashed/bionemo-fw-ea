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

from hashlib import md5
import os
import requests
from abc import ABC, abstractmethod
import pathlib
import tarfile

from dataclasses import dataclass
from typing import Optional, List

from nemo.utils import logging


@dataclass
class RemoteResource(object):
    """ Responsible for downloading remote files, along with optional processing of downloaded files for downstream usecases.

    Each object is invoked through either its constructor (setting up the destination and checksum), or through a pre-configured class method.
    `download_resource()` contains the core functionality, which is to download the file at `url` to the fully qualified filename. Class methods
    can be used to further configure this process.

    Receive:
        a file, its checksum, a destination directory, and a root directory

        Our dataclass then provides some useful things:
            - fully qualified destination folder (property)
            - fully qualified destination file (property)
            - check_exists()
            - download_resource()

        Form the fully qualified destination folder.
        Create a fully qualified path for the file

        (all lives in the download routine)
        Check that the fq destination folder exists, otherwise create it
        Download the file.
        Checksum the download.
        Done.

        Postprocessing should be their own method with their own configuration.

    Example usage:
        >>> # The following will download and preprocess the prepackaged resources.
        >>> GRCh38Ensembl99ResourcePreparer().prepare()
        >>> Hg38chromResourcePreparer().prepare()
        >>> GRCh38p13_ResourcePreparer().prepare()


    Attributes:
        dest_directory: The directory to place the desired file upon completing the download. Should have the form {dest_directory}/{dest_filename}
        dest_filename: The desired name for the file upon completing the download.
        checksum: checksum associated with the file located at url. If set to None, check_exists only checks for the existance of `{dest_directory}/{dest_filename}`
        url: URL of the file to download
        root_directory: the bottom-level directory, the fully qualified path is formed by joining root_directory, dest_directory, and dest_filename.
    """

    checksum: Optional[str]
    dest_filename: str
    dest_directory: str

    root_directory: str = "/"  # NOTE, to use a tmpdir, this should be invoked at the toplevel. We could _perhaps_ offer a utility method for getting the tmpdir
    url: Optional[str] = None

    @property
    def fully_qualified_dest_folder(self):
        return os.path.join(self.root_directory, self.dest_directory)

    @property
    def fully_qualified_dest_filename(self):
        """Returns the fully qualified destination path of the file.

        Example:
            /tmp/my_folder/file.tar.gz
        """
        return os.path.join(self.fully_qualified_dest_folder, self.dest_filename)

    def exists_or_create_destination_directory(self, exist_ok=True):
        """ Checks that the `fully_qualified_destination_directory` exists, if it does not, the directory is created (or fails).

        exists_ok: Triest to create `fully_qualified_dest_folder` if it doesnt already exist.
        """
        os.makedirs(self.fully_qualified_dest_folder, exist_ok=exist_ok)

    @staticmethod
    def get_env_tmpdir():
        """ Convenience method that exposes the environment TMPDIR variable. """
        return os.environ.get("TMPDIR")

    def download_resource(self, overwrite=False) -> str:
        """ Downloads the resource to its specified fully_qualified_dest name.

        Returns: the fully qualified destination filename.
        """
        self.exists_or_create_destination_directory()

        if not self.check_exists() or overwrite:
            with requests.get(self.url, stream=True) as r, open(
                self.fully_qualified_dest_filename, "wb"
            ) as fd:
                r.raise_for_status()
                for bytes in r:
                    fd.write(bytes)

        self.check_exists()
        return self.fully_qualified_dest_filename

    def check_exists(self):
        """ returns true if `fully_qualified_dest_filename` exists and the checksum matches `self.checksum` """
        if os.path.exists(self.fully_qualified_dest_filename):
            with open(self.fully_qualified_dest_filename, "rb") as fd:
                data = fd.read()
                result = md5(data).hexdigest()
            if matches := (self.checksum is None):
                logging.info(
                    f"No checksum provided, filename exists. Assuming it is complete."
                )
            elif matches := (result == self.checksum):
                logging.info(f"Resource exists, checksum: {matches=}")
            return matches
        return False


@dataclass
class ResourcePreparer(ABC):
    """ Interface defining a ResourcePreparer. Implementors promise to provide both a complete RemoteResource and a freeform
    preprocess method. This interface can be used to generically define a workflow from a config file.

        remote -> prepare -> prepared data.
    """

    root_directory: Optional[str] = RemoteResource.get_env_tmpdir()

    @abstractmethod
    def get_remote_resources(self) -> List[RemoteResource]:
        """ Gets the remote resources associated with this preparor. """
        raise NotImplementedError

    @abstractmethod
    def prepare(self) -> List:
        """ Returns a list of prepared filenames. """
        raise NotImplementedError


@dataclass
class GRCh38p13_ResourcePreparer(ResourcePreparer):
    """ ResourcePreparer for the human genome assembly produced by encode.
    GRCh38.p13, all primary chromosomes are downloaded. Since this resource does not create an archive for contigs,
    we must create a remote resource for each file. Preprocessing therefore requires working on sets of files.
    """
    dest_dir: Optional[str] = "GRCh38.p13"

    def get_remote_resources(self) -> List[RemoteResource]:
        basename = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GCF_000001405.39_GRCh38.p13_assembly_structure/Primary_Assembly/assembled_chromosomes/FASTA/"
        resources = list()

        checksums = {
            "chr1.fna.gz": "43aab6c472a690d9c29c62fb0b47a3f6",
            "chr2.fna.gz": "9e54b2ba05e6866a72b9971a7354353b",
            "chr3.fna.gz": "41931af8120dd459c738f1baec318e6f",
            "chr4.fna.gz": "2c54d3c6ba306720991f739926c6f21e",
            "chr5.fna.gz": "8e0d2b7564f12365dbfa4a8a0d5a853f",
            "chr6.fna.gz": "c6b5c2b2e2ca35ca344f0effde41a8ef",
            "chr7.fna.gz": "4a1783cdaaa2eabaf519d4af64b28a97",
            "chr8.fna.gz": "9eb00c85eea5bd8c3cd657c5222e11f8",
            "chr9.fna.gz": "20f430cf235d488731b22feb94761cb8",
            "chr10.fna.gz": "bb89574740611a25a39e701c6d325af8",
            "chr11.fna.gz": "e734e9db468dc00e3f66750125afb95c",
            "chr12.fna.gz": "ee240c28c4b1a3d8c8629584cd74a1dc",
            "chr13.fna.gz": "0d34e0e2ee8b4c9f6ce9ff1d9da45ebc",
            "chr14.fna.gz": "62dc673400b14e3358bfadbe1a3eeb1e",
            "chr15.fna.gz": "67fc014176fdd16dee78f7ada13442bf",
            "chr16.fna.gz": "5bbf216e70f852b92086873ff57a75f0",
            "chr17.fna.gz": "bec08127a5a09e7a5ebb4b0d7935da71",
            "chr18.fna.gz": "cd7fba61549c548a9cf0933a75ad0f8d",
            "chr19.fna.gz": "74f27da749d5413748750004ae5f41aa",
            "chr20.fna.gz": "72f06b446f0419457f5db73ba5952686",
            "chr21.fna.gz": "73a7361a0ec60c88fc8a19dd4e071201",
            "chr22.fna.gz": "af2eb2257f7a4f42802f53e360d86400",
            "chrX.fna.gz": "c4b5470fe49284db1f98a94c997179e9",
            "chrY.fna.gz": "e32e7eac0f2f17f070c0244ecaa60a46",
        }

        # Check the remote for the filename structure, one for each autosome.
        for contig in list(range(1, 23)) + ["X", "Y"]:
            filename = f"chr{contig}.fna.gz"
            url = basename + filename
            resource = RemoteResource(
                dest_directory=self.dest_dir,
                dest_filename=filename,
                root_directory=self.root_directory,
                checksum=checksums.get(filename),
                url=url,
            )
            resources.append(resource)
        return resources

    def prepare_resource(self, resource: RemoteResource) -> str:
        logging.info(f"Downloading {resource.url}")
        resource.download_resource()
        return resource.fully_qualified_dest_filename

    def prepare(self):
        return [
            self.prepare_resource(resource) for resource in self.get_remote_resources()
        ]


class Hg38chromResourcePreparer(ResourcePreparer):
    """ Prepackaged object for downloading hg38 chroms from UCSC. Returns the GenomeResource object associated with it.
    Methods like these should be tightly coupled with the data, and should NOT be very reusable. They specify a specific way
    to download and prepare a specific dataset. We should chose from a predefine set of pipelines.
    """

    def get_remote_resources(self) -> List[RemoteResource]:
        dest_dir = "hg38"

        url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chromFa.tar.gz"
        checksum = "a5aa5da14ccf3d259c4308f7b2c18cb0"

        obj = RemoteResource(
            dest_directory=dest_dir,
            dest_filename="hg38.chromFa.tar.gz",
            root_directory=self.root_directory,
            checksum=checksum,
            url=url,
        )
        return [obj]

    def prepare(self) -> List[str]:
        """ hg38 prepare method:

        Download the remote tarball to a local temp file
        Unpack the tarball
        (optionally) split all sequences on 'N's

        Returns the fully qualified path of the prepared filenames.
        """
        logging.info("Downloading")
        [resource] = self.get_remote_resources()
        resource.download_resource(overwrite=False)

        # For hg38 we expect there to be tar.gz
        with tarfile.open(resource.fully_qualified_dest_filename) as file:
            logging.info("Extracting hg38")
            file.extractall(resource.fully_qualified_dest_folder)
        basename = os.path.join(resource.fully_qualified_dest_folder, "chroms")
        return [str(filepath) for filepath in pathlib.Path(basename).glob("*.fa")]


class GRCh38Ensembl99ResourcePreparer(ResourcePreparer):
    """ Downloads the Ensembl datasets required for SpliceSite prediction in the DNABERT publication.
    We download the gff files as well, as we dont currently have another usecase for this reference.
    """

    def get_remote_resources(self) -> List[RemoteResource]:
        fasta_basename = "http://ftp.ensembl.org/pub/release-99/fasta/homo_sapiens/dna/"
        dest_dir = "GRCh38.ensembl.99"
        resources = list()

        fasta_checksums = {
            "Homo_sapiens.GRCh38.dna.chromosome.1.fa.gz": "6f157cbdaed6fcc811dfdef69f22b505",
            "Homo_sapiens.GRCh38.dna.chromosome.10.fa.gz": "24d6c88bf2a518a0ca386122bd568b78",
            "Homo_sapiens.GRCh38.dna.chromosome.11.fa.gz": "8cd66df217b4e872a65525db39a9b702",
            "Homo_sapiens.GRCh38.dna.chromosome.12.fa.gz": "96cdb417f9b049b75e3005cbf03027d9",
            "Homo_sapiens.GRCh38.dna.chromosome.13.fa.gz": "9049a7bcafadb55ad236052969a43500",
            "Homo_sapiens.GRCh38.dna.chromosome.14.fa.gz": "bd4f6eac70a32aaa64aeb05a59047cd8",
            "Homo_sapiens.GRCh38.dna.chromosome.15.fa.gz": "92d35108cfd2c0352116eed6c5d07dcf",
            "Homo_sapiens.GRCh38.dna.chromosome.16.fa.gz": "aa06df6a5350030370a5a5a7be87f2db",
            "Homo_sapiens.GRCh38.dna.chromosome.17.fa.gz": "f063898e374644bdc2502ba49c2cf7f1",
            "Homo_sapiens.GRCh38.dna.chromosome.18.fa.gz": "d8e1fd676b8527e7d0be5e8bca1d6b7d",
            "Homo_sapiens.GRCh38.dna.chromosome.19.fa.gz": "ca4f1a9927913fe5a62585a492d13c1a",
            "Homo_sapiens.GRCh38.dna.chromosome.2.fa.gz": "83420da4ab136c529d69a97784cc06e0",
            "Homo_sapiens.GRCh38.dna.chromosome.20.fa.gz": "09f9d3715f2f96dd7cb655e3e70c70f5",
            "Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz": "91d0fdc8e16fd9d6418765365bc5f50e",
            "Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz": "eb321ecd19ebe81ea0c8b88a969cd5fd",
            "Homo_sapiens.GRCh38.dna.chromosome.3.fa.gz": "2c81cc8b83ae869025eed0652c177955",
            "Homo_sapiens.GRCh38.dna.chromosome.4.fa.gz": "8522a6ed38b3d304654d8a43d73cec91",
            "Homo_sapiens.GRCh38.dna.chromosome.5.fa.gz": "6619cd2c8a4ff5152225b4342bfeccd2",
            "Homo_sapiens.GRCh38.dna.chromosome.6.fa.gz": "5f619230e8ce6e2282106d51bb0ab1b3",
            "Homo_sapiens.GRCh38.dna.chromosome.7.fa.gz": "6595478d50e702f16376afd6561c7efc",
            "Homo_sapiens.GRCh38.dna.chromosome.8.fa.gz": "e50080ae3c5169906499a6ade6cabe16",
            "Homo_sapiens.GRCh38.dna.chromosome.9.fa.gz": "69cdf198b7d535617285cb049b2701d1",
            "Homo_sapiens.GRCh38.dna.chromosome.X.fa.gz": "974059b6a79eeed96ccc8ee9d36d9e8e",
            "Homo_sapiens.GRCh38.dna.chromosome.Y.fa.gz": "c4d00789488f974a30f9bdf80718940b",
        }
        gff_basename = "http://ftp.ensembl.org/pub/release-99/gff3/homo_sapiens/"

        gff_checksums = {
            "Homo_sapiens.GRCh38.99.chromosome.1.gff3.gz": "d25f02ad6ae1a4282d36712ba18ffd7e",
            "Homo_sapiens.GRCh38.99.chromosome.10.gff3.gz": "12d89463b97578cd7d707220840b9cc8",
            "Homo_sapiens.GRCh38.99.chromosome.11.gff3.gz": "4fae3f3ed60cfacbb11d0e80e41426ca",
            "Homo_sapiens.GRCh38.99.chromosome.12.gff3.gz": "5f4afb1b18a4217486f1c02fae940ee0",
            "Homo_sapiens.GRCh38.99.chromosome.13.gff3.gz": "a30128f4721169fb02e39bd9491bc235",
            "Homo_sapiens.GRCh38.99.chromosome.14.gff3.gz": "1ea87c335645a44ca6199a18fbbd1889",
            "Homo_sapiens.GRCh38.99.chromosome.15.gff3.gz": "5a4403ea419263f645e9afdf7870dcc6",
            "Homo_sapiens.GRCh38.99.chromosome.16.gff3.gz": "a5a71da50a8c670f7aeb45d9a8161c49",
            "Homo_sapiens.GRCh38.99.chromosome.17.gff3.gz": "c05033a6feda2d7556ebcc7d9ecb09a8",
            "Homo_sapiens.GRCh38.99.chromosome.18.gff3.gz": "de29c4f9c5517a16b5a94215285cf544",
            "Homo_sapiens.GRCh38.99.chromosome.19.gff3.gz": "80eb251e613b677e68f378f0ce7d93dd",
            "Homo_sapiens.GRCh38.99.chromosome.2.gff3.gz": "9ade0d0ceea30d09ac0fa78b17b2bcd8",
            "Homo_sapiens.GRCh38.99.chromosome.20.gff3.gz": "045fc9d7c92684a265763732e42f8836",
            "Homo_sapiens.GRCh38.99.chromosome.21.gff3.gz": "ef107abe0babacc5b1bebed6c30c22de",
            "Homo_sapiens.GRCh38.99.chromosome.22.gff3.gz": "cdaae16026f02d8de6d299e15370d8e3",
            "Homo_sapiens.GRCh38.99.chromosome.3.gff3.gz": "49a9ee40a63dbe69b74a183c7419d9bc",
            "Homo_sapiens.GRCh38.99.chromosome.4.gff3.gz": "e55b4b344a88595f3db27dad307592e8",
            "Homo_sapiens.GRCh38.99.chromosome.5.gff3.gz": "5824c261df8161eb6494ae168f43ecf1",
            "Homo_sapiens.GRCh38.99.chromosome.6.gff3.gz": "416ab751d58299ea38067b40a7d49e46",
            "Homo_sapiens.GRCh38.99.chromosome.7.gff3.gz": "33532bca0f8d1f898494daf7d5217aa2",
            "Homo_sapiens.GRCh38.99.chromosome.8.gff3.gz": "7e04f12425dd01fe77fa66a38498c44b",
            "Homo_sapiens.GRCh38.99.chromosome.9.gff3.gz": "17d121d9f6aa0c6fe35912850cb900fe",
            "Homo_sapiens.GRCh38.99.chromosome.X.gff3.gz": "4bbbb922322e57399bea4b9247834849",
            "Homo_sapiens.GRCh38.99.chromosome.Y.gff3.gz": "549fee6028a3b992637f6d8d775160fb",
        }

        # Check the remote for the filename structure, one for each autosome.
        for contig in list(range(1, 23)) + ["X", "Y"]:
            fasta_filename = f"Homo_sapiens.GRCh38.dna.chromosome.{contig}.fa.gz"
            gff_filename = f"Homo_sapiens.GRCh38.99.chromosome.{contig}.gff3.gz"
            fasta_url = fasta_basename + fasta_filename
            gff_url = gff_basename + gff_filename
            fasta_resource = RemoteResource(
                dest_directory=dest_dir,
                dest_filename=fasta_filename,
                root_directory=self.root_directory,
                checksum=fasta_checksums.get(fasta_filename),
                url=fasta_url,
            )
            resources.append(fasta_resource)
            gff_resource = RemoteResource(
                dest_directory=dest_dir,
                dest_filename=gff_filename,
                root_directory=self.root_directory,
                checksum=gff_checksums.get(gff_filename),
                url=gff_url,
            )
            resources.append(gff_resource)
        return resources

    def prepare_resource(self, resource: RemoteResource) -> str:
        logging.info(f"Downloading {resource.url}")
        resource.download_resource()
        return resource.fully_qualified_dest_filename

    def prepare(self):
        return [
            self.prepare_resource(resource) for resource in self.get_remote_resources()
        ]
