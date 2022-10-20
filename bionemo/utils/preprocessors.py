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

from abc import ABC, abstractmethod

from dataclasses import dataclass
from hashlib import md5
from typing import List, TypeVar
from bionemo.utils.fasta import FastaUtil


# NOTE these serve no purpose other than showing the expected inputs.
T = TypeVar("T")
S = TypeVar("S")


class Preprocessor(ABC):
    """ Generic preprocessing method. Given an applicable (function) and a sequence of objects,
    we can apply the passed function to every object in the sequence with the `map` method.

    This method captures one interface for representing these constraints. Both `get_elements` and `apply` are
    entirely up to the implementor, if there is a set of files and a set of operations that make sense for a 
    specific preprocessing protocol, implement it!

    ResourcePreparer().prepare() always should return a list where every element has the same type.
    
    Examples:
    ```python
    Resource
    elements: T = ResourcePreparer().prepare() -> List[T]
    filename: S = 'a_filename_we_guarantee_exists.fa'

    elements: List[T] = ResourcePreparer().prepare() 
    Preprocessor: Callable[[T], S] # A function that accepts an object of type T, and returns type S.

    def my_preprocsesor(obj: T) -> S:
        # Do work
        yield S

    preprocess = map(my_preprocessor, elements)
    ``` 
    """

    @abstractmethod
    def apply(self, thing: T) -> S:
        """ User implemented function that is applied to all objects in `get_elements`. """
        raise NotImplementedError

    @abstractmethod
    def get_elements(self) -> List[T]:
        """ User implemented function that defines 'things' """
        raise NotImplementedError

    def map(self):
        yield from map(self.apply, self.get_elements())


@dataclass
class FastaSplitNsPreprocessor(Preprocessor):
    ''' Preprocessor that removes Ns, and creates new contigs between Ns. Runs of Ns are split once. '''
    fasta_filenames: List[str]

    def apply(self, fasta_filename):
        new_filename = FastaUtil.from_filename(fasta_filename).split_on_ns().write(
            fasta_filename + ".chunked.fa"
        )
        return new_filename

    def get_elements(self):
        return self.fasta_filenames
