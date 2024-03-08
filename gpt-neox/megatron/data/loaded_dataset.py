# Copyright (c) 2021, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
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

"""GPT2 style dataset."""

import os
import time

import numpy as np
import torch

from megatron import mpu, print_rank_0


class LoadedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_prefix,
        # documents,
        # indexed_dataset,
        # num_samples,
        # seq_length,
        # seed,
        # build_index_mappings=True,
        # use_shared_fs=True,
    ):
        self.dataset = np.memmap(data_prefix+".bin", dtype=np.uint16, mode="r", order="C", shape=(10000*1024, 2049))

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {"text": np.array(sample, dtype=np.int64)}
