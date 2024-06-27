# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from bionemo.bionemo.model.molecule.moco.models.model_zoo.eqgat.eqgat_denoising_model import DenoisingEdgeNetwork
from bionemo.model.molecule.moco.models.moco import MoCo


class ModelBuilder:
    """A builder class for creating model instances based on a given model name and arguments."""

    def __init__(self):
        """Initializes the ModelBuilder with a dictionary of available model classes."""
        self.model_classes = {"moco": MOCOWrapper, "eqgat": EQGATWrapper}

    def create_model(self, model_name: str, args_dict: dict, wrapper_args: dict):
        """
        Creates an instance of the specified model.

        Args:
            model_name (str): The name of the model to create.
            args_dict (dict): A dictionary of arguments to pass to the model.

        Returns:
            nn.Module: An instance of the specified model.

        Raises:
            ValueError: If the model name is not recognized.
        """
        args_dict = args_dict if args_dict is not None else {}
        wrapper_args = wrapper_args if wrapper_args is not None else {}
        model_class = self.model_classes.get(model_name.lower())
        if model_class is None:
            raise ValueError(f"Unknown model name: {model_name}")
        return model_class(args_dict, **wrapper_args)

    def register_model(self, model_name: str, model_class):
        """
        Registers a new model class with the builder.

        Args:
            model_name (str): The name of the model.
            model_class (type): The class of the model.
        """
        self.model_classes[model_name.lower()] = model_class


class MOCOWrapper(MoCo):
    """A wrapper class for the MoCo model."""

    def __init__(self, args_dict):
        """
        Initializes the MOCOWrapper.

        Args:
            args_dict (dict): A dictionary of arguments for initializing the MoCo model.
        """
        self.args = args_dict
        super().__init__(**args_dict)

    def forward(self, batch, time):
        """
        Forward pass of the MoCo model.

        Args:
            batch (torch_geometric.data.Batch): The input batch.
            time (Tensor): The time tensor.

        Returns:
            dict: The output of the MoCo model.
        """
        out = super().forward(
            batch=batch.batch,
            X=batch["x_t"],
            H=batch["h_t"],
            E=batch["edge_attr_t"],
            E_idx=batch["edge_index"],
            t=time,
        )
        return out


class EQGATWrapper(DenoisingEdgeNetwork):
    """A wrapper class for the EQGAT model."""

    def __init__(self, args_dict, time_type="continuous", timesteps=None):
        """
        Initializes the EQGATWrapper.

        Args:
            args_dict (dict): A dictionary of arguments for initializing the EQGAT model.
        """
        self.args = args_dict
        self.time_type = time_type
        self.timesteps = timesteps
        super().__init__(**args_dict)

    def forward(self, batch, time):
        """
        Forward pass of the EQGAT model.

        Args:
            batch (torch_geometric.data.Batch): The input batch.
            time (Tensor): The time tensor.

        Returns:
            dict: The output of the EQGAT model.
        """
        if self.time_type == "discrete" and self.timesteps is not None:
            time = (self.timesteps - time.float()) / self.timesteps
        temb = time.clamp(min=0.001)
        temb = temb.unsqueeze(dim=1)
        out = super().forward(
            x=batch["h_t"],
            t=temb,  # should be in [0, 1]?
            pos=batch["x_t"],
            edge_index_global=batch["edge_index"],
            edge_attr_global=batch["edge_attr_t"],
            batch=batch["batch"],
            batch_edge_global=batch["batch"][batch["edge_index"][0]],
        )
        out = {
            "x_hat": out["coords_pred"],
            "h_logits": out["atoms_pred"],
            "edge_attr_logits": out["bonds_pred"],
        }
        return out
