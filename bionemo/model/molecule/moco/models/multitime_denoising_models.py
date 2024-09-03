# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from bionemo.model.molecule.moco.arch.multitime_model import MegalodonDotFN


class ModelBuilder:
    """A builder class for creating model instances based on a given model name and arguments."""

    def __init__(self):
        """Initializes the ModelBuilder with a dictionary of available model classes."""
        self.model_classes = {
            "mtmegadotfn": MegalodonDotFnWrapper,
        }

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


class MegalodonDotFnWrapper(MegalodonDotFN):
    """A wrapper class for the MoCo model."""

    def __init__(self, args_dict, time_type="continuous", timesteps=None):
        """
        Initializes the DiTWrapper.

        Args:
            args_dict (dict): A dictionary of arguments for initializing the MoCo model.
        """
        self.args = args_dict
        self.time_type = time_type
        self.timesteps = timesteps
        super().__init__(**args_dict)

    def forward(self, batch, time, conditional_batch=None, timesteps=None):
        """
        Forward pass of the MoCo model.

        Args:
            batch (torch_geometric.data.Batch): The input batch.
            time (Tensor): The time tensor.

        Returns:
            dict: The output of the MoCo model.
        """
        timesteps = timesteps if timesteps is not None else self.timesteps
        if self.time_type == "discrete" and timesteps is not None:
            continuous_time = (timesteps - time["continuous"].float() - 1) / timesteps
            discrete_time = (timesteps - time["discrete"].float() - 1) / timesteps
        else:
            continuous_time, discrete_time = time['continuous'], time['discrete']

        out = super().forward(
            batch=batch["batch"],
            X=batch["x_t"],
            H=batch["h_t"],
            E=batch["edge_attr_t"],
            E_idx=batch["edge_index"],
            continuous_time=continuous_time,
            discrete_time=discrete_time,
        )
        return out
