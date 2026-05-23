"""LICENSE.

This project as a whole is licensed under the Apache License, Version 2.0.
THIRD-PARTY LICENSES
Third-party software already included in HoloAgent is governed by the separate
Open Source license terms under which the third-party software has been
distributed.
NOTICE ON LICENSE COMPATIBILITY FOR DISTRIBUTORS
Notably, this project depends on the third-party software FAST-LIVO2 and HOVSG.
Their default licenses restrict commercial use—separate permission from their
original authors is required for commercial integration/redistribution.
The third-party software FAST-LIVO2 dependency (licensed under GPL-2.0-only)
utilizes rpg_vikit-ros2 which contains components under the GPL-3.0. Please be
aware of license compatibility when distributing a combined work.
DISCLAIMER
Users are solely responsible for ensuring compliance with all applicable
license terms when using, modifying, or distributing the project. Project
maintainers accept no liability for any license violations arising from such
use."""

# pylint: disable=E,W,R,F
"""semantic_scene_reconstruction.

Performs semantic scene reconstruction and builds a Hierarchical Multimodal Scene Graph (HMSG)
for different scenes based on the provided configuration.

The script loads configuration via Hydra, creates a `Graph` instance,
generates and saves the feature map, point cloud, masked point clouds, and feature files,
then calls the graph construction pipeline to write results to disk.

Notes
-----
- Depends on `hmsg.graph.graph.Graph` and config files under `config/semantic_scene_reconstruction/`.
- Creates and writes to the output directory on disk at runtime (has side effects).
- Select a scene profile via Hydra override, e.g.:
    python semantic_scene_reconstruction.py profiles=ic3f"""


import os
import sys
import warnings

import hydra
from omegaconf import DictConfig

warnings.filterwarnings("ignore")
# Add project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from memory.hmsg.graph.graph_runtime import GraphRuntime


def run_scene_reconstruction(params: DictConfig):
    """
    Core function that performs semantic reconstruction and scene graph construction for a single scene.

    Initializes and runs the `Graph` reconstruction pipeline based on Hydra-injected `params`. Main steps:
    1. Compute input/output paths from `params.main.scene_id` and `params.main.dataset_path`;
    2. Create the output directory and initialize `Graph(params)`;
    3. Generate the feature map and save the point cloud, masked point clouds, and features;
    4. Call `build_hier_multimodal_scene_graph` to build and save the multimodal scene graph.

    Args:
        params (DictConfig): Hydra configuration object, expected to contain `params.main.scene_id`,
            `params.main.dataset_path`, `params.main.save_path`, and `params.main.dataset`.

    Returns:
        None

    Side effects:
        Creates and writes to `save_dir` on disk, containing point clouds, features, and graph data.
    """
    scene_ids = [params.main.scene_id]  # Use the scene ID specified in the config

    if hasattr(params, "main") and hasattr(params.main, "scene_ids") and params.main.scene_ids:
        # If multiple scene IDs are defined in the config, use them
        scene_ids = params.main.scene_ids

    for scene_id in scene_ids:
        # Update the scene_id in params
        if hasattr(params, "main"):
            params.main.scene_id = scene_id
        # Create save directory
        params.main.dataset_path = os.path.join(
            params.main.dataset_path, scene_id
        )  # dataset_path/scene_id
        save_dir = os.path.join(params.main.save_path, params.main.dataset, scene_id)
        params.main.save_path = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        print("dataset_path: ", params.main.dataset_path)
        print("save_path: ", save_dir)
        # Build scene graph
        print(params.main.dataset)
        service = GraphRuntime(params)
        service.build(save_path=save_dir)


@hydra.main(
    version_base=None,
    config_path="../../config/semantic_scene_reconstruction",
    config_name="semantic_scene_reconstruction",
)
def main(params: DictConfig):
    """Main function that loads configuration via Hydra and runs scene reconstruction.

    Profile selection (via Hydra override):
        python semantic_scene_reconstruction.py profiles=ic3f
        python semantic_scene_reconstruction.py profiles=ic4f
        python semantic_scene_reconstruction.py profiles=ic7f
        python semantic_scene_reconstruction.py profiles=sh3f
        python semantic_scene_reconstruction.py profiles=custom main.scene_id=MyScene main.dataset_path=/path/to/data
        python semantic_scene_reconstruction.py profiles=custom main.slow_reasoning=true
    """
    run_scene_reconstruction(params)


if __name__ == "__main__":
    main()
