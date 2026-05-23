# FSR-VLN

(This Repository is adapted from the original codebase in [HoloAgent](https://github.com/HorizonRobotics/HoloAgent) and is under active development. Please refer to the original HoloAgent repository for the most up-to-date code and documentation.)

[![📄 arXiv](https://img.shields.io/badge/📄-arXiv-b31b1b)](https://arxiv.org/abs/2509.13733)

**FSR-VLN: Fast and Slow Reasoning for Vision-Language Navigation with Hierarchical Multi-modal Scene Graph**

<img src="./docs/assets/FSR_VLN_framework.png" alt="Overall Framework" width="700"/>

## Overview

A vision-language navigation system that combines hierarchical multi-modal scene graphs (HMSG) with fast-to-slow reasoning (FSR) for efficient robot navigation and spatial understanding.

## Setup

1. Create workspace and clone repository:

  ```bash
  mkdir -p 3DSG/data/ # For datasets 
  cd 3DSG
  git clone <repo-url> scene_graph
  cd scene_graph
  ```

  Expected workspace structure:

  ```text
  3DSG/
  ├── data
  │   └── <dataset name>/
  │       └── <scene name>/
  │           ├── images/
  │           ├── depth/
  │           ├── poses.txt
  │           └── camera_info.yaml
  └── scene_graph/
      ├── dataset_generation/
      ├── docs/
      ├── environment.yaml
      ├── fsr_vln/
      ├── nav_agent/
      ├── README.md
      └── scripts/
  ```

1. Install dependencies:

```bash
# Navigate to FSR-VLN directory
cd scene_graph/fsr_vln/

# Create conda environment
conda env create -f ../environment.yaml
conda activate fsrvln

# Install package
pip install -e .

# (Optional) Install Habitat Sim for simulation
conda install habitat-sim -c conda-forge -c aihabitat
```

1. Set up Environment Variables:

```bash
mv .env.example .env
# Edit .env with your Azure OpenAI credentials and other config as needed
```

1. Download model checkpoints (optional because checkpoints are also automatically downloaded at runtime if not found):

**Open CLIP Model** ([CLIP-ViT-L-14-laion2B-s32B-b82K](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K)):

```bash
mkdir checkpoints
wget https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K/resolve/main/open_clip_pytorch_model.bin?download=true -O checkpoints/open_clip_pytorch_model.bin
```

**SAM Model** ([sam_vit_h_4b8939.pth](https://github.com/facebookresearch/segment-anything)):

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h_4b8939.pth
```

## Dataset (pre-scanned RGBD scenes)

Download [Horizon RGBD-Datasets](https://huggingface.co/datasets/HorizonRobotics/fsrvln_datasets):

```bash
mkdir -p /mnt/holoagent/fsrvln/rgbd_datasets/
# Extract to: /mnt/holoagent/fsrvln/rgbd_datasets/
unzip "icra_*.zip" -d /mnt/holoagent/fsrvln/rgbd_datasets/
```

Available scenes: `icra_sh3f`, `icra_ic3f`, `icra_ic4f`, `icra_ic7f`

## Usage

### Build Scene Graphs

Build hierarchical multi-modal scene graphs from RGBD datasets:

```bash
# from root of repository (3DSG/)
cd scene_graph/fsr_vln/
```

```bash
# Use preset scene profiles (ic3f, ic4f, ic7f, sh3f)
python application/semantic_scene_reconstruction_offline/semantic_scene_reconstruction.py \
  profiles=ic4f

# Or with custom dataset
python application/semantic_scene_reconstruction_offline/semantic_scene_reconstruction.py \
  profiles=custom

# custom profile with overrides:
python application/semantic_scene_reconstruction_offline/semantic_scene_reconstruction.py \
  profiles=custom main.scene_id=MyScene main.dataset_path=/path/to/rgbd_data
```

**Available profiles**: `ic3f`, `ic4f`, `ic7f`, `sh3f`, `custom`

Configs: `fsr_vln/config/semantic_scene_reconstruction/`

Output saved to: `/mnt/holoagent/fsrvln/scene_graphs_opensource/`

### Query Scene Graphs

Query and visualize scene graphs with natural language:

**Setup Azure OpenAI:**

1. Create [Azure OpenAI](https://portal.azure.com) resource
2. Deploy model (gpt-4, gpt-4o, or gpt-35-turbo)
3. Record: API key, Endpoint, Deployment name
4. Update config: `fsr_vln/config/visualize_graph/visualize_query_graph.yaml`

**Run queries:**

```bash
# Use preset scene profiles
python application/visualize_query_graph/visualize_query_graph.py \
  profiles=ic4f

# Or with custom profile
python application/visualize_query_graph/visualize_query_graph.py \
  profiles=custom

# custom profile with overrides:
python application/visualize_query_graph/visualize_query_graph.py \
  profiles=custom main.scene_id=MyScene main.dataset_path=/path/to/rgbd_data

# Query with slow reasoning enabled (LLM + symbolic)
python application/visualize_query_graph/visualize_query_graph.py \
  profiles=custom main.slow_reasoning=True

# Query with LLM retrieval enabled (LLM-only)
python application/visualize_query_graph/visualize_query_graph.py \
  profiles=custom main.llm_enable=True

# Query with both LLM retrieval and slow reasoning enabled
python application/visualize_query_graph/visualize_query_graph.py \
  profiles=custom main.llm_enable=True main.slow_reasoning=True
```

Configs: `fsr_vln/config/visualize_graph/`

## Citation

```bibtex
@misc{zhou2025fsrvlnfastslowreasoning,
      title={FSR-VLN: Fast and Slow Reasoning for Vision-Language Navigation with Hierarchical Multi-modal Scene Graph},
      author={Xiaolin Zhou and Tingyang Xiao and Liu Liu and Yucheng Wang and Maiyue Chen and Xinrui Meng and Xinjie Wang and Wei Feng and Wei Sui and Zhizhong Su},
      year={2025},
      eprint={2509.13733},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.13733}
}
```
