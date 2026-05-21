# IsaacLab Dyros Reinforcement Learning Environment

This reinforcement learning environment is configured to work with:
- **IsaacLab** version 2.3.1
- **rsl-rl-lib** version 3.2.0
- **Isaac Sim** version 5.1.0
- **Ubuntu** version $\ge$ 22.04

## Prerequisites

Before installing this environment, you must have the following installed:
- **IsaacLab** version 2.3.1
- **Isaac Sim** version 5.1.0

Please refer to the official IsaacLab and Isaac Sim documentation for installation instructions.

## Installation

**Important:** Use a conda environment for this project.

1. Activate the conda environment that was created during IsaacLab installation:
   ```bash
   conda activate your_env_name
   ```
   
   Note: The conda environment should have been created when you installed IsaacLab. If you haven't installed IsaacLab yet, please do so first following the official IsaacLab installation instructions.

2. Run the installation script:
   ```bash
   ./install_pip.sh
   ```

This script will install all required Python packages from the `source` directory in editable mode.

## Usage

After installation, you can use this environment for reinforcement learning tasks with the specified versions of IsaacLab, rsl-rl-lib, and Isaac Sim.
