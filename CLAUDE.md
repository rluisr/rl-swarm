# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL Swarm is a peer-to-peer system for reinforcement learning that allows collaborative model training leveraging collective intelligence. It's built on the GenRL library and currently runs the reasoning-gym swarm on the Gensyn Testnet.

## Common Development Commands

### Running the Swarm

**CPU mode:**
```bash
docker-compose run --rm --build -Pit swarm-cpu
```

**GPU mode:**
```bash
docker-compose run --rm --build -Pit swarm-gpu
```

**Experimental mode (for development):**
```bash
python3 -m venv .venv
source .venv/bin/activate
./run_rl_swarm.sh
```

**GPU optimized startup:**
```bash
./start_with_gpu_optimization.sh
```

### Installing Dependencies

```bash
pip install --upgrade pip
pip install gensyn-genrl==0.1.4
pip install reasoning-gym>=0.1.20
pip install trl
pip install hivemind@git+https://github.com/gensyn-ai/hivemind@639c964a8019de63135a2594663b5bec8e5356dd
```

### Running Tests

Tests are located in `web/api/` directory and use pytest:
```bash
pytest web/api/dht_pub_test.py
pytest web/api/kinesis_test.py
```

## Architecture & Key Components

### Core Module Structure

The main module is `rgym_exp` which contains:

- **`rgym_exp/src/trainer.py`**: `GRPOTrainerModule` - Handles model training with GPU optimization features including:
  - Multi-GPU support via DataParallel
  - Gradient checkpointing
  - Automatic batch size adjustment
  - Memory optimization

- **`rgym_exp/src/manager.py`**: `SwarmGameManager` - Manages the training lifecycle:
  - Coordinates with blockchain via ModalSwarmCoordinator
  - Handles peer communication through Hivemind
  - Manages model pushing to HuggingFace
  - Periodic GPU cache clearing

- **`rgym_exp/src/data.py`**: `ReasoningGymDataManager` - Handles dataset management for reasoning tasks

- **`rgym_exp/src/rewards.py`**: `RGRewards` - Defines reward functions for the reasoning gym environment

### Configuration

Main configuration is in `rgym_exp/config/rg-swarm.yaml` with key settings:
- Training parameters (max_round, max_stage, num_generations)
- GPU optimization flags (gradient_checkpointing, gradient_accumulation_steps)
- Blockchain settings (contract address, org_id)
- Communication settings (initial peers, hivemind configuration)

### GPU Optimization Features

Located in `rgym_exp/src/utils/`:
- `omega_gpu_resolver.py`: GPU detection and batch size optimization
- `gpu_monitor.py`: Real-time GPU monitoring and CUDA OOM error handling

### Communication Layer

Uses Hivemind for peer-to-peer communication:
- DHT-based peer discovery
- Gossip protocol for model sharing
- Initial bootstrap peers configured in YAML

### Blockchain Integration

- Uses Alchemy Modal for Web3 authentication
- Smart contract interaction for on-chain identity
- Vote tracking at contract address: 0xFaD7C5e93f28257429569B854151A1B8DCD404c2

## Important Files and Locations

- **Entry point**: `run_rl_swarm.sh` - Main startup script
- **Config**: `rgym_exp/config/rg-swarm.yaml` - All configuration
- **Logs**: `logs/` directory contains swarm.log, yarn.log, wandb logs
- **Identity**: `swarm.pem` - Peer identity file (preserve for consistent peer ID)
- **Modal login**: `modal-login/` - Alchemy authentication server
- **User data**: `modal-login/temp-data/userData.json` - Authentication data

## Development Notes

### When modifying the trainer
- Ensure DataParallel compatibility for multi-GPU setups
- Handle both wrapped and unwrapped model attributes
- Clear GPU cache periodically to prevent OOM errors

### When updating configuration
- User configs are copied to `configs/` directory
- Set `GENSYN_RESET_CONFIG` environment variable to reset to defaults
- Configuration uses Hydra framework with OmegaConf

### Testing changes
- Run tests in `web/api/` for API functionality
- Monitor GPU memory usage with `rgym_exp/src/utils/gpu_monitor.py`
- Check logs in `logs/` directory for debugging

### Environment Variables
- `IDENTITY_PATH`: Path to swarm.pem file
- `CONNECT_TO_TESTNET`: Enable testnet connection
- `HF_TOKEN` or `HUGGINGFACE_ACCESS_TOKEN`: For model uploads
- `MODEL_NAME`: Override default model selection
- `CPU_ONLY`: Force CPU mode even with GPU available
- `DOCKER`: Set when running in Docker container