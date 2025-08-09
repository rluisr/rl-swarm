# SwarmGameManager Multi-GPU Fix

## Problem
SwarmGameManager initialization failed with:
```
TypeError: SwarmGameManager.__init__() missing 2 required positional arguments: 'coordinator' and 'communication'
```

## Root Cause
The Hydra instantiate function was trying to create SwarmGameManager without properly passing the required `coordinator` and `communication` arguments.

## Solution Applied
Modified `rgym_exp/runner/swarm_launcher.py` to:
1. Instantiate `coordinator` and `communication` separately from the config
2. Pass them explicitly to the `game_manager` instantiation

## Code Change
```python
# Before:
game_manager = instantiate(cfg.game_manager)

# After:
coordinator = instantiate(cfg.game_manager.coordinator)
communication = instantiate(cfg.game_manager.communication)
game_manager = instantiate(
    cfg.game_manager,
    coordinator=coordinator,
    communication=communication
)
```

## Result
SwarmGameManager now properly receives both required arguments during initialization, resolving the multi-GPU compatibility issue.