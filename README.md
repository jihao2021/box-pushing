# Box-Pushing Simulation (TensorFlow 2)

This repository implements a multi-agent box-pushing simulation based on the JCISE & AIEDAM paper. The simulation uses reinforcement learning with TensorFlow 2, modeling agents (pushers) that collaboratively move a box toward a target while avoiding obstacles.

---

## 🛠️ Installation

To set up the environment, use either `mamba` or `conda`:

```bash
# Using mamba
mamba env create -f tf2.yml
mamba activate tf2

# Or using conda
conda env create -f tf2.yml
conda activate tf2
```

---

## Running the Simulation

Once the environment is set up:

```bash
python train.py
```

This command starts the training process for the agents in the simulation.

---

## 📁 Project Structure

```
.
├── train.py
├── tf2.yml
├── scripts/
│   ├── utils.py
│   ├── game_object.py
│   ├── pushers.py
│   ├── box.py
│   ├── pygame_env.py
│   ├── boxer.py
│   └── DQN_modified_plus_simplified.py
```

### Description of Modules

- **scripts/utils.py**

  - Defines constants and helper functions.
  - Includes `setup_background` to visualize the box-pushing layout.
  - Coordinate transformation utilities (global ↔ local frames).

- **scripts/game_object.py**

  - `Target`: Defines the goal location for the box.
  - `Obstacle`: Represents immovable barriers.
  - `Patches`, `BoxChecker`: Utility classes for displaying results.

- **scripts/pushers.py**

  - Contains the `Pushers` class representing agents that interact with the box.

- **scripts/box.py**

  - `Box` class that handles box properties and physics for movement.

- **scripts/pygame_env.py**

  - Initializes the entire environment using Pygame, integrating all components (box, pushers, target, obstacles).

- **scripts/boxer.py**

  - Implements the `Boxer` class with the neural network policy for each agent.
  - Selects optimal actions for navigation and collision avoidance.

- **scripts/DQN_modified_plus_simplified.py**
  - Modified DQN architecture.
  - Manages state, action space, model loading/saving, and interaction with the `Boxer` agent.

---

## Citation

If you use this codebase for your research or academic work, please consider citing the original JCISE & AIEDAM paper the simulation is based on.
