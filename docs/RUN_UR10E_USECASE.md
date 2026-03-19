# How to run the UR10e use case (in-detail flow)

**Start here:** For an overview of what we did, the repo layout, and how Isaac Sim differs from Isaac Lab Arena (physics, coordinates, inference), see **docs/README_UR10E_AND_ISAAC_SIM.md**.

This guide walks through running the UR10e pick-cube policy use case after the changes have been merged and pushed to the fork. It covers **two ways**: (A) clone + build from this repo, (B) use an existing Isaac Sim standalone install.

---

## Prerequisites (both flows)

- **OS**: Linux (Ubuntu 22.04 recommended) or Windows 10/11  
- **GPU**: NVIDIA GPU meeting [Isaac Sim requirements](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html) (e.g. RTX 4080+ or datacenter equivalent)  
- **Driver**: Up-to-date NVIDIA driver  
- **Display**: For the default GUI example, a display (or virtual display) is required; for headless, set `SimulationApp({"headless": True})` in the script  

---

## Flow A: Clone this repo, build, then run

Use this when you want to build Isaac Sim from source and run the UR10e example from the same repo.

### Step 1 – Clone the repository and get the branch

```bash
# Clone the fork (use your fork URL; example below)
git clone https://github.com/pshahare/IsaacSim.git
cd IsaacSim

# Or with SSH:
# git clone git@github.com:pshahare/IsaacSim.git
# cd IsaacSim

# Install Git LFS and pull large files (required for assets)
git lfs install
git lfs pull

# Switch to the branch that has the UR10e changes
git checkout feature/stackbot
# or:  git checkout feature/ur10e-policy
# or:  git checkout main   # if changes were merged to main
```

### Step 2 – Install build dependencies (Linux)

```bash
# Essential build tools
sudo apt-get update
sudo apt-get install -y build-essential git

# GCC/G++ 11 (Isaac Sim build expects 11; 12+ is not supported)
sudo apt-get install -y gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200

# Verify
gcc --version
g++ --version
```

On Windows: install Visual Studio with “Desktop development with C++” and Windows SDK (see main [README.md](../README.md)).

### Step 3 – Build Isaac Sim

From the **repo root**:

```bash
./build.sh
```

- First build can take a long time (downloads and compiles).
- You may be prompted to accept Omniverse licensing terms.
- For release-only build: `./build.sh -r`
- For help: `./build.sh -h`

When the build finishes, you should have:

- **Linux**: `_build/linux-x86_64/release/` (and optionally `_build/linux-x86_64/debug/`)
- **Windows**: `_build/windows-x86_64/release/`

### Step 4 – Set up the Python environment (once per shell or session)

From the **repo root**, source the script that sets `CARB_APP_PATH`, `ISAAC_PATH`, `PYTHONPATH`, `LD_LIBRARY_PATH`, etc.:

**Linux:**

```bash
source _build/linux-x86_64/release/setup_python_env.sh
```

**Windows (PowerShell):**

```powershell
.\_build\windows-x86_64\release\setup_python_env.bat
```

Keep this terminal open for the next step (or run the same `source` in any new terminal where you want to run the example).

### Step 5 – Run the UR10e pick-cube example

Still from the **repo root**, with the environment from Step 4 active:

**Linux:**

```bash
_build/linux-x86_64/release/kit/python/bin/python3 \
  source/standalone_examples/api/isaacsim.robot.policy.examples/ur10e_pick_cube.py
```

**With a custom goal position (x y z):**

```bash
_build/linux-x86_64/release/kit/python/bin/python3 \
  source/standalone_examples/api/isaacsim.robot.policy.examples/ur10e_pick_cube.py \
  --goal 0.5 0.0 0.35
```

**Alternative (if a `python.sh` exists in the build output):**

```bash
cd _build/linux-x86_64/release
./python.sh  # if present; then you can pass the script path relative to repo root
```

If your shell has `python3` on the PATH and it is the kit Python (because you sourced `setup_python_env.sh` and it prepended the kit path), you can also run:

```bash
python3 source/standalone_examples/api/isaacsim.robot.policy.examples/ur10e_pick_cube.py
```

**Simpler UR10e example (ground + cube, fixed goal):**

```bash
_build/linux-x86_64/release/kit/python/bin/python3 \
  source/standalone_examples/api/isaacsim.robot.policy.examples/ur10e_standalone.py
```

### Step 6 – What you should see

- Isaac Sim window opens (unless you changed the script to headless).
- Scene: ground, table (SeattleLabTable for pick_cube), DexCube, UR10e with Robotiq 2f-140.
- Robot uses the trained policy to reach toward the cube and lift toward the goal.
- Press **Play** in the timeline if the simulation does not start automatically.
- Close the window or stop the script to exit.

### Optional – Run the UR10e tests

From repo root, with the same environment sourced:

```bash
# Run the policy extension tests (includes test_ur10e)
# Exact command depends on the repo’s test runner; often something like:
_build/linux-x86_64/release/kit/python/bin/python3 -m pytest \
  source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/tests/test_ur10e.py \
  -v
```

If the project uses a different test entry (e.g. via kit or premake), use the test instructions in the main README or in the extension’s `config/extension.toml`.

---

## Flow B: Use an existing Isaac Sim standalone install

Use this when you already have a **prebuilt** Isaac Sim install (e.g. from NVIDIA or your own `python.sh` layout) and only need the UR10e scripts and policy data.

### Step 1 – Clone the repo and get the branch

Same as Flow A, Step 1:

```bash
git clone https://github.com/pshahare/IsaacSim.git
cd IsaacSim
git lfs install
git lfs pull
git checkout feature/stackbot
```

### Step 2 – Copy the UR10e files into your install

Assume your Isaac Sim install root is `ISAAC_INSTALL` (e.g. `/home/user/isaac-sim` or `C:\Users\...\isaac-sim`).

**Policy extension (required):**

- Copy the entire folder:
  - From: `IsaacSim/source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/`
  - To: `ISAAC_INSTALL/exts/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/`
- So that you have:
  - `ISAAC_INSTALL/exts/.../robots/ur10e.py`
  - `ISAAC_INSTALL/exts/.../robots/ur10e_policy/` (with `agent.yaml`, `env.yaml`, `ur10e_policy.pt`, README)

If the path under `exts` differs (e.g. no `isaacsim` subfolder), adjust so that the relative path from `ur10e.py` to `ur10e_policy/` is the same (the code looks for `ur10e_policy` next to or above the script).

**Standalone examples (optional but recommended):**

- Copy:
  - `IsaacSim/source/standalone_examples/api/isaacsim.robot.policy.examples/ur10e_pick_cube.py`
  - `IsaacSim/source/standalone_examples/api/isaacsim.robot.policy.examples/ur10e_standalone.py`
- To:
  - `ISAAC_INSTALL/standalone_examples/api/isaacsim.robot.policy.examples/`

(Again, if your install uses a different path for standalone examples, place them there and use that path in the command below.)

### Step 3 – Run from the install root

From `ISAAC_INSTALL`:

```bash
./python.sh standalone_examples/api/isaacsim.robot.policy.examples/ur10e_pick_cube.py
```

With a custom goal:

```bash
./python.sh standalone_examples/api/isaacsim.robot.policy.examples/ur10e_pick_cube.py --goal 0.5 0.0 0.35
```

Simpler example:

```bash
./python.sh standalone_examples/api/isaacsim.robot.policy.examples/ur10e_standalone.py
```

No build step is required in this flow.

---

## One-page quick reference (Flow A – build from repo)

| Step | Command (Linux, from repo root) |
|------|---------------------------------|
| 1. Clone & branch | `git clone https://github.com/pshahare/IsaacSim.git && cd IsaacSim && git lfs install && git lfs pull && git checkout feature/stackbot` |
| 2. Dependencies | `sudo apt-get install -y build-essential gcc-11 g++-11` (+ alternatives as above) |
| 3. Build | `./build.sh` |
| 4. Env (each new shell) | `source _build/linux-x86_64/release/setup_python_env.sh` |
| 5. Run pick-cube | `_build/linux-x86_64/release/kit/python/bin/python3 source/standalone_examples/api/isaacsim.robot.policy.examples/ur10e_pick_cube.py` |
| 5b. Run with goal | Same as 5, add `--goal 0.5 0.0 0.35` |

---

## Troubleshooting

- **“Could not find Isaac Sim assets folder”**  
  Assets come from the build or from Nucleus. After building, ensure you ran `source _build/.../release/setup_python_env.sh` so the app can find the assets root.

- **“No module named 'isaacsim'”**  
  You are not using the kit Python or the env is not set. Use the full path to `_build/.../release/kit/python/bin/python3` and run after sourcing `setup_python_env.sh`.

- **“ur10e_policy.pt not found”**  
  Ensure `ur10e_policy/` (with `ur10e_policy.pt`, `env.yaml`, `agent.yaml`) lives next to or above `ur10e.py` in the extension tree (built or copied).

- **Display/headless**  
  For headless runs, edit the example script and set `SimulationApp({"headless": True})`. You may need a virtual display or run on a machine with a GPU and headless support.

- **Build failures**  
  See the main [README.md](../README.md) and [Isaac Sim documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html); ensure GCC 11 and all required dependencies are installed.
