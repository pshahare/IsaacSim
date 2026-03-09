# Pushing UR10e changes so others can clone and run

This repo is your **fork** (`origin` → `git@github.com:pshahare/IsaacSim.git`). To share the UR10e policy and examples with your team, push a branch they can clone.

## 1. Create a branch (you're currently in detached HEAD at v5.1.0)

```bash
cd /home/horde/ds/ps/IsaacSim

# Create and switch to a branch with your changes (e.g. feature/ur10e-policy)
git checkout -b feature/ur10e-policy
```

## 2. Stage all UR10e-related changes

```bash
git add source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/__init__.py
git add source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/ur10e.py
git add source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/ur10e_policy/
git add source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/tests/test_ur10e.py
git add source/standalone_examples/api/isaacsim.robot.policy.examples/ur10e_pick_cube.py
git add source/standalone_examples/api/isaacsim.robot.policy.examples/ur10e_standalone.py
git add docs/MIGRATION_FROM_STANDALONE.md
git add docs/PUSH_UR10E_CHANGES.md
```

Or stage everything in one go:

```bash
git add source/extensions/isaacsim.robot.policy.examples/ \
  source/standalone_examples/api/isaacsim.robot.policy.examples/ur10e_*.py \
  docs/MIGRATION_FROM_STANDALONE.md docs/PUSH_UR10E_CHANGES.md
```

## 3. Commit

```bash
git commit -m "Add UR10e reach/lift policy and standalone examples

- Add UR10eReachTargetPolicy (ur10e.py) and ur10e_policy assets (agent/env yaml, checkpoint)
- Add standalone examples: ur10e_pick_cube.py, ur10e_standalone.py
- Add test_ur10e.py for policy extension
- Add docs: MIGRATION_FROM_STANDALONE.md, PUSH_UR10E_CHANGES.md"
```

## 4. Push to your fork

```bash
git push -u origin feature/ur10e-policy
```

If your default branch is `main` and you want this on `main` instead:

```bash
git checkout main
git merge feature/ur10e-policy
git push origin main
```

## 5. What others do to clone and run your use case

**Option A – Clone and build from source (this fork)**

```bash
git clone git@github.com:pshahahare/IsaacSim.git
cd IsaacSim
git checkout feature/ur10e-policy   # or main if you merged there
./build.sh
# Then run examples with built kit Python (see MIGRATION_FROM_STANDALONE.md)
```

**Option B – Use a prebuilt Isaac Sim and only the scripts + policy data**

They can clone the repo, then copy into their Isaac Sim install:

- `source/extensions/isaacsim.robot.policy.examples/.../robots/ur10e.py` and `ur10e_policy/` into their `exts/isaacsim.robot.policy.examples/.../robots/`
- `source/standalone_examples/.../ur10e_*.py` into their `standalone_examples/.../`

Then run with their install’s `./python.sh` as in the migration doc.

## Optional: Don’t commit the large checkpoint

If you prefer not to push the `.pt` file (~1.3 MB), add to `.gitignore` and document where to get it:

```bash
echo "source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/ur10e_policy/*.pt" >> .gitignore
```

Then in `ur10e_policy/README.md` note that users must copy `ur10e_policy.pt` from a shared location or from their own training.
