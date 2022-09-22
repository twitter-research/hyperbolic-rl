# Hyperbolic reinforcement learning minimal repository

Install dependencies via [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html):
```sh
conda env create -f environment.yml
conda activate procgen_hyper
```

## Replicating the results

Run experiments by executing _main_hydra.py_ and override the appropriate arguments (see [hydra](https://hydra.cc/docs/intro/) for details), e.g. to run PPO with hyperbolic representations on _starpilot_
with generalization configurations:

```setup
python main_hydra.py agent=onpolicy/hyperbolic/ppo env=gen/starpilot
```
