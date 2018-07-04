# One-Shot Visual Imitation Learning via Meta-Learning

*A TensorFlow implementation of the two papers [One-Shot Visual Imitation Learning via Meta-Learning (Finn*, Yu* et al., 2017)](https://arxiv.org/abs/1709.04905) and [One-Shot Imitation from Observing Humans
via Domain-Adaptive Meta-Learning (Yu*, Finn* et al., 2018)](https://arxiv.org/abs/1802.01557).* Here are the instructions to run our experiments shown in the paper.

First clone the fork of the gym repo found [here](https://github.com/tianheyu927/gym), and following the instructions there to install gym. Switch to branch *mil*.

Then go to the `mil` directory and run `./scripts/get_data.sh` to download the data.

After downloading the data, training and testing scripts for MIL are available in `scripts/`.

**UPDATE (7/3/2018)**: to run the experiment with learned temporal loss as in the [One-Shot Imitation from Observing Humans
via Domain-Adaptive Meta-Learning](https://arxiv.org/abs/1802.01557) paper, take a look at `scripts/run_sim_push_video_only.sh` script.

*Note: The code only includes the simulated experiments.*
