# Training adversarial agents to exploit weaknesses in deep control policies

This is the repo for paper Training adversarial agents to exploit weaknesses in deep control policies. 
Uses a vehicle following scenario where the adversary is the lead vehicle, 
whilst the follower vehicle is controlled by an learned policy.
Two target policies for the vehicle follower model are presented, an Imitation Learning (IL) and 
an A2C Reinforcement Learning (RL) policy.
The aim of the agent is to act in such a way that the follower vehicle behind it collides into it. 
Actions and states are limited to ensure the collisions could have been avoidable, 
and therefore present a weakness in the vehicle control policy used by the follower vehicle. 
The adversarial agent controlling the lead vehicle is trained by A2C Reinforcement Learning, maximising a reward function
which incentivizes collisions.

For further details see the paper: https://arxiv.org/abs/2002.12078


## Installation
Clone the repo

```bash
git clone https://github.com/sampo-kuutti/training-adversarial-agents
```

install requirements:
```bash
pip install -r requirements.txt
```

## Training the adversarial models


To train against the IL policy run `train_arl_il_follower.py`, for training against the RL policy run `train_arl_rl_follower.py`


## Citing the Repo

If you find the code useful in your research or wish to cite it, please use the following BibTeX entry.

```text
@inproceedings{kuutti2020training,
  title={Training adversarial agents to exploit weaknesses in deep control policies},
  author={Kuutti, Sampo and Fallah, Saber and Bowden, Richard},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={108--114},
  year={2020},
  organization={IEEE}
}
```