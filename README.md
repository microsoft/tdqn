# Template-DQN and DRRN

This repository contains reference implementations of TDQN and DRRN as mentioned in [Interactive Fiction Games: A Colossal Adventure](https://arxiv.org/abs/1909.05398).

# Quickstart

Install Dependencies:
```bash
sudo apt-get install redis-server
pip install -U spacy
pip install --user jericho==2.1.0
git clone https://github.com/microsoft/tdqn.git
```

Train TDQN:
```bash
cd tdqn/tdqn && python3 train.py --rom_path <path_to_your_rom_file>
```

Train DRRN:
```bash
cd tdqn/drrn && python3 train.py --rom_path <path_to_your_rom_file>
```

# Citing

If these agents are helpful in your work, please cite the following:

```
@article{hausknecht19colossal,
  title={Interactive Fiction Games: A Colossal Adventure},
  author={Matthew Hausknecht and Prithviraj Ammanabrolu and Marc-Alexandre C{\^{o}}t{\'{e}} and Xingdi Yuan},
  journal={CoRR},
  year={2019},
  volume={abs/1909.05398}
}
```
