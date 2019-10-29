# Template-DQN and DRRN

This repository contains reference implementations of TDQN and DRRN as mentioned in [Interactive Fiction Games: A Colossal Adventure](https://arxiv.org/abs/1909.05398).

# Quickstart

Install Dependencies:
```bash
sudo apt-get install redis-server
pip install -U spacy
pip install jericho==2.1.0
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
