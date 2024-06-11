# CLAT


## Usage

### Installation

Recommended environment:

- python 3.9.7
- pytorch 2.0.1
- torchvision 0.15.2
- lightning 2.1.0

To install the dependencies, run:

```bash
git clone https://github.com/Sorades/CLAT.git
cd CLAT
pip install -r requirements.txt
```

### Training and Testing

```bash
python main.py fit_and_test --config configs/default.yaml --data configs/data/FGADDR.yaml

# test with automatic intervention
python main.py exp_int --config configs/default.yaml --data configs/data/FGADDR.yaml
```

