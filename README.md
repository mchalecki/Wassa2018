# Wassa2018
Entry for WASSA 2018 Implicit Emotion Shared Task Data

# Setup
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Download dataset.

Wassa dataset is not publicly avalible.
You have to have an account and then download from [wassa official website](http://implicitemotions.wassa2018.com/data/)
Train dataset insert into data/train and test into data/test.

# Training
1. (Optional) Edit params, config in [trainer.py](src/trainer.py)
2. Run training.
```bash
python3 src/train.py
``` 