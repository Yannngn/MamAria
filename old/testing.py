from munch import munchify, unmunchify
from yaml import safe_load
with open('config.yaml') as f:
    c = munchify(safe_load(f))

print(unmunchify(c.HYPERPARAMETERS))