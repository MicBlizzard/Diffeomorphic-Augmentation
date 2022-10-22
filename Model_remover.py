import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, required=True)
parser.add_argument("--dest", type=str, required=True)

args = parser.parse_args()

model_state = torch.load(args.source)

for i,c in enumerate(model_state,0):
    new_model_state = {}
    for key in c[0].keys():
        new_model_state[key[7:]] = c[0][key]
    torch.save(new_model_state, args.dest+str(i+1)+".pt")
