import numpy as np
import importlib
import os
import torch
import GPUtil
import ml_collections
import time
import matplotlib.pyplot as plt
import tree
from plotly.subplots import make_subplots

from data.diffuser import Diffuser
from data import utils as du
from model import reverse_diffusion

from experiments import torch_train_diffusion
from analysis import plotting
from analysis import utils as au

from data import pdb_dataset
from data import diffuser
from data import utils as du
from model import reverse_diffusion

from Bio import PDB
import numpy as np

####################################################
######  Parameters Can to Be Adjusted  ##########
ckpt_dir = "./ckpt/"   #path of directory where model pkl saved
pdb_file ="AAAA.pdb"   #your pdb initial structure file
sample_dir = 'model_samples/AAAA'  #output path
target_num =100                #desired number of output structures
t_step = 800 #Adjust the diffusion step for desired noise intensity
####################################################
####################################################

torch.cuda.empty_cache()
torch.cuda.init()
torch.manual_seed(0)
np.random.seed(0)

os.environ["CUDA_DEVICE_ORDER"] = "0"
chosen_gpu = ''.join(
    [str(x) for x in GPUtil.getAvailable(order='memory')])
os.environ["CUDA_VISIBLE_DEVICES2"] = chosen_gpu

ckpt_path = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[0]).replace('.pth', '.pkl')
ckpt_pkl = du.read_pkl(ckpt_path)
ckpt_cfg = ckpt_pkl['cfg']
ckpt_state = ckpt_pkl['exp_state']
data_setting = 'pdb'
cfg = torch_train_diffusion.get_config()
cfg = dict(cfg)
cfg['experiment'].update(ckpt_cfg.experiment)
cfg['experiment']['data_setting'] = data_setting
cfg['model'].update(ckpt_cfg.model)

cfg = ml_collections.ConfigDict(cfg)
cfg['data']['max_len'] = ckpt_cfg.data.max_len
cfg['data']['inpainting_training'] = False
cfg['data']['rmsd_filter'] = None
cfg['data']['monomer_only'] = True

exp_cfg = cfg['experiment']

diffuser = Diffuser() 
parser = PDB.PDBParser()
structure = parser.get_structure("structure_id", pdb_file)

num_res_sample = sum(1 for _ in structure.get_residues())
batch_size = target_num
cfg['experiment']['batch_size'] = batch_size
exp = torch_train_diffusion.Experiment(cfg)
exp.model.load_state_dict(ckpt_state)

os.makedirs(sample_dir, exist_ok=True)
noise_scale = 1.
N = num_res_sample
bb_mask = np.zeros((batch_size, N))
bb_mask[:, :num_res_sample] = 1
my_bb_mask = bb_mask

c4_coords = []
for model in structure:
    for chain in model:
        for residue in chain:
            for atom in residue:
                if atom.get_name() == "C4'":
                    c4_coords.append(atom.get_coord())

diffuser = Diffuser()

c4_coords = np.array(c4_coords)

c4_coords_diffused, noise_t = diffuser.closed_form_forward_diffuse(c4_coords,t_step)


# Set as initial coordinates for reverse diffusion process
initial_coords = c4_coords_diffused[np.newaxis, :, :]

def write_pdb_with_c4_only_transformed_coords(sampled_diffusion, pdb_file, save_path):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure_id", pdb_file)
    
    pdb_c4_lines = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and " C4'" in line:
                pdb_c4_lines.append(line)

    sampled_coords = sampled_diffusion / 10
    assert len(sampled_coords) == len(pdb_c4_lines), "Sampled coordinates do not match C4' atom count"

    new_pdb_content = []
    for line, coords in zip(pdb_c4_lines, sampled_coords):
        new_line = f"{line[:30]}{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}{line[54:]}"
        new_pdb_content.append(new_line)
    
    with open(save_path, 'w') as f:
        f.writelines(line for line in new_pdb_content if line.strip())
    print(f"Saved Structure to {save_path}")

sampled_diffusion = exp.sample_reverse_diffusion(bb_mask=my_bb_mask, initial_coords=initial_coords,rever_step=t_step)
for b_idx in range(batch_size):
    save_path = f'{sample_dir}/valid_len_{num_res_sample}_{b_idx}.pdb'
    write_pdb_with_c4_only_transformed_coords(sampled_diffusion[b_idx][-1], pdb_file, save_path)


