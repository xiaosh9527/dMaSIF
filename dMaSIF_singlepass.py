import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from pathlib import Path

# Custom data loader and model:
from data import ProteinPairsSurfaces, PairData, CenterPairAtoms, load_protein_pair
from data import RandomRotationPairAtoms, NormalizeChemFeatures, iface_valid_filter
from model import dMaSIF
from data_iteration import iterate
from helper import *
import argparse

def generate_descr(model_path, output_path, pdb_file, npy_directory, ground_energy = 0.0):
    parser = argparse.ArgumentParser(description="Network parameters")
    parser.add_argument("--experiment_name", type=str, default=model_path)
    parser.add_argument("--use_mesh", type=bool, default=False)
    parser.add_argument("--embedding_layer",type=str,default="dMaSIF")
    parser.add_argument("--curvature_scales",type=list,default=[1.0, 2.0, 3.0, 5.0, 10.0])
    parser.add_argument("--resolution",type=float,default=1.0)
    parser.add_argument("--distance",type=float,default=1.05)
    parser.add_argument("--variance",type=float,default=0.1)
    parser.add_argument("--sup_sampling", type=int, default=100)
    parser.add_argument("--atom_dims",type=int,default=6)
    parser.add_argument("--emb_dims",type=int,default=16)
    parser.add_argument("--in_channels",type=int,default=16)
    parser.add_argument("--orientation_units",type=int,default=16)
    parser.add_argument("--unet_hidden_channels",type=int,default=8)
    parser.add_argument("--post_units",type=int,default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--radius", type=float, default=12.0)
    parser.add_argument("--k",type=int,default=40)
    parser.add_argument("--dropout",type=float,default=0.0)
    parser.add_argument("--site", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--search",type=bool,default=True)
    parser.add_argument("--single_pdb",type=str,default=pdb_file)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random_rotation",type=bool,default=False)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--single_protein",type=bool,default=False)
    parser.add_argument("--no_chem", type=bool, default=False)
    parser.add_argument("--no_geom", type=bool, default=False)
    parser.add_argument("--ground_energy",type=float,default=ground_energy)
    
    args = parser.parse_args("")

    model_path = args.experiment_name
    save_predictions_path = Path(output_path)
    
    # Ensure reproducability:
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Load the train and test datasets:
    transformations = (
        Compose([NormalizeChemFeatures(), CenterPairAtoms(), RandomRotationPairAtoms()])
        if args.random_rotation
        else Compose([NormalizeChemFeatures()])
    )

    if args.single_pdb != "":
        single_data_dir = Path(npy_directory)
        test_dataset = [load_protein_pair(args.single_pdb, single_data_dir,single_pdb=True)]
        test_pdb_ids = [args.single_pdb]

    # PyTorch geometric expects an explicit list of "batched variables":
    batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, follow_batch=batch_vars
    )

    net = dMaSIF(args)
    # net.load_state_dict(torch.load(model_path, map_location=args.device))
    net.load_state_dict(
        torch.load(model_path, map_location=args.device)["model_state_dict"]
    )
    net = net.to(args.device)

    # Perform one pass through the data:
    info = iterate(
        net,
        test_loader,
        None,
        args,
        test=True,
        save_path=save_predictions_path,
        pdb_ids=test_pdb_ids,
    )
    return info
