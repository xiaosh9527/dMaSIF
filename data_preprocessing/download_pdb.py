from Bio.PDB import * 
from Bio.SeqUtils import IUPACData
import sys
import importlib
import os
import numpy as np
from subprocess import Popen, PIPE
from pathlib import Path
from .convert_pdb2npy import load_structure_np

if len(sys.argv) <= 1: 
    print("Usage: "+sys.argv[0]+" PDBID_A_B")
    print("A or B are the chains to include in this pdb.")
    sys.exit(1)

PROTEIN_LETTERS = [x.upper() for x in IUPACData.protein_letters_3to1.keys()]

# Exclude disordered atoms.
class NotDisordered(Select):
    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() == "A"  or atom.get_altloc() == "1" 


def find_modified_amino_acids(path):
    """
    Contributed by github user jomimc - find modified amino acids in the PDB (e.g. MSE)
    """
    res_set = set()
    for line in open(path, 'r'):
        if line[:6] == 'SEQRES':
            for res in line.split()[4:]:
                res_set.add(res)
    for res in list(res_set):
        if res in PROTEIN_LETTERS:
            res_set.remove(res)
    return res_set


def extractPDB(
    infilename, outfilename, chain_ids=None 
):
    # extract the chain_ids from infilename and save in outfilename. 
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(infilename, infilename)
    model = Selection.unfold_entities(struct, "M")[0]
    chains = Selection.unfold_entities(struct, "C")
    # Select residues to extract and build new structure
    structBuild = StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()

    # Load a list of non-standard amino acid names -- these are
    # typically listed under HETATM, so they would be typically
    # ignored by the orginal algorithm
    modified_amino_acids = find_modified_amino_acids(infilename)

    for chain in model:
        if (
            chain_ids == None
            or chain.get_id() in chain_ids
        ):
            structBuild.init_chain(chain.get_id())
            for residue in chain:
                het = residue.get_id()
                if het[0] == " ":
                    outputStruct[0][chain.get_id()].add(residue)
                elif het[0][-3:] in modified_amino_acids:
                    outputStruct[0][chain.get_id()].add(residue)

    # Output the selected residues
    pdbio = PDBIO()
    pdbio.set_structure(outputStruct)
    pdbio.save(outfilename, select=NotDisordered())

def protonate(in_pdb_file, out_pdb_file):
    # protonate (i.e., add hydrogens) a pdb using reduce and save to an output file.
    # in_pdb_file: file to protonate.
    # out_pdb_file: output file where to save the protonated pdb file. 
    
    # Remove protons first, in case the structure is already protonated
    args = ["reduce", "-Trim", in_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8').rstrip())
    outfile.close()
    # Now add them again.
    args = ["reduce", "-HIS", out_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8'))
    outfile.close()


def convert_to_npy(pdb_id, pdb_dir, npy_dir, tmp_dir, chains = "auto"):
    protonated_file = pdb_dir/f"{pdb_id}.pdb"
    if not protonated_file.exists():
        # Download pdb 
        pdbl = PDBList()
        pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=tmp_dir,file_format='pdb')

        ##### Protonate with reduce, if hydrogens included.
        # - Always protonate as this is useful for charges. If necessary ignore hydrogens later.
        protonate(pdb_filename, protonated_file)

    pdb_filename = protonated_file
    if chains == "auto":
        chains = pdb_id.split("_")[1:]
        
    pdb_id = pdb_id.split("_")[0]
    # Extract chains of interest.
    for chain in chains:
        # assert len(chain) == 1,\
        #     "Chain id must be a single letter."

        out_filename = pdb_dir/f"{pdb_id}_{chain}.pdb"
        extractPDB(pdb_filename, str(out_filename), chain)
        protein = load_structure_np(out_filename,center=False)
        np.save(npy_dir / f"{pdb_id}_{chain}_atomxyz", protein["xyz"])
        np.save(npy_dir / f"{pdb_id}_{chain}_atomtypes", protein["types"])