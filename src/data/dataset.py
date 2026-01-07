import io
import random
import warnings
import torch
import webdataset as wds

import json
import pickle

from pathlib import Path
from torch.utils.data import Dataset

from src.data.data_utils import TensorDict, collate_entity
from src.constants import WEBDATASET_SHARD_SIZE, WEBDATASET_VAL_SIZE

from src.constants import atom_encoder,bond_encoder
from src.data.data_utils import prepare_ligand

from collections import defaultdict
from tqdm import tqdm

from scipy.spatial.transform import Rotation

class ProcessedLigandPocketDataset(Dataset):
    def __init__(self, pt_path, geom_path=None, ligand_transform=None, pocket_transform=None,
                 catch_errors=False):

        self.ligand_transform = ligand_transform
        self.pocket_transform = pocket_transform
        self.catch_errors = catch_errors
        self.pt_path = pt_path

        self.ligand_data_path = Path(pt_path.as_posix().replace('.pt','_ligands.pt'))
        
        self.ligand_data = defaultdict(lambda:[])
        if geom_path is not None and not self.ligand_data_path.exists():
            self.geom_path = geom_path
            summary_file = geom_path.joinpath('summary_drugs.json')
            assert summary_file.exists(), f"GEOM summary file not found at {summary_file}"
            with open(summary_file,'r') as f:
                geom_drugs_summ = json.load(f)
                
            list_smiles = tqdm(list(geom_drugs_summ.keys()))
            for smiles in list_smiles:
                try:
                    pickle_path = Path(geom_drugs_summ[smiles]['pickle_path'])
                    pickle_path = self.geom_path.joinpath(pickle_path)

                    with open(pickle_path,'rb') as f:
                        mol_data = pickle.load(f)

                    conformers = mol_data['conformers']
                    bolzmann_weights = [cfm['boltzmannweight'] for cfm in conformers]
                    # target_conformer = random.choices(conformers,weights=bolzmann_weights,k=1)[0]
                    target_conformer = conformers[bolzmann_weights.index(max(bolzmann_weights))]
                    
                    rdmol = target_conformer['rd_mol']
                    ligand = prepare_ligand(rdmol,atom_encoder,bond_encoder)
                    for k in ligand.keys():
                        self.ligand_data[k].append(ligand[k])
                    self.ligand_data['name'].append(smiles)
                    
                except Exception as e:
                    warnings.warn(f"Failed to process GEOM data for {smiles}: {e}")
            torch.save(dict(self.ligand_data),self.ligand_data_path)
        elif geom_path is not None and self.ligand_data_path.exists():    
            self.ligand_data = torch.load(self.ligand_data_path, weights_only=False)
            
        self.ligand_pocket_data = torch.load(pt_path)
        print(pt_path)
        # add number of nodes for convenience
        for entity in ['ligands', 'pockets']:
            self.ligand_pocket_data[entity]['size'] = torch.tensor([len(x) for x in self.ligand_pocket_data[entity]['x']])
            self.ligand_pocket_data[entity]['n_bonds'] = torch.tensor([len(x) for x in self.ligand_pocket_data[entity]['bond_one_hot']])

        self.ligand_pocket_size = len(self.ligand_pocket_data['pockets']['x'])
        self.pocket_keys = self.ligand_pocket_data['pockets'].keys()
        
    def __len__(self):
        return len(self.ligand_pocket_data['ligands']['name']) + len(self.ligand_data['name'])

    def __getitem__(self, idx):
        # idx = 100000 # for debug
        data = {}
        random_rotquat = torch.randn(4)
        random_rot = Rotation(random_rotquat)
        
        if idx < self.ligand_pocket_size:
            data['ligand'] = {key: val[idx] for key, val in self.ligand_pocket_data['ligands'].items()}
            data['ligand']['x'] = torch.tensor(random_rot.apply(data['ligand']['x']))
            
            data['pocket'] = {key: val[idx] for key, val in self.ligand_pocket_data['pockets'].items()}
            data['pocket']['x'] = torch.tensor(random_rot.apply(data['pocket']['x']))
            data['pocket']['v'] = torch.tensor(random_rot.apply(data['pocket']['v'].reshape(-1,3))).reshape(*data['pocket']['v'].shape)
            data['pocket']['fixed_coord'] = torch.tensor(random_rot.apply(data['pocket']['fixed_coord'].reshape(-1,3))).reshape(*data['pocket']['fixed_coord'].shape)
            
            
            try:
                if self.ligand_transform is not None:
                    data['ligand'] = self.ligand_transform(data['ligand'])
                if self.pocket_transform is not None:
                    data['pocket'] = self.pocket_transform(data['pocket'])
            except (RuntimeError, ValueError) as e:
                if self.catch_errors:
                    warnings.warn(f"{type(e).__name__}('{e}') in data transform. "
                                f"Returning random item instead")
                    # replace bad item with a random one
                    rand_idx = random.randint(0, len(self) - 1)
                    return self[rand_idx]
                else:
                    raise e
        else:
            idx = idx - self.ligand_pocket_size
            data['ligand'] = {key: val[idx] for key, val in self.ligand_data.items()}
            data['ligand']['x'] = torch.tensor(random_rot.apply(data['ligand']['x']))
            
            data['pocket'] = {key: [] for key in self.pocket_keys}
            data['pocket']['is_existing'] = False
            data['pocket']['mask'] = torch.tensor([])
            try:
                if self.ligand_transform is not None:
                    data['ligand'] = self.ligand_transform(data['ligand'])
                if self.pocket_transform is not None:
                    data['pocket'] = self.pocket_transform(data['pocket'])
            except (RuntimeError, ValueError) as e:
                if self.catch_errors:
                    warnings.warn(f"{type(e).__name__}('{e}') in data transform. "
                                f"Returning random item instead")
                    # replace bad item with a random one
                    rand_idx = random.randint(0, len(self) - 1)
                    return self[rand_idx]
                else:
                    raise e  
        return data

    @staticmethod
    def collate_fn(batch_pairs, ligand_transform=None, use_scaffold=False):

        out = {}
        column_list = ['ligand', 'pocket']
        if use_scaffold:
            column_list.append('scaffold')
        
        for entity in column_list:
            batch = [x[entity] for x in batch_pairs]

            if entity == 'ligand' and ligand_transform is not None:
                max_size = max(x['size'].item() for x in batch)
                # TODO: might have to remove elements from batch if processing fails, warn user in that case
                batch = [ligand_transform(x, max_size=max_size) for x in batch]

            out[entity] = TensorDict(**collate_entity(batch))

        return out


class ClusteredDataset(ProcessedLigandPocketDataset):
    def __init__(self, pt_path, ligand_transform=None, pocket_transform=None,
                 catch_errors=False):
        super().__init__(pt_path, ligand_transform, pocket_transform, catch_errors)
        self.clusters = list(self.ligand_pocket_data['clusters'].values())

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, cidx):
        cluster_inds = self.clusters[cidx]
        # idx = cluster_inds[random.randint(0, len(cluster_inds) - 1)]
        idx = random.choice(cluster_inds)
        return super().__getitem__(idx)

class DPODataset(ProcessedLigandPocketDataset):
    def __init__(self, pt_path, ligand_transform=None, pocket_transform=None,
                 catch_errors=False):
        self.ligand_transform = ligand_transform
        self.pocket_transform = pocket_transform
        self.catch_errors = catch_errors
        self.pt_path = pt_path

        self.data = torch.load(pt_path)

        if not 'pockets' in self.data:
            self.data['pockets'] = self.data['pockets_w']
        if not 'ligands' in self.data:
            self.data['ligands'] = self.data['ligands_w']

        if (
            len(self.data["ligands"]["name"])
            != len(self.data["ligands_l"]["name"])
            != len(self.data["pockets"]["name"])
        ):
            raise ValueError(
                "Error while importing DPO Dataset: Number of ligands winning, ligands losing and pockets must be the same"
            )

        # add number of nodes for convenience
        for entity in ['ligands', 'ligands_l', 'pockets']:
            self.data[entity]['size'] = torch.tensor([len(x) for x in self.data[entity]['x']])
            self.data[entity]['n_bonds'] = torch.tensor([len(x) for x in self.data[entity]['bond_one_hot']])

    def __len__(self):
        return len(self.data["ligands"]["name"])

    def __getitem__(self, idx):
        data = {}
        data['ligand'] = {key: val[idx] for key, val in self.data['ligands'].items()}
        data['ligand_l'] = {key: val[idx] for key, val in self.data['ligands_l'].items()}
        data['pocket'] = {key: val[idx] for key, val in self.data['pockets'].items()}
        try:
            if self.ligand_transform is not None:
                data['ligand'] = self.ligand_transform(data['ligand'])
                data['ligand_l'] = self.ligand_transform(data['ligand_l'])
            if self.pocket_transform is not None:
                data['pocket'] = self.pocket_transform(data['pocket'])
        except (RuntimeError, ValueError) as e:
            if self.catch_errors:
                warnings.warn(f"{type(e).__name__}('{e}') in data transform. "
                              f"Returning random item instead")
                # replace bad item with a random one
                rand_idx = random.randint(0, len(self) - 1)
                return self[rand_idx]
            else:
                raise e
        return data
    
    @staticmethod
    def collate_fn(batch_pairs, ligand_transform=None):

        out = {}
        for entity in ['ligand', 'ligand_l', 'pocket']:
            batch = [x[entity] for x in batch_pairs]

            if entity in ['ligand', 'ligand_l'] and ligand_transform is not None:
                max_size = max(x['size'].item() for x in batch)
                batch = [ligand_transform(x, max_size=max_size) for x in batch]

            out[entity] = TensorDict(**collate_entity(batch))

        return out

##########################################
############### WebDatasets ##############
##########################################

class ProteinLigandWebDataset(wds.WebDataset):
    @staticmethod
    def collate_fn(batch_pairs, ligand_transform=None):
        return ProcessedLigandPocketDataset.collate_fn(batch_pairs, ligand_transform)


def wds_decoder(key, value):
    return torch.load(io.BytesIO(value))


def preprocess_wds_item(data):
    out = {}
    for entity in ['ligand', 'pocket']:
        out[entity] = data['pt'][entity]
        for attr in ['size', 'n_bonds']:
            if torch.is_tensor(out[entity][attr]):
                assert len(out[entity][attr]) == 0
                out[entity][attr] = 0

    return out


def get_wds(data_path, stage, ligand_transform=None, pocket_transform=None):
    current_data_dir = Path(data_path, stage)
    shards = sorted(current_data_dir.glob('shard-?????.tar'), key=lambda s: int(s.name.split('-')[-1].split('.')[0]))
    min_shard = min(shards).name.split('-')[-1].split('.')[0]
    max_shard = max(shards).name.split('-')[-1].split('.')[0]
    total_size = (int(max_shard) - int(min_shard) + 1) * WEBDATASET_SHARD_SIZE if stage == 'train' else WEBDATASET_VAL_SIZE

    url = f'{data_path}/{stage}/shard-{{{min_shard}..{max_shard}}}.tar'
    ligand_transform_wrapper = lambda _data: _data
    pocket_transform_wrapper = lambda _data: _data

    if ligand_transform is not None:
        def ligand_transform_wrapper(_data):
            _data['pt']['ligand'] = ligand_transform(_data['pt']['ligand'])
            return _data
        
    if pocket_transform is not None:
        def pocket_transform_wrapper(_data):
            _data['pt']['pocket'] = pocket_transform(_data['pt']['pocket'])
            return _data

    return (
        ProteinLigandWebDataset(url, nodesplitter=wds.split_by_node)
        .decode(wds_decoder)
        .map(ligand_transform_wrapper)
        .map(pocket_transform_wrapper)
        .map(preprocess_wds_item)
        .with_length(total_size)
    )
