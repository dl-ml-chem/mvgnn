import numpy as np
import torch
from pymatgen.core.periodic_table import Element


from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def atomic_feature(z):
    e = Element.from_Z(int(z))
    return [
        z,
        e.X if e.X else 0,
        float(e.atomic_mass),
        e.row,
        e.group if e.group else 0
    ]


_SGA_CACHE = {}


def augment_structure(structure, method="rattle", stdev=0.02):
    """
    Data augmentation for crystal structures.
    
    Parameters
    ----------
    method : "rattle" or "symmetry"
    stdev : perturbation intensity for rattle
    """
    if method == "rattle":
        aug_struct = structure.copy()
        aug_struct.perturb(stdev)
        return aug_struct
        
    elif method == "symmetry":
        # Apply a random symmetry operation from the space group
        struct_id = id(structure)
        if struct_id in _SGA_CACHE:
            ops = _SGA_CACHE[struct_id]
        else:
            sga = SpacegroupAnalyzer(structure)
            ops = sga.get_symmetry_operations()
            _SGA_CACHE[struct_id] = ops
            
        import random
        op = random.choice(ops)
        
        # Apply symmetry op to fractional coordinates
        new_frac_coords = [op.operate(site.frac_coords) for site in structure]
        aug_struct = structure.copy()
        for i, coords in enumerate(new_frac_coords):
            aug_struct.replace(i, structure[i].species, coords, coords_are_cartesian=False)
        return aug_struct
        
    return structure


class GaussianExpansion:
    def __init__(self, start=0.0, stop=6.0, num=16):
        self.filters = np.linspace(start, stop, num)
        self.var = (stop - start) / num

    def expand(self, d):
        if isinstance(d, np.ndarray):
            return np.exp(-((d[:, np.newaxis] - self.filters) ** 2) / (self.var ** 2))
        return np.exp(-((d - self.filters) ** 2) / (self.var ** 2))


def structure_to_graph(structure, cutoff=4.5):
    # Correct handling of PBC using pymatgen
    all_neighbors = structure.get_all_neighbors(cutoff)

    # species = [site.specie.Z for site in structure]
    species = [list(site.species.keys())[0].Z for site in structure]

    # Base atomic features
    base_features = np.array([atomic_feature(z) for z in species])

    node_angles_mean = []
    node_angles_std = []
    coordination_numbers = []

    edge_index = []
    all_dists = []

    ge = GaussianExpansion(stop=cutoff, num=16)

    for i, neighbors in enumerate(all_neighbors):
        # neighbors is a list of [neighbor_site, distance, index, image]

        # 1. Coordination Number
        num_neighbors = len(neighbors)
        coordination_numbers.append(num_neighbors)

        # 2. Bond Angles for node i (Vectorized)
        if num_neighbors > 1:
            neighbor_coords = np.array([n.coords for n in neighbors])
            center_coords = structure[i].coords
            vectors = neighbor_coords - center_coords # (K, 3)
            
            # Vectorized dot products and norms
            dot_products = np.dot(vectors, vectors.T) # (K, K)
            norms = np.linalg.norm(vectors, axis=1) # (K,)
            norms[norms == 0] = 1e-8 # Safety
            
            cos_theta = dot_products / np.outer(norms, norms)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            
            # Get upper triangle indices (excluding diagonal)
            iu = np.triu_indices(num_neighbors, k=1)
            angles = np.arccos(cos_theta[iu])
            
            node_angles_mean.append(np.mean(angles))
            node_angles_std.append(np.std(angles))
        else:
            node_angles_mean.append(0.0)
            node_angles_std.append(0.0)

        # 3. Collect edges for later vectorized expansion
        for neighbor in neighbors:
            edge_index.append([i, neighbor.index])
            all_dists.append(neighbor.nn_distance)

    # Combine all node features: atomic(5) + angle_mean(1) + angle_std(1) + coord(1) = 8 dims
    x = np.column_stack([
        base_features,
        node_angles_mean,
        node_angles_std,
        coordination_numbers
    ])
    x = torch.tensor(x, dtype=torch.float)

    if len(edge_index) == 0:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.tensor([ge.expand(0.0)], dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(ge.expand(np.array(all_dists)), dtype=torch.float)

    return x, edge_index, edge_attr