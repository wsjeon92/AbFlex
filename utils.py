import numpy as np
import torch

vocab = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', '*']

def gram_schmidt(A):
    """Orthogonalize a set of vectors stored as the columns of matrix A."""
    # N-CA-C order
    v1 = A[2]-A[1]
    e1 = v1/np.linalg.norm(v1)
    v2 = A[0]-A[1]
    u2 = v2 - np.dot(e1, v2) * e1
    e2 = u2/np.linalg.norm(u2)
    e3 = np.cross(e1, e2)
    return np.array([e1, e2, e3])

def encoding_res(seq):
    feature=[]    
    for i in seq:
        feature.append(np.eye(len(vocab))[vocab.index(i)])
    return np.asarray(feature)

def masking(mask, cdr_s, cdr_e, coordinates, sequence):
    cdr_s = np.clip(cdr_s, 0, coordinates.shape[0])
    cdr_e = np.clip(cdr_e, 0, coordinates.shape[0])    
    if coordinates is not None:
        coordinates = coordinates.copy()        
        coordinates[cdr_s:cdr_e] = np.linspace(coordinates[cdr_s], coordinates[cdr_e-1], coordinates[cdr_s:cdr_e].shape[0])
    if sequence is not None:
        sequence = sequence.copy()
        for i in mask:
            sequence[i]='*'
    return coordinates, sequence

def gen_mask(seq):
    mask = torch.zeros((len(seq),1))
    for idx, i in enumerate(seq):
        if i=='*':
            mask[idx]=1
    return mask


def RMSD(x1, x2):
    return torch.pow((torch.pow((x1-x2), 2).sum(-1)).mean()+1e-6, 0.5)

def calc_dist(x1, x2):
    return torch.pow(torch.pow((x1-x2), 2).sum(-1)+1e-6, 0.5)


