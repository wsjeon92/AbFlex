import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats

from Bio.PDB.PDBParser import PDBParser
from Bio import BiopythonWarning

import json
import os
import sys
import urllib.request
import pickle
import warnings
warnings.filterwarnings("ignore")
from collections import OrderedDict
import gc
import copy
import subprocess
import time
import io

from pdbfixer import PDBFixer
from openmm.app import *
from openmm import *
from openmm.unit import *
import pymol

from utils import *
from AbFlex import *

# load configuration file
with open('./config.json') as f:
    config = json.load(f)

pdb_id = config["pdb_id"]
ab_chain_list = config["ab_chain_list"]
ag_chain_list =config["ag_chain_list"]
D_chain = config["design_chain"]
cdr_type = config["cdr_type"]
cdr_seq = config["cdr_seq"]
out_dir = config["out_dir"]
FoldX_dir = config["FoldX_dir"]
IA_dir = config["IA_dir"]
n_sample = config["n_sample"]
n_relax = config["n_relax"]

print('configuration file loaded')
print(config)

# generate input data with the given configuration file.
vocab = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', '*']
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = ['A',  'R',  'N',  'D',  'C',  'Q',  'E',  'G',  'H',  'I',  'L',  'K',  'M',  'F',  'P',  'S',  'T', 'W', 'Y',  'V']
d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

dataset = []
structure_id = pdb_id
filename = structure_id.lower()+".pdb"
cdr_len = len(cdr_seq)
parser = PDBParser(PERMISSIVE=1)

## fetch the original pdb if not exist.
os.system(f'rm {out_dir}{pdb_id}.pdb')
try:
    if not os.path.isfile(f"{out_dir}{pdb_id}.pdb"):
        pymol.cmd.set('fetch_path', pymol.cmd.exp_path(out_dir), quiet=0)        
        pymol.cmd.fetch(f"{pdb_id}", type="pdb")
        print(f"{pdb_id.lower()} is fetched @ {out_dir}")
except:
    print(f'cannot fetch {pdb_id} as PDB format from the database')
    sys.exit()

## load pdb file with biopython
with warnings.catch_warnings():
    try:
        warnings.simplefilter('ignore', BiopythonWarning)
        structure = parser.get_structure(structure_id, out_dir+filename)
        model = structure[0]
    except: print('Unable to load the PDB file'); sys.exit()
    
for c_ab in ab_chain_list:
    if c_ab!=D_chain: continue
    try:
        model = structure[0]
        chain_ab = model[c_ab]
    except:
        print('error occured during loading the antibody chain using biopython')
        sys.exit()     

    ### define residue id
    ab_seq_id = list(chain_ab.child_dict.keys())
    ab_seq_id = [i for i in ab_seq_id if i[0]==' '] # remove hetero atoms

    # check antibody backbone atoms whether there are missing backbone atoms
    temp_seq_idx = []
    for i, datai in enumerate(ab_seq_id):
        try:
            chain_ab[datai]["N"]
            chain_ab[datai]["CA"]
            chain_ab[datai]["C"]
            chain_ab[datai]["O"]
            temp_seq_idx.append(datai)
        except:
            continue
    ab_seq_id = temp_seq_idx

    if len(ab_seq_id)>300 or len(ab_seq_id)<80:
        print('antibody length is out of the standard range. please check the input antibody chain.')

    coor_ab = {'coor':[], 'seq':[], 'idx':[], 'ori':[]} 
    coor_ag = {}
    info_cdr = {}

    for idx, residue in enumerate(chain_ab):
        if residue.get_id()[0] != ' ' or not residue.get_resname() in vocab: continue       
        try:
            chain_ab[residue.full_id[-1]]["N"]
            chain_ab[residue.full_id[-1]]["CA"]
            chain_ab[residue.full_id[-1]]["C"]
            chain_ab[residue.full_id[-1]]["O"]
        except:       
            continue                  
        for atom in residue:
            if atom.get_name() in ['CA']:
                coor_ab['coor'].append(atom.get_coord())
                coor_ab['seq'].append(residue.get_resname())    
                coor_ab['idx'].append(idx)
                coor_ab['ori'].append(gram_schmidt([residue['N'].coord, residue['CA'].coord, residue['C'].coord]))    

    coor_ab['coor'] = np.vstack(coor_ab['coor'])

    check_cdr=[]                
    for c_ag in ag_chain_list+ab_chain_list:        
        if c_ag==c_ab: continue
        try:
            chain_ag = model[c_ag]
        except:
            continue    

        ag_seq_id = list(chain_ag.child_dict.keys())
        ag_seq_id = [i for i in ag_seq_id if i[0]==' '] # remove hetero atoms

        # check antigne backbone atoms        
        temp_seq_idx = []        
        for j, dataj in enumerate(ag_seq_id):
            try:
                chain_ag[dataj]["N"]
                chain_ag[dataj]["CA"]
                chain_ag[dataj]["C"]
                chain_ag[dataj]["O"]
                temp_seq_idx.append(dataj)                
            except:       
                continue
        ag_seq_id = temp_seq_idx

        ### backbone atoms inter-distance between chains ========================
        AbAg_dist = np.zeros((4, len(ab_seq_id), len(ag_seq_id)))
        for i, datai in enumerate(ab_seq_id):
            for j, dataj in enumerate(ag_seq_id):
                AbAg_dist[0, i,j] = chain_ab[datai]["N"]-chain_ag[dataj]["N"]
                AbAg_dist[1, i,j] = chain_ab[datai]["CA"]-chain_ag[dataj]["CA"]
                AbAg_dist[2, i,j] = chain_ab[datai]["C"]-chain_ag[dataj]["C"]
                AbAg_dist[3, i,j] = chain_ab[datai]["O"]-chain_ag[dataj]["O"]

        if AbAg_dist.shape[-1]==0: continue      

        coor_ag_temp = {'coor':[], 'seq':[], 'idx':[], 'ori':[]}
        for chain in model:
            if chain.get_id()!=c_ag: continue
            if chain.get_id() in ag_chain_list:
                is_ag = 1
            else: is_ag = 0     
            res_idx=0
            for idx, residue in enumerate(chain):
                if residue.get_id()[0] != ' ' or not residue.get_resname() in vocab: continue     
                try:
                    chain[residue.full_id[-1]]["N"]
                    chain[residue.full_id[-1]]["CA"]
                    chain[residue.full_id[-1]]["C"]
                    chain[residue.full_id[-1]]["O"]
                except:       
                    continue                    
                if np.min(AbAg_dist[1, :, res_idx])<16:
                    for atom in residue:
                        if atom.get_name() in ['CA']:
                            coor_ag_temp['coor'].append(atom.get_coord())
                            coor_ag_temp['seq'].append(residue.get_resname())                                                        
                            coor_ag_temp['idx'].append(idx)
                            coor_ag_temp['ori'].append(gram_schmidt([residue['N'].coord, residue['CA'].coord, residue['C'].coord]))                                 
                res_idx+=1
        if coor_ag_temp['coor'] == []:
            continue                            
        coor_ag_temp['coor'] = np.vstack(coor_ag_temp['coor'])
        coor_ag[c_ag] = (is_ag, coor_ag_temp)

        try:
            ab_chain_seq = ''.join([idx2char[char2idx[chain_ab[i].get_resname()]] for i in ab_seq_id])
        except:
            print('wrong res name in seq')
            continue                
        cdr_idx_start = ab_chain_seq.find(cdr_seq)
        if cdr_idx_start == -1: print(structure_id, c_ab, c_ag, 'cdr s not found.'); print(ab_chain_seq); print(ab_seq_id); continue                 
        try: 
            abag_dist = (AbAg_dist[1,cdr_idx_start:cdr_idx_start+cdr_len,:]).min()
        except: continue
        if abag_dist>16: continue
        if cdr_type in info_cdr.keys():
            cdr_ag_list = info_cdr[cdr_type][3]+[c_ag]
            abag_dist = info_cdr[cdr_type][-1]+[abag_dist]
            info_cdr[cdr_type] = (structure_id, c_ab, cdr_type, cdr_ag_list, np.arange(cdr_idx_start, cdr_idx_start+cdr_len), abag_dist)
        else:
            info_cdr[cdr_type] = (structure_id, c_ab, cdr_type, [c_ag], np.arange(cdr_idx_start, cdr_idx_start+cdr_len), [abag_dist])
        if len(info_cdr[cdr_type][3]) != len(info_cdr[cdr_type][-1]): print(info_cdr[cdr_type][3], info_cdr[cdr_type][-1])

    if coor_ag:
        info_cdr = OrderedDict(sorted(info_cdr.items()))
        dataset.append((coor_ab, coor_ag, info_cdr))

torch.save(dataset, f'{out_dir}input_{pdb_id}_{D_chain}_{cdr_type}')
print('Input data preprocessing done.')


# predict CDR with AbFlex
n_hid= 256
n_pos_emb=32
n_layer=16
model_gnn = AbFlex(n_hid+2, n_hid, n_layer, 64, 16).cuda()
model_pred = pred_seq(n_hid+2, 20).cuda()

model_dir = "./AbFlex_pretrained.pt"

try:
    checkpoint = torch.load(model_dir)
    model_gnn.load_state_dict(checkpoint['model_gnn'])
    model_pred.load_state_dict(checkpoint['model_pred'])
    print(PATH, 'loaded')
except:
    print('pre-trained model is not loaded.')

#load input data
input_data=torch.load(f'{out_dir}input_{pdb_id}_{D_chain}_{cdr_type}')
parser = PDBParser(PERMISSIVE=1)

model_gnn.eval()
model_pred.eval()
start_time = time.time()
with torch.no_grad():
    rec_info = {'H1':[], 'H2':[], 'H3':[], 'L1':[], 'L2':[], 'L3':[]}
    rec_coor = {'H1':[], 'H2':[], 'H3':[], 'L1':[], 'L2':[], 'L3':[]}
    rec_ab_len = {'H1':[], 'H2':[], 'H3':[], 'L1':[], 'L2':[], 'L3':[]}
    rec_seq_all = {'H1':[], 'H2':[], 'H3':[], 'L1':[], 'L2':[], 'L3':[]}
    rec_label_all = {'H1':[], 'H2':[], 'H3':[], 'L1':[], 'L2':[], 'L3':[]}  
    rec_chains = {'H1':[], 'H2':[], 'H3':[], 'L1':[], 'L2':[], 'L3':[]}
    
    for idx_data, data in enumerate(input_data):
        str_ab, str_ag, info_cdr = data
        coor_ab= str_ab['coor'].copy()
        for cdr in info_cdr.values():         
            cdr_s=cdr[-2][0]-1
            cdr_e=cdr[-2][-1]+2
            cdr_s2=cdr[-2][0]
            cdr_e2=cdr[-2][-1]+1

            temp_ab_chains=[]
            temp_ag_chains=[]
            input_coor, input_seq = masking(cdr[-2], cdr_s, cdr_e, coor_ab, str_ab['seq']) #cdr[0] is type of CDR  
            input_coor_unmasked=copy.deepcopy(coor_ab)
            input_chain = np.zeros((len(str_ab['seq'])))
            temp_ab_chains.append(cdr[1])
            input_coor_ag = []
            for cdr_c in cdr[3]:
                is_ag, temp_ag = str_ag[cdr_c]
                input_coor = np.concatenate((input_coor, temp_ag['coor']), axis=0)
                input_coor_unmasked = np.concatenate((input_coor_unmasked, temp_ag['coor']), axis=0)
                input_seq += temp_ag['seq']
                if is_ag:
                    input_chain = np.concatenate((input_chain, np.ones((len(temp_ag['seq'])))))
                    temp_ag_chains.append(cdr_c)
                else:
                    input_chain = np.concatenate((input_chain, np.zeros((len(temp_ag['seq'])))))
                    temp_ab_chains.append(cdr_c)
                if is_ag==1:
                    try:
                        input_coor_ag = np.concatenate((input_coor_ag, temp_ag['coor']), axis=0)
                    except:
                        input_coor_ag = temp_ag['coor']
            
            label = str_ab['seq'].copy()
            for cdr_c in cdr[3]:
                is_ag, temp_ag = str_ag[cdr_c]
                label += temp_ag['seq']                    
            label = encoding_res(label).argmax(-1)

            input_mask = gen_mask(input_seq)

            output_feat, output_coor = model_gnn((input_seq, input_chain), input_mask, input_coor)
            output_seq = model_pred(output_feat[-1])
            
            rec_seq_all[cdr[2]].append(output_seq.detach().cpu().numpy())
            rec_label_all[cdr[2]].append(label)
            rec_info[cdr[2]].append(cdr)
            rec_ab_len[cdr[2]].append(coor_ab.shape[0])
            rec_coor[cdr[2]].append((output_coor.detach().cpu().numpy(), input_coor_unmasked))
            rec_chains[cdr[2]].append([temp_ab_chains, temp_ag_chains])

print('CDR prediction done.')
gc.collect()
torch.cuda.empty_cache()

print('Building a full atom model is started')
idx=0
idxx=0
cdr_id = cdr_type

print(rec_info[cdr_id][idx][0].lower(), 'reconstruction started.')
rec_BE_foldx=[]
rec_BE_ia=[]
seq_cat=[]
file_list = []
interface_chains=','.join([''.join(i) for i in rec_chains[cdr_id][idx]])
ab_chain = rec_info[cdr_id][idx][1]

# fetch the original pdb if not exist.
pdb_id=rec_info[cdr_id][idx][0].lower()

# remove the "TER" within the same chain. This makes error when use InterfaceAnalyzer.
with open (f"{out_dir}{rec_info[cdr_id][idx][0].lower()}.pdb", 'r') as f:
    lines = f.readlines()
with open (f"{out_dir}{rec_info[cdr_id][idx][0].lower()}.pdb", 'w') as f:
    for l_idx, line in enumerate(lines[1:]):
        previous_chain = lines[l_idx-1][13]
        if line.split()[0]=="TER" and line[13]==previous_chain:
            continue
        f.write(line)

## remove ANISOU and hetero atoms
pymol.cmd.load(out_dir+pdb_id+'.pdb')
pymol.cmd.select('hetero_atoms', 'het')
pymol.cmd.remove('hetero_atoms')
pymol.cmd.save(out_dir+pdb_id+'.pdb')
subprocess.run(f"grep -v 'ANISOU' {out_dir}{pdb_id}.pdb > {out_dir}{pdb_id}_no_anisou.pdb", shell=True)
subprocess.run(f"mv {out_dir}{pdb_id}_no_anisou.pdb {out_dir}{pdb_id}.pdb", shell=True)


# sampling n different CDR sequences from the predicted probability distribution.
while len(seq_cat)<n_sample+1:
    seq_temp=[]
    for i in range(rec_ab_len[cdr_id][idx]):
        if i in rec_info[cdr_id][idx][-2]:
            seq_temp.append(vocab[torch.multinomial(torch.softmax(torch.from_numpy(rec_seq_all[cdr_id][idx][i]), -1), 1, replacement=True)])
        else:
            seq_temp.append(vocab[rec_label_all[cdr_id][idx][i]])
    if not seq_temp in seq_cat:
        seq_cat.append(seq_temp)

with open(f"{out_dir}{pdb_id}_{D_chain}_{cdr_type}_sequence_samples.fasta", 'w') as f:
    for i in range(n_sample):
        f.write(f'>{pdb_id}_{D_chain}_{cdr_type}_sample{i}\n')
        f.write(''.join([d3to1[j] for j in seq_cat[i]])+'\n')

# write PDB files with the predicted cdr coordinates and the sampled sequences.
for j in range(n_sample):
    with open(f"{out_dir}{'_'.join(rec_info[cdr_id][idx][:3])}_{j}.pdb", "w") as f:
        structure = parser.get_structure(pdb_id, out_dir+pdb_id+'.pdb')
        i=0
        models_idx = list(structure.child_dict.keys())
        for m in models_idx:
            model = structure[m]        
            for chain in model:
                if chain.id == ab_chain:
                    for r_idx, residue in enumerate(chain):
                        if r_idx not in rec_info[cdr_id][idx][-2]:
                            for atom in residue:
                                f.write('ATOM{:>7}  {:<3}{:>4}{:>6}    {:>8.3f}{:>8.3f}{:>8.3f}  1.00                 {}\n'.format(i,atom.id, residue.get_resname(), r_idx, atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2],atom.element))                 
                                i+=1
                        else:
                            f.write('ATOM{:>7}  {:<3}{:>4}{:>6}    {:>8.3f}{:>8.3f}{:>8.3f}  1.00                 {}\n'.format(i,'CA', seq_cat[j][r_idx], r_idx, rec_coor[cdr_id][idx][0][-1][r_idx,0], rec_coor[cdr_id][idx][0][-1][r_idx,1], rec_coor[cdr_id][idx][0][-1][r_idx,2],'C'))
                            i+=1
                else:
                    continue
        f.write('TER')            
    print(f"{out_dir}{'_'.join(rec_info[cdr_id][idx][:3])}_{j}.pdb was saved")
    file_list.append(f"{'_'.join(rec_info[cdr_id][idx][:3])}_{j}")

# build full-atom models
print(file_list)

for file_name in file_list:
    score_list_foldx, score_list_ia = [], []
    best_foldx, best_ia = 1e6, 1e6
    for k in range(n_relax):
        print(idxx, file_name, 'merging')
        pymol.cmd.delete('all')                    
        pymol.cmd.load(out_dir+file_name+'.pdb')
        pymol.cmd.alter(file_name, f"chain='{file_name.split('_')[1]}'")
        pymol.cmd.load(out_dir+file_name.split('_')[0].lower()+'.pdb')
        pymol.cmd.remove('hydrogens')
        pymol.cmd.remove('resn hoh')
        pymol.cmd.remove(file_name.split('_')[0].lower()+' and chain '+file_name.split('_')[1]) # remove antibody chain in the original pdb

        pymol.cmd.select('non_bind_chains', f'{file_name.split("_")[0].lower()} and not (chain {" chain ".join(rec_info[cdr_id][idx][3])})') # selelct non binding antigens
        pymol.cmd.remove('non_bind_chains') # remove non binding antigens
        pymol.cmd.create('merged', file_name+' | '+file_name.split('_')[0].lower())
        pymol.cmd.delete(file_name[0]+'*')
        pymol.cmd.save(out_dir+file_name+'_merge.pdb','merged')
        pymol.cmd.delete('all')

        print(idxx, file_name, 'fixing merged complex')    
        fixer = PDBFixer(filename=out_dir+file_name+'_merge.pdb')
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(True)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms(seed=0)
        fixer.addMissingHydrogens(7.0)
        PDBFile.writeFile(fixer.topology, fixer.positions, open(out_dir+file_name+'_complex.pdb', 'w'), keepIds=True)            

        print(idxx, file_name, k, 'relaxing complex')        
        pdb = PDBFile(out_dir+file_name+'_complex.pdb')
        modeller = Modeller(pdb.topology, pdb.positions)
        forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        modeller.addHydrogens(forcefield)
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff,
                nonbondedCutoff=1*nanometer, constraints=HBonds)
        integrator = LangevinIntegrator(0, 0.01, 0.0)
        platform = openmm.Platform.getPlatformByName("CUDA")            
        simulation = Simulation(pdb.topology, system, integrator, platform)
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy(maxIterations=0, tolerance=2.39)
        state = simulation.context.getState(getEnergy=True, getPositions=True)        
        with open(out_dir+file_name+'_openmm.pdb', 'w') as f:
            with io.StringIO() as f2:
                PDBFile.writeFile(simulation.topology, state.getPositions(), f2, keepIds=True)
                pdb_min = f2.getvalue()            
            f.write(pdb_min)

            
        # calculate binding energies
        if FoldX_dir!="":
            if os.path.isfile(f'{out_dir}Summary_{file_name}_openmm_AC.fxout'):
                os.system(f'rm {out_dir}Summary_{file_name}_openmm_AC.fxout')
            print(idxx, file_name, k, 'calculating FoldX binding energy')
            os.system(f"{FoldX_dir} --command=AnalyseComplex --pdb={file_name}_openmm.pdb --pdb-dir={out_dir} --analyseComplexChains={interface_chains} --output-dir={out_dir} >/dev/null 2>&1")
            if not os.path.isfile(f'{out_dir}Summary_{file_name}_openmm_AC.fxout'):
                os.system(f"{FoldX_dir} --command=AnalyseComplex --pdb={file_name}_openmm.pdb --pdb-dir={out_dir} --analyseComplexChains={interface_chains} --output-dir={out_dir} >/dev/null 2>&1")            
                time.sleep(3)            
            with open(f'{out_dir}Summary_{file_name}_openmm_AC.fxout') as f:
                a = f.readlines()
            score_foldx = float(a[-1].split()[-3])            
            score_list_foldx.append(score_foldx)
            print(idxx, file_name, 'relax:', k, 'score:', score_foldx)
            if score_foldx < best_foldx and n_relax!=1:
                best_foldx=score_foldx
                with open(out_dir+file_name+'_openmm_best_foldx.pdb', 'w') as f:
                    with io.StringIO() as f2:
                        PDBFile.writeFile(simulation.topology, state.getPositions(), f2, keepIds=True)
                        pdb_min = f2.getvalue()            
                    f.write(pdb_min)                
        
        if IA_dir!="":
            print(idxx, file_name, k, 'calculating InterfaceAnalyer binding energy')    
            if os.path.isfile(f'{out_dir}score_ia.sc'):
                os.system(f'rm {out_dir}score_ia.sc')
            os.system(f"{IA_dir} \
                        -s {out_dir}{file_name}_openmm.pdb -interface {interface_chains.replace(',','_')} \
                        -pack_separated -out:file:score_only {out_dir}score_ia.sc >/dev/null 2>&1")
            if not os.path.isfile(f'{out_dir}{file_name}_score_ia.sc'):
                os.system(f"{IA_dir} \
                            -s {out_dir}{file_name}_openmm.pdb -interface {interface_chains.replace(',','_')} \
                            -pack_separated -out:file:score_only {out_dir}{file_name}_score_ia.sc >/dev/null 2>&1")            
                time.sleep(5)
            with open(f'{out_dir}score_ia.sc') as f:
                a = f.readlines()
            score_ia = float(a[-1].split()[a[-2].split().index('dG_separated')])
            score_list_ia.append(score_ia)
            print(idxx, file_name, 'relax:', k, 'score:', score_ia)
            if score_ia < best_ia and n_relax!=1:
                best_ia = score_ia
                with open(out_dir+file_name+'_openmm_best_ia.pdb', 'w') as f:
                    with io.StringIO() as f2:
                        PDBFile.writeFile(simulation.topology, state.getPositions(), f2, keepIds=True)
                        pdb_min = f2.getvalue()            
                    f.write(pdb_min)                               

    # print(idxx, file_name, 'save scores')        
    rec_BE_foldx.append([file_name]+score_list_foldx)
    rec_BE_ia.append([file_name]+score_list_ia)


# save calculated binding energies 
if FoldX_dir!="":    
    df_fx= pd.DataFrame(rec_BE_foldx)
    df_fx.to_excel(out_dir++file_name.split('_')[0]+'_foldx_energy_scores.xlsx', header=['file_name']+[f'sample {i}' for i in range(n_relax)])
if IA_dir!="":
    df_ia= pd.DataFrame(rec_BE_ia)
    df_ia.to_excel(out_dir+file_name.split('_')[0]+'_ia_energy_scores.xlsx', header=['file_name']+[f'sample {i}' for i in range(n_relax)])
print('job is done!!!')
print("---{}s seconds---".format(time.time()-start_time))

