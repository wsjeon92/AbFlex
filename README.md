# AbFlex
AbFlex is a CDR design method with a given antibody-antigen complex.

## Preparation
Under Python libraries are needed to run AbFlex. The dependencies below are verified.
1. PyTorch: 1.9.1+cu102
2. egnn-pytorch: the latest version from https://github.com/lucidrains/egnn-pytorch
3. pandas: 1.3.5
4. numpy: 1.18.5
5. scikit-learn: 1.0
6. scipy: 1.7.1
7. PDBFixer: the latest version from https://github.com/openmm/pdbfixer
8. OpenMM: 7.6 or the latest version from https://github.com/openmm/openmm
9. PyMOL: 2.4.0 Open-Source or https://github.com/schrodinger/pymol-open-source
10. BioPython: 1.77

For binding energy calculation, you need to install Rosetta or FoldX additionally. Otherwise, AbFlex will generate designed PDB files only.

11. FoldX (https://foldxsuite.crg.eu/)
12. Rosetta (https://new.rosettacommons.org/demos/latest/tutorials/install_build/install_build)

## Demo
1. Edit the configuration file "config.json" for your purpose. The configuration file describes information about which cdr to design. **All the directories should be entered as an absolute path.**
```
config.json
{
    "pdb_id": "7bz5",            # str. PDB ID of antibody-antigen complex
    "ab_chain_list": ["H", "L"], # list. Antibody chains in the input complex.
                                   It should contain all the antibody chains in the input PDB file.
    "ag_chain_list": ["A"],      # list. Antigen chains in the input complex.
                                   It should contain all the antigen chains in the input PDB file.
    "Design_chain": "H",         # str. One antibody chain to be designed
    "cdr_type": "H3",            # str. CDR type to be designed
    "cdr_seq": "EAYGMDV",        # str. The original sequence of the CDR is to be designed in the input PDB file.
    "out_dir": "/absolute-path/AbFlex/your-working-dir/",
                                 # str. The working directory. Output files will be generated here.
    "FoldX_dir": "",             # str. If you wish to utilize FoldX, please input the location of
                                   the executable FoldX file. If not, you may leave it blank.
    "IA_dir": "",                # str. If you wish to utilize Rosetta InterfaceAnalyzer, please input
                                   the location of the executable InterfaceAnalyzer file.
                                   e.g. "/where-the-rosetta-install/bin/InterfaceAnalyzer.linuxgccrelease"
                                   If not, you may leave it blank.
    "n_sample": 1,               # int. The number of CDR structure samples to be generated from the predicted results.
                                   The samples will have different CDR sequences from each other.
    "n_relax": 1,                # int. The number of relaxations by openMM. If you use FoldX or Rosetta InterfaceAnalyer,
                                   then use 1~10. otherwise, set it to 1.
}
```

2. locate the "config.json" file in the same location as the "run.py" file.
3. run "run.py" file

## Output file description
Whether or not to use FoldX and Rosetta InterfaceAnalyzer, the file below will be generated as default.
1. ```{pdb_id}_{chain_id}_{CDR_type}_{sample#}_openmm.pdb```: The relaxed full-atom model built with the predicted CDR C alpha coordinates and samples sequences.

If you using Rosetta InterfaceAnalzer or FoldX to evaluate the binding energy and n_relax>1, the below files will be generated additionally.
2.  ```{pdb_id}_{chain_id}_{CDR_type}_{sample#}_best_{Foldx or IA}.pdb```: The best binding energy structure through the relaxation
3.  ```{pdb_id}_*_energy_scores.xlsx```: Calculated binding energies of the generated samples.
