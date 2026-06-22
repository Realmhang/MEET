import os
SCHRODINGER = '/opt/schrodinger2024-1'

def prepare_protein(input_dir,output_dir,pdb_name):
    os.chdir(output_dir)
    input_pdb = os.path.join(input_dir,pdb_name+'.pdb')
    output_maegz = os.path.join(output_dir,pdb_name+'.maegz')
    output_pdb = os.path.join(output_dir,pdb_name+'.pdb')
    os.system(f'{SCHRODINGER}/utilities/prepwizard \
-JOBNAME prepwizard_{pdb_name} \
-disulfides \
-mse \
-captermini \
-cap_termini_min_atoms 5 \
-assign_all_residues \
-rehtreat \
-max_states 1 \
-epik_pH 7.4 \
-epik_pHt 2.0 \
-samplewater \
-propka_pH 7.4 \
-f S-OPLS \
-rmsd 0.30 \
-watdist 5 \
-HOST localhost:4 \
-ATTACHED \
-WAIT \
{input_pdb} \
{output_maegz}')
    os.system(f'{SCHRODINGER}/utilities/structconvert {output_maegz} {output_pdb}')

if __name__ == '__main__':
    from random import shuffle
    from tqdm import tqdm
    #please write a cli to run this script
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    pdb_file_list = os.listdir(input_dir)
    shuffle(pdb_file_list)
    #multi process @16 
    from multiprocessing import Pool
    with Pool(16) as p:
        list(tqdm(p.starmap(prepare_protein, [(input_dir, output_dir, pdb_name.split('.')[0]) for pdb_name in pdb_file_list]), total=len(pdb_file_list)))


