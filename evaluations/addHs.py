import os
from schrodinger import structure
from schrodinger.structutils.build import add_hydrogens

input_folder = '/data_hdd/home/chenzitao/Cls_vae/PostProcessForRebuttle/ex1-fix'
output_folder = '/data_hdd/home/chenzitao/Cls_vae/PostProcessForRebuttle/ex1-fix-addHs'

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.sdf'):
        input_sdf_path = os.path.join(input_folder, filename)

        try:
            mols = list(structure.StructureReader(input_sdf_path))
        except Exception as e:
            print(f"Error loading file {filename}: {e}")
            continue

        output_sdf_path = os.path.join(output_folder, f'hydrogen_{filename}')
        try:
            with structure.StructureWriter(output_sdf_path) as writer:
                for i, mol in enumerate(mols):
                    add_hydrogens(mol)
                    writer.append(mol)
                    print(f"Processed molecule {i+1} in file {filename}")
            print(f"Processed and saved file: {output_sdf_path}")
        except Exception as e:
            print(f"Error saving file {filename}: {e}")