# Generate molecules
python -m debugpy --wait-for-client --listen 5678 src/generate.py \
  --protein examples/kras/kras.pdb \
  --ref_ligand examples/kras/kras_ref_ligand.sdf \
  --checkpoint checkpoints/drugflow.ckpt \
  --output examples/kras/samples.sdf

python -m debugpy --wait-for-client --listen 5678 src/generate.py \
  --checkpoint runs/pred_z1/checkpoints/epoch=341-step=264708.ckpt \
  --protein examples/7RPZ/7RPZ_protein.pdb \
  --ref_ligand examples/7RPZ/7RPZ_ligand.sdf \
  --output examples/7RPZ/sample.sdf 

python -m debugpy --wait-for-client --listen 5678 src/inpaint.py \
  --checkpoint checkpoints/epoch=68-step=53406.ckpt \
  --protein examples/7RPZ/7RPZ_protein.pdb \
  --ref_ligand examples/7RPZ/7RPZ_ligand.sdf \
  --output examples/7RPZ/sample.sdf \
  --scaffold_ligand examples/7RPZ/7RPZ_scaffold.sdf \
  --n_samples 10 \
  --batch_size 32


# train command
python -m debugpy --wait-for-client --listen 5678 src/train.py --config configs/training/drugflow.yml

# data preprocess command 
python -m debugpy --wait-for-client --listen 5678 src/data/process_crossdocked.py \
  /home/jwang/Workplace/e3pen/dataset/raw/CrossDocked/ \
  --outdir /home/jwang/Workplace/e3pen/dataset/processed/CrossDocked \
  --flex