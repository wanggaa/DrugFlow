# Generate molecules
python -m debugpy --wait-for-client --listen 5678 src/generate.py \
  --protein examples/kras/kras.pdb \
  --ref_ligand examples/kras/kras_ref_ligand.sdf \
  --checkpoint checkpoints/drugflow.ckpt \
  --output examples/kras/samples.sdf

python src/generate.py \
  --checkpoint  checkpoints/epoch=729-step=2300230.ckpt \
  --protein examples/7RPZ/7RPZ_protein.pdb \
  --ref_ligand examples/7RPZ/7RPZ_ligand.sdf \
  --output examples/7RPZ/free_sample.sdf \
  --n_samples 96 \
  --batch_size 32

python -m debugpy --wait-for-client --listen 5678 src/inpaint.py \
  --checkpoint checkpoints/epoch=729-step=2300230.ckpt \
  --protein examples/7RPZ/7RPZ_protein.pdb \
  --ref_ligand examples/7RPZ/7RPZ_ligand.sdf \
  --output examples/7RPZ/inpaint_sample.sdf \
  --scaffold_ligand examples/7RPZ/7RPZ_scaffold.sdf \
  --n_samples 32 \
  --batch_size 32


# train command
python -m debugpy --wait-for-client --listen 5678 src/train.py --config configs/training/drugflow.yml

# data preprocess command 
python -m debugpy --wait-for-client --listen 5678 src/data/process_crossdocked.py \
  /home/jwang/Workplace/e3pen/dataset/raw/CrossDocked/ \
  --outdir /home/jwang/Workplace/e3pen/dataset/processed/CrossDocked \
  --flex