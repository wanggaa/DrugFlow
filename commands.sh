# Generate molecules
python -m debugpy --wait-for-client --listen 5678 src/generate.py \
  --protein examples/kras/kras.pdb \
  --ref_ligand examples/kras/kras_ref_ligand.sdf \
  --checkpoint checkpoints/drugflow.ckpt \
  --output examples/kras/samples.sdf

python -m debugpy --wait-for-client --listen 5678 src/generate.py \
  --checkpoint checkpoints/drugflow.ckpt \
  --protein examples/7RPZ/7RPZ_protein.pdb \
  --ref_ligand examples/7RPZ/7RPZ_ligand.sdf \
  --output examples/7RPZ/sample.sdf 

python -m debugpy --wait-for-client --listen 5678 src/inpaint.py \
  --checkpoint checkpoints/drugflow.ckpt \
  --protein examples/7RPZ/7RPZ_protein.pdb \
  --ref_ligand examples/7RPZ/7RPZ_ligand.sdf \
  --output examples/7RPZ/sample.sdf \
  --scaffold_ligand examples/7RPZ/7RPZ_scaffold.sdf \
  --n_samples 10 \
  --batch_size 32