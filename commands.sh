# Generate molecules
python src/generate.py \
  --protein examples/kras/kras.pdb \
  --ref_ligand examples/kras/kras_ref_ligand.sdf \
  --checkpoint checkpoints/drugflow.ckpt \
  --output examples/kras/samples.sdf


python src/generate.py \
  --checkpoint checkpoints/drugflow.ckpt \
  --protein examples/7RPZ/7RPZ_protein.pdb \
  --ref_ligand examples/7RPZ/7RPZ_ligand.sdf \
  --output examples/7RPZ/sample.sdf \
  --gnina $CONDA_PREFIX/bin/gnina

