# (Bayesian) Normalizing Flow Network

![Outline of the Normalizing Flow Network](nfn_outline.png)

## This repo implements:
- Conditional Density Estimators: NFN, MDN, KMN
- Normalizing Flows: Radial, Planar, Affine


### Installing dependencies
```bash
pip install -r requirements.txt
# Only necessary for running the evaluation scripts
pip install --no-dependencies cde
```
The TensorFlow version of the cde package conflicts with our version. This is not a problem since we import only `cde.density_simulation` which doesn't depend on TensorFlow.
### Runing tests
Tests are implemented using pytests
```bash
make tests
```
