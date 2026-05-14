# Outputs directory

Store intermediate and final outputs of model runs and experiments.

Since output files may be large and numerous, this directory is ignored from git tracking by default.

If you think some outputs may be useful outside your local scope, use DVC to 
track and sync them with the main server. Example:

```
dvc add outputs/[path-to-experiments]
dvc push outputs/[path-to-experiments]
```
