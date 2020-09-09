Title: Installing a jupyter kernel
Date: 2020-09-09
Summary: How to install a jupyter kernel.
Tags: jupyter
Slug: install_jupyter_kernel

## Installing a jupyter kernel.

Imagine "main" is the name of your environment.

```bash
python -m ipykernel install --user --name main --display-name "main"
```

Note that the display name can be different than the enviroment name (in the
example they are both "main").
