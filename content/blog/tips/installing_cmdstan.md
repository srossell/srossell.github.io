Title: Installing cmdstan
Date: 2020-09-08
Summary: Instructions for installing command Stan within a conda environment.
Tags: stan
Slug: installing-cmdstan

## Installing command Stan within a conda environmnet.

First create (or activate) a conda enviroment.
```bash
conda activate myenv
conda install gcc_linux-64
```

Clone the git repository (in a path of your choice)
```bash
git clone https://github.com/stan-dev/cmdstan.git

cd cmdstan

git submodule update --init --recursive
```

Typing "make" shows a set of instructions
```bash
make
```

I used (on a big server)
```bash
make build -j10
```

Then you can check if you can compile and run the bernulli model, following the
instructions you get when you type "make".

