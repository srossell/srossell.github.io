Title: Connecting to a remote jupyter notebook instance
Date: 2020-09-09
Summary: How to connect to a jupyter instance running on are remote server.
Tags: jupyter
Slug: connect_remote_jupyter

## Connecting to a remote jupyter instance.

These instructions work for both "jupyter notebook" and "jupyter-lab". For the
latter, replace "jupyter notebook" by jupyter-lab.

### On the server

Imagine you are using port 6666.

```bash
jupyter notebook --no-browser --port=6666 --NotebookApp.token="mytoken"
```

### On the client side

```bash
ssh -f user@my.server.here -L 6666:localhost:6666 -N
```

