"""Execute a notebook with parameter overrides, saving the executed copy after every cell.

usage: python scripts/paper_figures/run_nb.py NOTEBOOK TAG [--set name=python_expr ...] [--cfg key=python_expr ...]

--set replaces the first top-level assignment `name = ...` in the notebook.
--cfg injects `cfg[key] = value` right after the dataset yaml is loaded (benchmarking.ipynb).
A cell that raises SystemExit (the notebooks' intentional early stops) ends the run successfully.
The kernel runs in notebooks/, as when the notebook is opened in Jupyter.
"""
import argparse
import datetime
import os
import re
import sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

here = os.path.dirname(os.path.abspath(__file__))
nb_dir = os.path.abspath(os.path.join(here, "..", "..", "notebooks"))

ap = argparse.ArgumentParser()
ap.add_argument("notebook")
ap.add_argument("tag")
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--cfg", action="append", default=[])
args = ap.parse_args()

nb = nbformat.read(os.path.join(nb_dir, args.notebook), as_version=4)
code = [c for c in nb.cells if c.cell_type == "code"]

for item in args.set:
    name, expr = item.split("=", 1)
    pat = re.compile(rf"^{re.escape(name)}\s*=.*$", re.M)
    cell = next((c for c in code if pat.search(c.source)), None)
    if cell is None:
        sys.exit(f"no top-level assignment to {name!r} in {args.notebook}")
    cell.source = pat.sub(lambda m: f"{name} = {expr}  # set by run_nb.py", cell.source, count=1)

if args.cfg:
    anchor = "cfg = cs_utils.load_dataset_yaml(yaml_file)"
    cell = next((c for c in code if anchor in c.source), None)
    if cell is None:
        sys.exit(f"no cfg load in {args.notebook}")
    inject = "".join(f"\ncfg[{k!r}] = {v}  # set by run_nb.py" for k, v in (i.split("=", 1) for i in args.cfg))
    cell.source = cell.source.replace(anchor, anchor + inject, 1)

out_dir = os.path.join(nb_dir, "output", "paper_figures", "executed")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"{args.tag}.ipynb")

client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": nb_dir}})
status = "ok"
with client.setup_kernel():
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        print(f"{datetime.datetime.now():%H:%M:%S} cell {i}", flush=True)
        try:
            client.execute_cell(cell, i)
            # cell magics such as %%time print an exception without failing the cell
            err = next((o for o in cell.outputs if o.get("output_type") == "error"), None)
            if err is not None:
                raise CellExecutionError("", err["ename"], err["evalue"])
        except CellExecutionError as e:
            if e.ename == "SystemExit":
                status = f"stopped at cell {i} (sys.exit)"
                break
            status = f"FAILED at cell {i}: {e.ename}: {e.evalue}"
            break
        finally:
            nbformat.write(nb, out_path)
nbformat.write(nb, out_path)
print(f"{datetime.datetime.now():%H:%M:%S} {args.tag}: {status}", flush=True)
sys.exit(0 if not status.startswith("FAILED") else 1)
