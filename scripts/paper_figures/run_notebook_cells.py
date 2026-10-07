"""Execute selected cells of a notebook in one kernel and write their outputs back into the notebook itself.

For notebooks where only one section produces a figure (e.g. 8cube.ipynb's Trem2 section, which its own notes say
to run after the setup cells, skipping the eight-plate analysis). The other cells keep the outputs they already have.

usage: python scripts/paper_figures/run_notebook_cells.py NOTEBOOK --cells 1,3,4,38-56
NOTEBOOK is a file in notebooks/; cell numbers are 0-based positions in the notebook (markdown cells in a range are skipped).
"""
import argparse
import os

import nbformat
from nbclient import NotebookClient

here = os.path.dirname(os.path.abspath(__file__))
nb_dir = os.path.abspath(os.path.join(here, "..", "..", "notebooks"))


def parse_cells(spec):
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    return out


ap = argparse.ArgumentParser()
ap.add_argument("notebook")
ap.add_argument("--cells", required=True)
args = ap.parse_args()

path = os.path.join(nb_dir, args.notebook)
nb = nbformat.read(path, as_version=4)
cells = [i for i in parse_cells(args.cells) if nb.cells[i].cell_type == "code"]

client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": nb_dir}})
with client.setup_kernel():
    for i in cells:
        print(f"cell {i}", flush=True)
        client.execute_cell(nb.cells[i], i)
        err = [o for o in nb.cells[i].outputs if o.get("output_type") == "error"]
        if err:
            raise SystemExit(f"cell {i} failed: {err[0]['ename']}: {err[0]['evalue']}")

tracked = nbformat.read(path, as_version=4)   # re-read, so edits made while this ran are kept
for i in cells:
    tracked.cells[i].outputs = nb.cells[i].outputs
    tracked.cells[i].execution_count = nb.cells[i].get("execution_count")
nbformat.write(tracked, path)
print(f"done: outputs of cells {cells} written to {path}", flush=True)
