"""Execute ``notebooks/02_visualizations_demo.ipynb`` end-to-end and
print a one-line summary per cell.

Used as a smoke test for the demo notebook. Runs with whatever
``RUN_LIVE`` setting the notebook has committed (default ``False`` ->
cassette mode). With ``--save`` the executed notebook (with all
outputs, including rendered Mermaid display_data) is written back to
disk so it can be committed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOK = Path("notebooks/02_visualizations_demo.ipynb")


def main() -> int:
    save_outputs = "--save" in sys.argv
    repo_root = Path(__file__).resolve().parents[1]
    nb_path = repo_root / NOTEBOOK
    if not nb_path.exists():
        print(f"Notebook not found: {nb_path}")
        return 1

    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    try:
        client.execute()
    except CellExecutionError as e:
        print(f"Notebook execution failed: {e}")
        return 2

    if save_outputs:
        nbformat.write(nb, str(nb_path))
        print(f"Saved executed notebook to {nb_path}")

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    print(f"Executed {len(code_cells)} code cells successfully.")
    for i, cell in enumerate(code_cells):
        outputs = cell.get("outputs", [])
        text_lines: list[str] = []
        for out in outputs:
            if out.get("output_type") == "stream":
                text_lines.extend(out.get("text", "").splitlines())
            elif out.get("output_type") in ("execute_result", "display_data"):
                data = out.get("data", {})
                if "text/plain" in data:
                    text_lines.extend(str(data["text/plain"]).splitlines())
        preview = next((line for line in text_lines if line.strip()), "(no output)")
        print(f"  cell {i + 1}: {preview[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
