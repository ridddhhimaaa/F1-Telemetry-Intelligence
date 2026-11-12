import io
import sys
import time
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import nbformat

nb_path = Path("notebooks/features/01_advanced_feature_engineering.ipynb")
nb = nbformat.read(nb_path, as_version=4)

global_env = {}

print("Executing notebook cells against PostgreSQL...")
execution_count = 1

for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        print(f"Running cell {i} (execution_count={execution_count})...")
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        start = time.perf_counter()
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(cell.source, global_env)
            elapsed = time.perf_counter() - start
            out_text = stdout_buf.getvalue()
            err_text = stderr_buf.getvalue()
            
            outputs = []
            if out_text:
                outputs.append(nbformat.v4.new_output(output_type="stream", name="stdout", text=out_text))
            if err_text:
                outputs.append(nbformat.v4.new_output(output_type="stream", name="stderr", text=err_text))
                
            cell.outputs = outputs
            cell.execution_count = execution_count
            execution_count += 1
            print(f"Cell {i} succeeded in {elapsed:.3f}s.")
        except Exception as e:
            print(f"ERROR in cell {i}: {e}")
            raise e

nbformat.write(nb, nb_path)
print("Notebook execution completed successfully and saved with fresh outputs.")
