"""Run the Part 3 notebook in a fresh project Python kernel, without nbconvert."""

import argparse
import os
import queue
import sys
import time
from pathlib import Path

import nbformat
from jupyter_client import KernelManager


def run_notebook(path, timeout=180):
  path = Path(path).resolve()
  notebook = nbformat.read(path, as_version=4)
  original = path.read_bytes()
  manager = KernelManager(kernel_name='python3')
  manager.kernel_spec.argv = [
    sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}',
  ]
  started = time.monotonic()
  # inline backend, as in Jupyter, so plt.show() puts the figures in the notebook
  manager.start_kernel(cwd=str(path.parent), env={
    **os.environ, 'MPLBACKEND': 'module://matplotlib_inline.backend_inline'})
  client = manager.client()
  client.start_channels()
  try:
    client.wait_for_ready(timeout=60)
    for index, cell in enumerate(notebook.cells):
      if cell.cell_type != 'code':
        continue
      print(f'Cell {index + 1}: {cell.source.splitlines()[0]}', flush=True)
      cell.outputs = []
      cell.execution_count = None
      message_id = client.execute(cell.source, allow_stdin=False, stop_on_error=True)
      deadline = time.monotonic() + timeout
      failure = None
      while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
          raise TimeoutError(f'Cell {index + 1} exceeded {timeout}s.')
        try:
          message = client.get_iopub_msg(timeout=min(remaining, 1))
        except queue.Empty:
          if not manager.is_alive():
            raise RuntimeError('The notebook kernel exited unexpectedly.')
          continue
        if message['parent_header'].get('msg_id') != message_id:
          continue
        kind, content = message['msg_type'], message['content']
        if kind == 'status' and content['execution_state'] == 'idle':
          break
        if kind == 'execute_input':
          cell.execution_count = content['execution_count']
        elif kind == 'stream':
          cell.outputs.append(nbformat.v4.new_output(
            'stream', name=content['name'], text=content['text']))
          print(content['text'], end='', flush=True)
        elif kind in ('display_data', 'execute_result'):
          output = {'output_type': kind, 'data': content['data'],
                    'metadata': content.get('metadata', {})}
          if kind == 'execute_result':
            output['execution_count'] = content['execution_count']
          cell.outputs.append(nbformat.from_dict(output))
          if 'text/plain' in content['data']:
            print(content['data']['text/plain'], flush=True)
        elif kind == 'error':
          cell.outputs.append(nbformat.v4.new_output('error', **content))
          failure = '\n'.join(content['traceback'])
        elif kind == 'clear_output':
          cell.outputs = []
      if failure:
        raise RuntimeError(f'Cell {index + 1} failed:\n{failure}')
    nbformat.validate(notebook)
    if path.read_bytes() != original:
      raise RuntimeError('Notebook changed during execution; refusing to replace it.')
    nbformat.write(notebook, path)
    print(f'Completed in {time.monotonic() - started:.1f}s: {path}', flush=True)
  finally:
    manager.shutdown_kernel(now=True)
    client.stop_channels()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--timeout', type=int, default=180, help='Seconds per code cell.')
  args = parser.parse_args()
  run_notebook(Path(__file__).resolve().parents[1] / 'notebook/3_pooled_data.ipynb',
               timeout=args.timeout)
