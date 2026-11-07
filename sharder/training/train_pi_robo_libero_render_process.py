"""Run unchanged LIBERO in a fresh process, optionally on another allocation.

The learner, data loader and training loop are the original implementations.
Only the environment transport changes: synchronous local IPC isolates EGL
from the learner's CUDA context. Arrays are pickled losslessly, without resize,
compression, action clipping, or changes to simulation timing.
"""

import atexit
import json
from multiprocessing.connection import Client, Listener
import os
from pathlib import Path
import pickle
import random
import resource
import socket
import subprocess
import sys
import time
import traceback
import weakref

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
_live = weakref.WeakSet()


def _worker(fd):
    import numpy as np
    from dcrl.libero_env import LiberoTaskEnv

    connection = socket.socket(fileno=fd)
    stream = connection.makefile('rwb')
    env = None
    while True:
        try:
            command, arguments, np_state, py_state = pickle.load(stream)
        except EOFError:
            break
        if command in ('init', 'reset', 'seed'):
            np.random.set_state(np_state)
            random.setstate(py_state)
        try:
            if command == 'init':
                args, kwargs = arguments
                env = LiberoTaskEnv(*args, **kwargs)
                result = (env.instruction, len(env.init_states))
                print('[RENDER PROCESS] pid=%s CUDA_VISIBLE_DEVICES=%s resolution=256'
                      % (os.getpid(), os.getenv('CUDA_VISIBLE_DEVICES')), flush=True)
                gpu = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
                print('[RENDER GPU] job=%s gpu=%s host=%s'
                      % (os.getenv('SLURM_JOB_ID'), gpu, socket.gethostname()), flush=True)
            elif command == 'reset':
                env.episode_index = arguments[0]
                result = (env.reset(), env.episode_index)
                if os.getenv('RENDER_DEBUG_FRAMES') == '1':
                    print('[RENDER FRAME] ' + json.dumps({
                        key: {'shape': list(image.shape), 'dtype': str(image.dtype),
                              'min': int(image.min()), 'max': int(image.max()),
                              'mean': float(image.mean())}
                        for key, image in result[0]['image'].items()}), flush=True)
            elif command == 'step':
                if os.getenv('RENDER_TRACE_ACTIONS') == '1':
                    print('[RENDER ACTION] ' + json.dumps({'pid': os.getpid(),
                          'seed': env.seed, 'episode_index': env.episode_index,
                          'action': np.asarray(arguments[0]).tolist()}), flush=True)
                result = env.step(*arguments)
            elif command == 'seed':
                result = env.env.seed(*arguments)
            elif command == 'demo_path':
                result = env.demo_path(*arguments)
            elif command == 'close':
                result = env.close()
            else:
                raise ValueError(command)
            response = (True, result, np.random.get_state(), random.getstate())
        except Exception:
            response = (False, traceback.format_exc(), np.random.get_state(), random.getstate())
        pickle.dump(response, stream, protocol=5)
        stream.flush()
        if command == 'close':
            break
    stream.close()
    connection.close()


class _SeedAdapter:
    def __init__(self, owner):
        self.owner = weakref.ref(owner)

    def seed(self, value):
        return self.owner()._call('seed', value)


class ProcessLiberoTaskEnv:
    def __init__(self, *args, **kwargs):
        self.seed = kwargs['seed']
        self.episode_index = 0
        self.env = _SeedAdapter(self)
        self._closed = False
        child_environment = os.environ.copy()
        # No learner or accelerator computations run in the renderer process.
        # Its OpenGL context still uses the SAME Slurm-assigned NVIDIA GPU.
        child_environment['JAX_PLATFORMS'] = 'cpu'
        render_job = os.getenv('RENDER_JOB_ID')
        if render_job:
            # Authenticate local IPC before accepting any pickled objects.
            authkey = os.urandom(32)
            child_environment['RENDER_RPC_AUTH'] = authkey.hex()
            # Slurm gives each allocation a private /tmp mount. Do not pass
            # the learner's allocation-local temporary path to the renderer.
            child_environment['TMPDIR'] = '/tmp'
            rpc_host = os.getenv('RENDER_RPC_HOST', '127.0.0.1')
            child_environment['RENDER_CONNECT_HOST'] = rpc_host
            bind_host = '127.0.0.1' if rpc_host == '127.0.0.1' else '0.0.0.0'
            listener = Listener((bind_host, 0), family='AF_INET', authkey=authkey)
            listener._listener._socket.settimeout(5)
            self._process = subprocess.Popen(
                ['srun', '--jobid='+render_job, '--overlap', '--nodes=1', '--ntasks=1',
                 '--cpus-per-task=1', '--cpu-bind=none', '--exact', '--export=ALL',
                 '--chdir=/projects/bgqe/rpolonsky/dcrl',
                 sys.executable, str(Path(__file__).resolve()),
                 '--render-worker-connect', str(listener.address[1])],
                env=child_environment,
            )
            deadline = time.monotonic() + 300
            try:
                while True:
                    try:
                        authenticated = listener.accept()
                        break
                    except socket.timeout:
                        if self._process.poll() is not None:
                            raise RuntimeError('Render srun exited before connecting')
                        if time.monotonic() >= deadline:
                            self._process.terminate()
                            raise TimeoutError('Render allocation did not connect to the learner')
                parent = socket.socket(fileno=os.dup(authenticated.fileno()))
                authenticated.close()
            finally:
                listener.close()
        else:
            parent, child = socket.socketpair()
            self._process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), '--render-worker', str(child.fileno())],
                pass_fds=(child.fileno(),), env=child_environment,
            )
            child.close()
        self._connection = parent
        self._stream = parent.makefile('rwb')
        _live.add(self)
        try:
            self.instruction, count = self._call('init', args, kwargs)
            self.init_states = range(count)
        except BaseException:
            self.close()
            raise

    def _call(self, command, *arguments):
        import numpy as np

        try:
            pickle.dump((command, arguments, np.random.get_state(), random.getstate()),
                        self._stream, protocol=5)
            self._stream.flush()
            okay, result, np_state, py_state = pickle.load(self._stream)
        except (EOFError, BrokenPipeError, ConnectionResetError) as error:
            raise RuntimeError('LIBERO render process exited: %s' % self._process.poll()) from error
        # Steps, path queries and close are deterministic with respect to
        # process-global RNGs. Do not rewind replay-prefetch RNG draws that
        # may occur in the learner while a remote step is in progress.
        if command in ('init', 'reset', 'seed'):
            np.random.set_state(np_state)
            random.setstate(py_state)
        if not okay:
            raise RuntimeError(result)
        return result

    def reset(self):
        result, self.episode_index = self._call('reset', self.episode_index)
        return result

    def step(self, action):
        return self._call('step', action)

    def demo_path(self, demos_root=None):
        return self._call('demo_path', demos_root)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.poll() is None:
                self._call('close')
        except (OSError, RuntimeError):
            pass
        finally:
            self._stream.close()
            self._connection.close()
            try:
                self._process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                self._process.wait(timeout=15)


def _close_workers():
    for env in list(_live):
        env.close()


atexit.register(_close_workers)

if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--render-worker':
        _worker(int(sys.argv[2]))
        # The explicit close RPC already closed MuJoCo and acknowledged the
        # caller. Avoid running EGL extension destructors again at interpreter
        # shutdown: on this driver those can hang after successful env.close().
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    elif len(sys.argv) == 3 and sys.argv[1] == '--render-worker-connect':
        authenticated = Client((os.environ.pop('RENDER_CONNECT_HOST', '127.0.0.1'), int(sys.argv[2])),
                               authkey=bytes.fromhex(os.environ.pop('RENDER_RPC_AUTH')))
        fd = os.dup(authenticated.fileno())
        authenticated.close()
        _worker(fd)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    else:
        from dcrl import libero_env
        libero_env.LiberoTaskEnv = ProcessLiberoTaskEnv
        import train_pi_robo_libero_logged as logged
        print('[RENDER PROCESS] original trainer + original environment; render allocation=%s'
              % os.getenv('RENDER_JOB_ID', 'same GPU'), flush=True)
        logged.trainer.app.run(logged.trainer.main)
