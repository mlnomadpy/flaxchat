"""Read the launcher's explicit multi-host environment or single-host defaults."""
import os


def launcher_topology(environ=None, *, expected_processes=None):
    env = os.environ if environ is None else environ
    present = [name in env for name in ('JAX_PROCESS_INDEX', 'JAX_PROCESS_COUNT')]
    if any(present) and not all(present):
        raise ValueError('Incomplete launcher topology: rank and process count must be set together')
    try:
        rank, count = int(env.get('JAX_PROCESS_INDEX', '0')), int(env.get('JAX_PROCESS_COUNT', '1'))
    except (TypeError, ValueError) as error:
        raise ValueError('Launcher rank and process count must be integers') from error
    if count < 1 or not 0 <= rank < count:
        raise ValueError('Launcher rank outside process count')
    if expected_processes is not None and expected_processes != count:
        raise ValueError(f'Expected {expected_processes} processes but launcher provides {count}')
    return rank, count
