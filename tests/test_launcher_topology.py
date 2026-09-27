import pytest
from scripts.launcher_topology import launcher_topology


def test_single_host_without_distributed_environment():
    assert launcher_topology({}, expected_processes=1) == (0, 1)


def test_multi_host_keeps_explicit_rank_and_count():
    assert launcher_topology({'JAX_PROCESS_INDEX':'3','JAX_PROCESS_COUNT':'4'}, expected_processes=4) == (3, 4)


@pytest.mark.parametrize('env', [ {'JAX_PROCESS_INDEX':'0'}, {'JAX_PROCESS_COUNT':'4'},
    {'JAX_PROCESS_INDEX':'4','JAX_PROCESS_COUNT':'4'},
    {'JAX_PROCESS_INDEX':'-1','JAX_PROCESS_COUNT':'1'},
    {'JAX_PROCESS_INDEX':'0','JAX_PROCESS_COUNT':'0'},
    {'JAX_PROCESS_INDEX':'bad','JAX_PROCESS_COUNT':'4'}])
def test_incomplete_or_invalid_topology_is_not_silently_single_host(env):
    with pytest.raises(ValueError):
        launcher_topology(env)


def test_multi_host_requirement_cannot_be_satisfied_by_single_host():
    with pytest.raises(ValueError, match='Expected 2 processes'):
        launcher_topology({}, expected_processes=2)
