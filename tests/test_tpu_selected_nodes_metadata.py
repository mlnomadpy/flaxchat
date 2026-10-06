"""Selected TPU campaign orchestration; never imports or executes a model."""
import json
from pathlib import Path

import pytest

from scripts import validate_test_suite as suite


MODULE = 'tests/test_fixture.py'
NODE = 'tests.test_fixture::test_resume[True-False-None]'


@pytest.mark.parametrize('selection,required', [
    (None, {}), ({}, {}), ({MODULE: []}, {}),
    ({MODULE: [NODE, NODE]}, {}),
    ({MODULE: ['tests.test_other::test_resume']}, {}),
    ({MODULE: ['tests.test_fixture::../test_other.py']}, {}),
    ({MODULE: [NODE], 'tests/test_other.py': [NODE]}, {}),
    ({MODULE: [NODE]}, {MODULE: ['tests.test_fixture::test_missing']}),
])
def test_invalid_selection_rejects_before_dispatch(selection, required):
    with pytest.raises(ValueError):
        suite.selected_test_arguments(selection, [Path(MODULE)], required)


def test_exact_selection_requires_exact_passed_inventory(tmp_path):
    xml = tmp_path / 'receipt.xml'
    case = '<testcase classname="tests.test_fixture" name="test_resume[True-False-None]"/>'
    xml.write_text(f'<testsuite>{case}</testsuite>')
    assert suite.validate_inventory(xml, 0, [NODE], [NODE])[1]
    assert not suite.validate_inventory(xml, 0, (), [NODE, 'tests.test_fixture::test_missing'])[1]
    xml.write_text(f'<testsuite>{case}<testcase classname="tests.test_fixture" name="test_extra"/></testsuite>')
    assert not suite.validate_inventory(xml, 0, [NODE], [NODE])[1]
    xml.write_text('<testsuite><testcase classname="tests.test_fixture" name="test_resume[True-False-None]"><skipped/></testcase></testsuite>')
    assert not suite.validate_inventory(xml, 0, [NODE], [NODE])[1]


def test_selection_dispatch_is_explicit_and_recorded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'tests').mkdir()
    (tmp_path / MODULE).write_text('# No execution: dispatch is intercepted.\n')
    selection = tmp_path / 'selected.json'
    selection.write_text(json.dumps({MODULE: [NODE]}))
    monkeypatch.setattr('sys.argv', ['suite', '--output', str(tmp_path / 'results'),
        '--prefix', 'gs://fixture/evidence', '--test-files', MODULE,
        '--selected-nodes', str(selection), '--required-nodes', str(selection)])
    calls = []
    def execute(command, log, timeout, environment):
        calls.append(command)
        assert environment['JAX_PLATFORMS'] == 'tpu'
        assert environment['FLAXCHAT_PHYSICAL_TPU'] == '1'
        log.write_text('metadata-only simulated dispatch')
        if '-c' in command:
            return 0
        assert command[3] == MODULE + '::test_resume[True-False-None]'
        assert MODULE not in command
        xml = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--junitxml=')))
        xml.write_text('<testsuite><testcase classname="tests.test_fixture" name="test_resume[True-False-None]"/></testsuite>')
        return 0
    monkeypatch.setattr(suite, 'bounded', execute)
    monkeypatch.setattr(suite.subprocess, 'run', lambda *args, **kwargs: None)
    assert suite.main() == 0
    report = json.loads((tmp_path / 'results/summary.json').read_text())
    assert len(calls) == 2
    assert report['selected_nodes'] == {MODULE: [NODE]}
    assert report['required_nodes'] == {MODULE: [NODE]}
    assert report['scope'] == 'single_host_selected_tests'
    assert report['counts'] == {'passed': 1, 'failure': 0, 'error': 0, 'skipped': 0}
