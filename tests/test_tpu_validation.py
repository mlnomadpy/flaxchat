from scripts.validate_tpu import junit_inventory, source_digest


def test_junit_preserves_failures_errors_and_skip_reasons(tmp_path):
    xml = tmp_path / 'tests.xml'
    xml.write_text('''<testsuites><testsuite>
    <testcase classname="model" name="ok" time="1"/>
    <testcase classname="model" name="bad"><failure message="wrong loss"/></testcase>
    <testcase classname="model" name="error"><error message="setup failed"/></testcase>
    <testcase classname="model" name="skip"><skipped message="needs multiple hosts"/></testcase>
    </testsuite></testsuites>''')
    results = junit_inventory(xml)
    assert [item['status'] for item in results] == ['passed', 'failure', 'error', 'skipped']
    assert results[-1]['message'] == 'needs multiple hosts'


def test_source_hash_detects_uncommitted_changes(tmp_path):
    source = tmp_path / 'scripts'
    source.mkdir()
    file = source / 'train.py'
    file.write_text('before')
    original = source_digest(tmp_path)
    file.write_text('after')
    assert source_digest(tmp_path) != original
    original = source_digest(tmp_path)
    tasks = tmp_path / 'tasks'
    tasks.mkdir()
    (tasks / 'eval.py').write_text('changed evaluation protocol')
    assert source_digest(tmp_path) != original


def test_suite_rejects_empty_or_all_skipped_inventory(tmp_path):
    from scripts.validate_test_suite import validate_inventory
    xml = tmp_path/'tests.xml'
    for cases in ('', '<testcase name="skip"><skipped/></testcase>'):
        xml.write_text(f'<testsuite>{cases}</testsuite>')
        assert validate_inventory(xml, 0)[1] is False
    xml.write_text('<testsuite><testcase name="pass"/></testsuite>')
    assert validate_inventory(xml, 0)[1] is True
    assert validate_inventory(xml, 124)[1] is False
    xml.write_text('<testsuite><testcase name="pass"/><testcase name="bad"><failure/></testcase></testsuite>')
    assert validate_inventory(xml, 0)[1] is False



def test_suite_records_backend_appropriate_precision_without_mutation():
    from scripts.validate_test_suite import module_environment
    env = {'JAX_DEFAULT_MATMUL_PRECISION': 'highest', 'JAX_PLATFORMS': 'tpu'}
    for file in ('tests/test_attention_accelerator.py',):
        assert module_environment(env, file)['JAX_DEFAULT_MATMUL_PRECISION'] == 'default'
    assert module_environment(env, 'tests/test_encoder.py')['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    assert module_environment(env, 'tests/test_encoder_training.py')['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    assert env['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'


def test_suite_main_preserves_failure_and_skip_inventory(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts import validate_test_suite as suite
    monkeypatch.chdir(tmp_path)
    (tmp_path/'tests').mkdir()
    for name in ('pass', 'fail', 'skip', 'timeout', 'missing'):
        (tmp_path/'tests'/f'test_{name}.py').write_text('')
    monkeypatch.setattr('sys.argv', ['suite', '--output', str(tmp_path/'results'),
        '--prefix', 'gs://test/evidence', '--include-distributed-cpu'])
    def execute(command, log, timeout, environment):
        assert timeout <= 240
        assert environment['FLAXCHAT_RUN_DISTRIBUTED_CPU'] == '1'
        log.write_text('captured log')
        if '-c' in command:
            return 0
        xml = next(arg.split('=',1)[1] for arg in command if arg.startswith('--junitxml='))
        from pathlib import Path
        name = Path(command[3]).stem
        if name != 'test_missing':
            status = '<failure message="bad"/>' if name == 'test_fail' else '<skipped message="opt-in"/>' if name == 'test_skip' else ''
            Path(xml).write_text(f'<testsuite><testcase name="case">{status}</testcase></testsuite>')
        return 124 if name == 'test_timeout' else 0
    uploads = []
    def upload(command, **kwargs):
        assert kwargs['timeout'] == 60
        uploads.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(suite, 'bounded', execute)
    monkeypatch.setattr(suite.subprocess, 'run', upload)
    assert suite.main() == 1
    report = json.loads((tmp_path/'results/summary.json').read_text())
    assert report['passed'] is False
    assert len(report['results']) == 5
    assert sum(r['passed'] for r in report['results']) == 1
    assert report['counts']['failure'] == 1
    assert report['counts']['skipped'] == 1
    assert any(r['evidence_error'] for r in report['results'])
    assert len(uploads) == 7
