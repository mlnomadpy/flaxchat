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
