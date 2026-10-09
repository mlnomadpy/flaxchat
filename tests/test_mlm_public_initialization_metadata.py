"""Host-only admission checks; no model construction or numerical execution."""
from dataclasses import asdict
import json

import pytest

from flaxchat.encoder import EncoderConfig
from flaxchat.encoder_data import file_hash
from scripts.release_contract import canonical_hash, checkpoint_identity
from scripts.train_encoder import public_initialization_metadata


def export(tmp_path, family='modernbert', config=None):
    config = config or EncoderConfig()
    (tmp_path / 'tokenizer.json').write_text('{}')
    tokenizer = file_hash(tmp_path / 'tokenizer.json')
    metadata = dict(model_family=family, resolved_config={'encoder': asdict(config)},
                    tokenizer_identity=tokenizer)
    manifest = dict(step=48236, metadata_sha256=canonical_hash(metadata),
                    identity=checkpoint_identity(metadata), model_state={})
    manifest['identity_sha256'] = canonical_hash(manifest['identity'])
    for name, value in [('config', asdict(config)), ('checkpoint-metadata', metadata),
                        ('checkpoint-manifest', manifest)]:
        (tmp_path / (name + '.json')).write_text(json.dumps(value))
    (tmp_path / 'model.safetensors').write_bytes(b'host-only placeholder; physical restore checks contents')
    return config, tokenizer, canonical_hash(metadata), canonical_hash(manifest)


def test_mlm_public_admission(tmp_path):
    config, tokenizer, metadata_hash, manifest_hash = export(tmp_path)
    committed, receipt = public_initialization_metadata(tmp_path, config, tokenizer, metadata_hash, manifest_hash)
    assert committed['step'] == receipt['step'] == 48236
    assert 'fresh_optimizer' in receipt['policy']
    assert receipt['weights_sha256'] == file_hash(tmp_path / 'model.safetensors')
    with pytest.raises(ValueError, match='identity differs'):
        public_initialization_metadata(tmp_path, config, tokenizer, '0' * 64, manifest_hash)
    with pytest.raises(ValueError, match='tokenizer differs'):
        public_initialization_metadata(tmp_path, config, 'other', metadata_hash, manifest_hash)
    (tmp_path / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='configuration differs'):
        public_initialization_metadata(tmp_path, config, tokenizer, metadata_hash, manifest_hash)


@pytest.mark.parametrize('family', ['yat_embedding_finetune', 'modernbert_contrastive_encoder'])
def test_reject_contrastive_as_mlm_parent(tmp_path, family):
    arguments = export(tmp_path, family)
    with pytest.raises(ValueError, match='identity differs'):
        public_initialization_metadata(tmp_path, *arguments)


def test_public_resume_identity_is_content_bound(tmp_path):
    from scripts.train_encoder import parser, public_initialization_identity
    values = export(tmp_path)
    _, receipt = public_initialization_metadata(tmp_path, *values)
    first = public_initialization_identity(receipt)
    assert public_initialization_identity(receipt | {'public_export': '/other/host/parent'}) == first
    assert public_initialization_identity(receipt | {'weights_sha256': '0' * 64}) != first
    assert public_initialization_identity(receipt | {'step': 1}) != first
    args = parser().parse_args(['--config', 'config.json', '--data', 'data', '--output', 'out',
                                '--initialize-from-public', str(tmp_path),
                                '--initialize-public-metadata-sha256', values[2],
                                '--initialize-public-manifest-sha256', values[3], '--resume'])
    assert args.resume and args.initialize_from_public == str(tmp_path)


def test_worker_preserves_native_yat_encoder_flags(tmp_path):
    from types import SimpleNamespace
    from dataclasses import replace
    from scripts import run_yat_mlm_continuation as worker
    from scripts import train_encoder
    parent = tmp_path / 'parent'
    parent.mkdir()
    config = EncoderConfig(compute_dtype='bfloat16', ffn_type='yat_glu', attention_score='yat_softmax',
        yat_compute_mode='bf16', yat_ffn_compute_mode='bf16_adaptive',
        yat_attention_implementation='centered_fp32_scores',
        yat_attention_block_size=64, yat_global_attention_block_size=64,
        residual_dtype='float32', mlm_projection='masked', mlm_loss_backend='xla_full')
    metadata = {'model_family': 'modernbert', 'resolved_config': {'encoder': asdict(config)}}
    manifest = {'step': 48236}
    for name, value in [('checkpoint-metadata', metadata), ('checkpoint-manifest', manifest)]:
        (parent / (name + '.json')).write_text(json.dumps(value))
    args = SimpleNamespace(root=tmp_path, parent_metadata_sha256=canonical_hash(metadata),
        parent_manifest_sha256=canonical_hash(manifest), steps=100000, batch_size=64,
        accumulation_steps=4, output='gs://bucket/checkpoints', save_every=250)
    cli = train_encoder.parser().parse_args(worker.training_arguments(args))
    assert cli.ffn_type == 'yat_glu' and cli.attention_score == 'yat_softmax'
    assert cli.yat_compute_mode == 'bf16' and cli.yat_ffn_compute_mode == 'bf16_adaptive'
    assert cli.mlm_projection == 'masked' and cli.mlm_loss_backend == 'xla_full'
    assert cli.yat_attention_block_size == cli.yat_global_attention_block_size == 64
    assert replace(config, compute_dtype=cli.dtype, residual_dtype=cli.residual_dtype,
                   use_remat=not cli.no_remat) == config
    assert worker.canonical_hash(metadata) == canonical_hash(metadata)


def test_qualified_continuation_binds_runtime_stage_and_evidence(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import run_yat_mlm_continuation as worker
    args = SimpleNamespace(root=tmp_path)
    (tmp_path / 'source').mkdir()
    (tmp_path / 'source/code.py').write_text('frozen = True\n')
    logs = tmp_path / 'mlm-run-evidence'
    logs.mkdir()
    monkeypatch.setenv('FLAXCHAT_RUN_MANIFEST_SHA256', 'a' * 64)
    monkeypatch.setenv('FLAXCHAT_RUNTIME_LOCK_SHA256', 'b' * 64)
    monkeypatch.setattr(worker, 'training_arguments', lambda _args: ['same', 'mlm', 'stage'])
    baseline = {'masked_token_loss': 2.0}
    evaluated = {'masked_token_loss': 1.9}
    qualification = worker.qualification_identity(args) | {
        'parent_evaluation_sha256': worker.canonical_hash(baseline),
        'step8_evaluation_sha256': worker.canonical_hash(evaluated),
        'tests': [{'id': name, 'status': 'passed'} for name in (
            'mlm-parent-import-update', 'mlm-checkpoint-resume', 'mlm-heldout-parent-gate')]}
    (tmp_path / 'mlm-qualification.json').write_text(json.dumps(qualification))
    (logs / 'evaluate-parent-4.json').write_text(json.dumps(baseline))
    (logs / 'evaluate-8.json').write_text(json.dumps(evaluated))
    (logs / 'status.json').write_text(json.dumps({'status': 'qualified', 'completed_step': 8,
        'physical_import_update_resume_qualified': True}))
    assert worker.verify_continuation(args, logs)[1] == baseline
    monkeypatch.setenv('FLAXCHAT_RUNTIME_LOCK_SHA256', 'c' * 64)
    with pytest.raises(ValueError, match='no longer matches'):
        worker.verify_continuation(args, logs)
    monkeypatch.setenv('FLAXCHAT_RUNTIME_LOCK_SHA256', 'b' * 64)
    (logs / 'evaluate-8.json').write_text('{}')
    with pytest.raises(ValueError, match='evidence changed'):
        worker.verify_continuation(args, logs)


def test_goat_public_migration_requires_explicit_exact_architecture_change(tmp_path):
    from dataclasses import replace
    from scripts.train_encoder import public_initialization_identity
    source = EncoderConfig(attention_score='yat_softmax', ffn_type='yat_glu')
    _, tokenizer, metadata_hash, manifest_hash = export(tmp_path, config=source)
    target = replace(source, attention_score='goat')
    with pytest.raises(ValueError, match='configuration differs'):
        public_initialization_metadata(tmp_path, target, tokenizer, metadata_hash, manifest_hash)
    _, receipt = public_initialization_metadata(tmp_path, target, tokenizer,
        metadata_hash, manifest_hash, migrate_to_goat=True)
    migration = receipt['architecture_migration']
    assert migration['source_encoder'] == asdict(source)
    assert migration['target_encoder'] == asdict(target)
    assert migration['optimizer'] == 'fresh'
    assert public_initialization_identity(receipt)['architecture_migration'] == migration
    with pytest.raises(ValueError, match='configuration differs'):
        public_initialization_metadata(tmp_path, replace(target, intermediate_size=128),
            tokenizer, metadata_hash, manifest_hash, migrate_to_goat=True)
    with pytest.raises(ValueError, match='requires yat_softmax'):
        public_initialization_metadata(tmp_path, source, tokenizer,
            metadata_hash, manifest_hash, migrate_to_goat=True)


def test_goat_worker_cli_and_physical_gate_are_explicit(tmp_path):
    from types import SimpleNamespace
    from scripts import run_yat_mlm_continuation as worker, train_encoder
    parent = tmp_path / 'parent'
    parent.mkdir()
    source = EncoderConfig(attention_score='yat_softmax')
    values = export(parent, config=source)
    args = SimpleNamespace(root=tmp_path, parent_metadata_sha256=values[2],
        parent_manifest_sha256=values[3], steps=100000, batch_size=64,
        accumulation_steps=4, output='gs://bucket/goat-checkpoints', save_every=250,
        migrate_to_goat=True)
    cli = train_encoder.parser().parse_args(worker.training_arguments(args))
    assert cli.attention_score == 'goat' and cli.migrate_to_goat
    assert cli.initialize_from_public == str(parent)
    assert 'goat-physical-forward-backward-migration' in worker.required_qualification_tests(args)
    args.migrate_to_goat = False
    cli = train_encoder.parser().parse_args(worker.training_arguments(args))
    assert cli.attention_score == 'yat_softmax' and not cli.migrate_to_goat
    assert len(worker.required_qualification_tests(args)) == 3


@pytest.mark.parametrize('exit_code', [0, 1])
def test_supervisor_preflight_uses_exited_cpu_metadata_child(tmp_path, monkeypatch, exit_code):
    from types import SimpleNamespace
    from scripts import run_yat_mlm_continuation as worker
    events = []
    class Process:
        pid = 12345
        def wait(self, **kwargs):
            events.append(('wait', kwargs))
            return exit_code
    def popen(command, **kwargs):
        events.append(('start', command, kwargs))
        assert '--preflight-only' in command and '--migrate-to-goat' in command
        assert kwargs['env']['JAX_PLATFORMS'] == 'cpu'
        assert kwargs['start_new_session'] is True
        return Process()
    monkeypatch.setattr(worker.subprocess, 'Popen', popen)
    monkeypatch.setattr(worker.os, 'killpg', lambda pid, sig: events.append(('kill', pid, sig)))
    monkeypatch.setattr(worker, 'preflight', lambda _args: pytest.fail('Supervisor imported model admission'))
    args = SimpleNamespace(root=tmp_path, output='gs://bucket/goat', parent_metadata_sha256='a'*64,
        parent_manifest_sha256='b'*64, steps=100000, batch_size=64, accumulation_steps=4,
        save_every=250, max_seconds=3600, migrate_to_goat=True)
    if exit_code:
        with pytest.raises(worker.subprocess.CalledProcessError):
            worker.isolated_preflight(args)
    else:
        worker.isolated_preflight(args)
    assert events[1] == ('wait', {'timeout': 900})
    assert events[-2][0] == 'kill' and events[-1] == ('wait', {})


def test_supervisor_main_dispatches_only_isolated_preflight(monkeypatch):
    from scripts import run_yat_mlm_continuation as worker
    class StopAfterAdmission(Exception):
        pass
    monkeypatch.setattr(worker, 'parent_config', lambda _args: {})
    monkeypatch.setattr(worker, 'preflight', lambda _args: pytest.fail('In-process admission leases TPU'))
    def isolated(_args):
        raise StopAfterAdmission()
    monkeypatch.setattr(worker, 'isolated_preflight', isolated)
    with pytest.raises(StopAfterAdmission):
        worker.main(['--root', '/tmp/metadata-test', '--output', 'gs://bucket/goat',
            '--parent-metadata-sha256', 'a'*64, '--parent-manifest-sha256', 'b'*64,
            '--migrate-to-goat', '--qualify-only'])


def test_random_goat_uses_reference_only_and_never_imports_weights(tmp_path):
    from types import SimpleNamespace
    from scripts import run_yat_mlm_continuation as worker, train_encoder
    parent = tmp_path / 'parent'
    parent.mkdir()
    values = export(parent, config=EncoderConfig(attention_score='yat_softmax'))
    # The random stage must work without any trained tensor artifact.
    (parent / 'model.safetensors').unlink()
    args = SimpleNamespace(root=tmp_path, parent_metadata_sha256=values[2],
        parent_manifest_sha256=values[3], steps=100000, batch_size=64,
        accumulation_steps=4, output='gs://bucket/random-goat', save_every=250,
        random_init=True, migrate_to_goat=False)
    cli = train_encoder.parser().parse_args(worker.training_arguments(args))
    assert cli.attention_score == 'goat' and cli.seed == 1006
    assert cli.learning_rate == 3e-4
    assert cli.initialize_from_public is cli.initialize_from_checkpoint is cli.pretrained is None
    assert cli.initialize_public_metadata_sha256 is cli.initialize_public_manifest_sha256 is None
    assert not cli.migrate_to_goat
    assert worker.stage_options(args) == ['--random-init']
    args.learning_rate = 2e-4
    assert worker.stage_options(args) == ['--random-init', '--learning-rate', '0.0002']
    assert train_encoder.parser().parse_args(worker.training_arguments(args)).learning_rate == 2e-4
    assert worker.baseline_action(args) == 'evaluate-random'
    assert 'mlm-random-initialization-update' in worker.required_qualification_tests(args)
    args.migrate_to_goat = True
    with pytest.raises(ValueError, match='cannot migrate'):
        worker.training_arguments(args)


def test_random_and_migration_cli_are_mutually_exclusive():
    from scripts import run_yat_mlm_continuation as worker
    with pytest.raises(SystemExit):
        worker.main(['--root', '/tmp/no-reading', '--output', 'gs://bucket/goat',
            '--parent-metadata-sha256', 'a'*64, '--parent-manifest-sha256', 'b'*64,
            '--random-init', '--migrate-to-goat'])


def test_random_baseline_constructor_uses_same_seed_without_restore(monkeypatch):
    from types import SimpleNamespace
    from flax import nnx
    from scripts import run_yat_mlm_continuation as worker
    calls = []
    # Metadata-only fakes: no JAX arrays, model construction or execution.
    monkeypatch.setattr(nnx, 'Rngs', lambda seed: ('host-only-seed', seed))
    evaluator = SimpleNamespace(ModernBert=lambda config, **kw: calls.append((config, kw)),
        restore_model_from_checkpoint=lambda *a, **kw: pytest.fail('Must not restore weights'))
    worker.install_random_baseline(evaluator)
    config = EncoderConfig(attention_score='goat')
    evaluator.ModernBert(config, rngs='ignored-evaluator-seed-zero')
    assert calls == [(config, {'rngs': ('host-only-seed', 1006)})]
    assert evaluator.restore_model_from_checkpoint('fake model', 'fake path') is None
    with pytest.raises(ValueError, match='GOAT checkpoint'):
        evaluator.ModernBert(EncoderConfig())


def test_random_quality_gate_is_bound_to_random_initializer():
    from types import SimpleNamespace
    from scripts import run_yat_mlm_continuation as worker
    args = SimpleNamespace(random_init=True)
    baseline = dict(masked_token_loss=10., selected_rows_sha256='same', masked_tokens=100)
    worker.validate_quality_report(args, baseline | {'masked_token_loss': 10.5}, baseline)
    with pytest.raises(RuntimeError, match='5%'):
        worker.validate_quality_report(args, baseline | {'masked_token_loss': 10.51}, baseline)
    with pytest.raises(RuntimeError, match='row identity'):
        worker.validate_quality_report(args, baseline | {'masked_tokens': 101}, baseline)
    args.random_init = False
    worker.validate_quality_report(args, baseline | {'masked_token_loss': 11.5}, baseline)
    assert worker.baseline_action(args) == 'evaluate-parent'


def test_random_continuation_retains_its_own_baseline_and_mode_identity(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import run_yat_mlm_continuation as worker
    args = SimpleNamespace(root=tmp_path, random_init=True, parent_metadata_sha256='c'*64,
                           parent_manifest_sha256='d'*64)
    (tmp_path / 'source').mkdir()
    (tmp_path / 'source/code.py').write_text('frozen = True\n')
    logs = tmp_path / 'mlm-run-evidence'
    logs.mkdir()
    monkeypatch.setenv('FLAXCHAT_RUN_MANIFEST_SHA256', 'a'*64)
    monkeypatch.setenv('FLAXCHAT_RUNTIME_LOCK_SHA256', 'b'*64)
    monkeypatch.setattr(worker, 'training_arguments', lambda _args: ['random', 'goat'])
    baseline, evaluated = {'masked_token_loss': 12.}, {'masked_token_loss': 11.8}
    xml = b'<testsuites><testsuite tests="14" failures="0" skipped="0" errors="0"/></testsuites>'
    (logs / 'goat-physical-tests.xml').write_bytes(xml)
    qualification = worker.qualification_identity(args) | {
        'random_initializer_evaluation_sha256': worker.canonical_hash(baseline),
        'step8_evaluation_sha256': worker.canonical_hash(evaluated),
        'goat_physical_tests_sha256': worker.hashlib.sha256(xml).hexdigest(),
        'tests': [{'id': name, 'status': 'passed'} for name in worker.required_qualification_tests(args)]}
    assert qualification['random_initialization']['weights_loaded'] is False
    (tmp_path / 'mlm-qualification.json').write_text(json.dumps(qualification))
    (logs / 'evaluate-random-4.json').write_text(json.dumps(baseline))
    (logs / 'evaluate-8.json').write_text(json.dumps(evaluated))
    (logs / 'status.json').write_text(json.dumps({'status': 'qualified', 'completed_step': 8,
        'physical_random_update_resume_qualified': True}))
    assert worker.verify_continuation(args, logs)[1] == baseline
    args.parent_metadata_sha256 = 'e'*64
    with pytest.raises(ValueError, match='no longer matches'):
        worker.verify_continuation(args, logs)


def test_input_goat_stage_uses_no_parent_weights_and_binds_score_source(tmp_path):
    from types import SimpleNamespace
    from scripts import run_yat_mlm_continuation as worker
    from scripts import train_encoder
    parent = tmp_path / 'parent'
    parent.mkdir()
    _, _, metadata_hash, manifest_hash = export(parent, config=EncoderConfig(ffn_type='yat_glu'))
    (parent / 'model.safetensors').unlink()
    args = SimpleNamespace(root=tmp_path, parent_metadata_sha256=metadata_hash,
        parent_manifest_sha256=manifest_hash, steps=100000, batch_size=64,
        accumulation_steps=4, output='gs://bucket/input-goat/checkpoints', save_every=250,
        random_init=True, migrate_to_goat=False, goat_score_source='input')
    cli = train_encoder.parser().parse_args(worker.training_arguments(args))
    assert cli.attention_score == 'goat_input'
    assert cli.initialize_from_public is None and cli.pretrained is None
    assert worker.stage_options(args) == ['--random-init', '--goat-score-source', 'input']
    assert 'goat-input-score-value-separation' in worker.required_qualification_tests(args)
    args.random_init = False
    with pytest.raises(ValueError, match='explicit new random stage'):
        worker.training_arguments(args)


def test_input_goat_physical_gate_requires_named_cases_and_no_skips(tmp_path):
    import xml.etree.ElementTree as ET
    from scripts.run_yat_mlm_continuation import validate_goat_test_report
    root = ET.Element('testsuites')
    suite = ET.SubElement(root, 'testsuite', tests='30', skipped='0', failures='0', errors='0')
    for i in range(30):
        ET.SubElement(suite, 'testcase', name=f'unrelated_{i}')
    xml = tmp_path / 'physical.xml'
    def write():
        ET.ElementTree(root).write(xml)
    write()
    with pytest.raises(RuntimeError, match='cases are missing'):
        validate_goat_test_report(xml, input_scores=True)
    for case, name in zip(list(suite)[:3], (
        'test_input_goat_geometry_is_independent_of_value_projection[0]',
        'test_input_goat_geometry_is_independent_of_value_projection[1]',
        'test_migrated_bf16_mlm_update_checkpoint_exact_resume[goat_input]'), strict=True):
        case.set('name', name)
    write()
    validate_goat_test_report(xml, input_scores=True)
    suite.set('skipped', '1')
    write()
    with pytest.raises(RuntimeError, match='did not all execute'):
        validate_goat_test_report(xml, input_scores=True)
