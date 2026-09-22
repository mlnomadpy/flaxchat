"""Offline SFT/RL resume contracts with real optimizer/checkpoint execution."""
from dataclasses import asdict
import json
import shutil

from flax import nnx
import jax
import optax

from flaxchat.config import GPTConfig
from flaxchat.gpt import GPT
from flaxchat.dataloader import _tokenizer_identity
from flaxchat.checkpoint import create_checkpoint_manager, save_checkpoint


class Tokenizer:
    def get_vocab_size(self): return 32
    def get_bos_token_id(self): return 0
    def render_conversation(self, conversation, max_tokens):
        ids = [1, 2, 3, conversation['answer'], 5, 6]
        return ids[:max_tokens], [0, 0, 0, 1, 1, 1][:max_tokens]
    def render_for_completion(self, conversation): return [1, 2]
    def decode(self, tokens): return ' '.join(map(str, tokens))
    def encode_special(self, token): return 0


def base_checkpoint(root):
    config = GPTConfig(sequence_len=8, vocab_size=32, n_layer=1, n_head=2,
                       n_kv_head=2, n_embd=32, standard_gpt=True)
    model = GPT(config, rngs=nnx.Rngs(3))
    optimizer = nnx.Optimizer(model, optax.sgd(.01), wrt=nnx.Param)
    manager = create_checkpoint_manager(str(root / 'base_checkpoints/d1'), async_checkpointing=False)
    try:
        save_checkpoint(manager, 0, model, optimizer, {'model_config': asdict(config), 'step': 0, 'tokenizer_identity': _tokenizer_identity(Tokenizer())})
        manager.wait_until_finished()
    finally:
        manager.close()


def test_prepared_training_artifact_handoff_to_eval_and_sft(tmp_path, monkeypatch):
    import numpy as np
    from flaxchat.token_pool import write_streaming_pool
    from flaxchat.tokenizer import ByteTokenizer
    from flaxchat.stages import eval as evaluation, sft
    from scripts.train_gpt2 import main as train
    from tasks.common import Task
    import tasks.mmlu
    tokenizer = ByteTokenizer()
    tokenizer_dir = tmp_path / 'tokenizer'
    tokenizer.save(str(tokenizer_dir))
    batch = jax.device_count()
    count = batch * 32 * 2 + 1
    manifest = write_streaming_pool(tmp_path / 'pool',
        {'train': [np.arange(count, dtype=np.int32) % 128],
         'validation': [np.arange(batch * 32 + 1, dtype=np.int32) % 128]},
        {'tokenizer_identity': _tokenizer_identity(tokenizer)}, tokenizer.get_vocab_size(), shard_tokens=31)
    checkpoint = tmp_path / 'prepared-checkpoints'
    assert train(['--token-manifest', str(manifest), '--depth', '1', '--seq-len', '32',
        '--tokens', str(count - 1), '--global-batch-size', str(batch), '--warmup-steps', '0',
        '--eval-every', '2', '--save-every', '2', '--ckpt-dir', str(checkpoint),
        '--artifact-dir', str(tmp_path / 'training')]) == 0
    class Example(Task):
        def __init__(self, **kwargs): super().__init__(stop=kwargs.get('stop'))
        def num_examples(self): return 1
        def get_example(self, index): return {'messages': [{'content': 'x'}], 'letters': ['A', 'B']}
        def evaluate(self, *args): return False
    monkeypatch.setattr(tasks.mmlu, 'MMLU', Example)
    assert evaluation.run(evaluation.EvalRequest(tasks='mmlu', checkpoint_dir=str(checkpoint),
        tokenizer_dir=str(tokenizer_dir))).metrics['mmlu']['total'] == 1
    monkeypatch.setattr(sft, 'get_base_dir', lambda: str(tmp_path / 'sft-output'))
    conversations = tmp_path / 'conversations.jsonl'
    conversations.write_text(json.dumps({'messages': [
        {'role': 'user', 'content': 'x'}, {'role': 'assistant', 'content': 'y'}]}) + '\n')
    assert sft.run(sft.SFTRequest(base_model='d1', checkpoint_dir=str(checkpoint),
        tokenizer_dir=str(tokenizer_dir), dataset=str(conversations), num_iterations=2,
        batch_size=batch, max_seq_len=32, warmup_steps=0, save_every=2)).exit_code == 0


def compare(full, resumed, stage):
    def manifest(root):
        return json.loads((root / f'{stage}_checkpoints/d1/4/manifest/metadata').read_text())
    left, right = manifest(full), manifest(resumed)
    assert all(left[key] == right[key] for key in ('model_state', 'optimizer_state', 'training_state'))


def prepare_resume(full, resumed, stage):
    shutil.copytree(full / 'base_checkpoints', resumed / 'base_checkpoints')
    shutil.copytree(full / f'{stage}_checkpoints/d1/2', resumed / f'{stage}_checkpoints/d1/2')


def test_sft_exact_resume(tmp_path, monkeypatch):
    from flaxchat.stages import sft
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    base_checkpoint(full)
    monkeypatch.setattr(sft, 'get_base_dir', lambda: str(full))
    monkeypatch.setattr(sft, 'get_tokenizer', Tokenizer)
    monkeypatch.setattr(sft, 'load_conversations', lambda _: [{'answer': 4}, {'answer': 7}])
    kwargs = dict(base_model='d1', num_iterations=4, batch_size=jax.device_count(),
                  max_seq_len=8, warmup_steps=1, save_every=2)
    assert sft.run(sft.SFTRequest(**kwargs)).exit_code == 0
    prepare_resume(full, resumed, 'sft')
    monkeypatch.setattr(sft, 'get_base_dir', lambda: str(resumed))
    assert sft.run(sft.SFTRequest(**kwargs, resume_from_step=2)).exit_code == 0
    compare(full, resumed, 'sft')


def test_rl_exact_resume(tmp_path, monkeypatch):
    from flaxchat.stages import rl
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    base_checkpoint(full)
    class Task:
        def __init__(self, **kwargs): pass
        def __len__(self): return 4
        def __getitem__(self, index): return {'answer': index + 3}
        def reward(self, conv, response): return float(response.endswith('5'))
        def evaluate(self, conv, response): return response.endswith('5')
    class Report:
        def log(self, *args, **kwargs): pass
    monkeypatch.setattr(rl, 'get_base_dir', lambda: str(full))
    monkeypatch.setattr(rl, 'get_tokenizer', Tokenizer)
    monkeypatch.setattr(rl, 'GSM8K', Task)
    monkeypatch.setattr(rl, 'get_report', lambda _: Report())
    monkeypatch.setattr(rl, 'generate_with_cache', lambda model, tokens, seed=0, **kwargs: tokens + [4 + seed % 2])
    kwargs = dict(model='d1', num_epochs=2, examples_per_step=2, num_samples=max(2, jax.device_count()),
                  max_new_tokens=2, eval_every=2, eval_examples=2, save_every=2)
    assert rl.run(rl.RLRequest(**kwargs)).exit_code == 0
    prepare_resume(full, resumed, 'rl')
    monkeypatch.setattr(rl, 'get_base_dir', lambda: str(resumed))
    assert rl.run(rl.RLRequest(**kwargs, resume_from_step=2)).exit_code == 0
    compare(full, resumed, 'rl')
