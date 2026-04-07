"""
HumanEval — code generation benchmark (164 problems).
https://huggingface.co/datasets/openai/humaneval

Evaluates by extracting Python code from completions and running test cases.
"""

import re
from datasets import load_dataset
from tasks.common import Task


def extract_code(completion):
    """Extract Python code from a completion (handles markdown blocks)."""
    # Try markdown code block first
    match = re.search(r'```python\s*\n(.*?)```', completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```\s*\n(.*?)```', completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Otherwise return as-is (might be raw code)
    return completion.strip()


def execute_code(code, timeout_seconds=5):
    """Execute Python code in a subprocess for isolation. Returns True if exit code 0."""
    import subprocess
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = subprocess.run(
                ['python', f.name],
                capture_output=True, text=True, timeout=timeout_seconds,
            )
        import os
        os.unlink(f.name)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        try:
            import os
            os.unlink(f.name)
        except Exception:
            pass
        return False


class HumanEval(Task):
    """OpenAI HumanEval benchmark — 164 coding problems."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("openai/humaneval", split="test")

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        prompt = row["prompt"]  # function signature + docstring
        test = row["test"]      # test cases
        entry_point = row["entry_point"]
        canonical = row["canonical_solution"]

        # Build conversation: user gives the prompt, assistant should complete
        user_msg = f"Complete the following Python function:\n\n```python\n{prompt}```"
        assistant_msg = canonical

        return {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            "prompt": prompt,
            "test": test,
            "entry_point": entry_point,
        }

    def evaluate(self, conversation, assistant_response):
        """Extract code, combine with prompt + tests, execute."""
        prompt = conversation["prompt"]
        test = conversation["test"]
        entry_point = conversation["entry_point"]

        # Extract code from response
        code = extract_code(assistant_response)

        # Build full program: prompt + completion + tests
        # Extract imports from prompt
        imports = []
        for line in prompt.split('\n'):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                imports.append(stripped)

        full_code = "\n".join(imports) + "\n" if imports else ""
        full_code += prompt + code + "\n" + test + f"\ncheck({entry_point})\n"

        return int(execute_code(full_code, timeout_seconds=5))
