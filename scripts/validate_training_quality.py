"""Repository CLI alias for the packaged training-quality gate."""
from flaxchat.training_quality import evaluate, main

__all__ = ['evaluate', 'main']

if __name__ == '__main__':
    raise SystemExit(main())
