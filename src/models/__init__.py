from .baselines import build_baseline


def build_model_adapter(*args, **kwargs):
    from .adapters import build_model_adapter as _build_model_adapter

    return _build_model_adapter(*args, **kwargs)


def build_seq2seq_behavior_model(*args, **kwargs):
    from .behavior_seq2seq import HfSeq2SeqBehaviorModel as _HfSeq2SeqBehaviorModel

    return _HfSeq2SeqBehaviorModel(*args, **kwargs)


__all__ = ["build_baseline", "build_model_adapter", "build_seq2seq_behavior_model"]
