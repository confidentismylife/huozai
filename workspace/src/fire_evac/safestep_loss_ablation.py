"""Scoped loss-only variants of the immutable frequency diagnostic engine."""
from . import safestep_frequency_ablation as engine

REFERENCE_LOSS = engine.public_vector_loss
REFERENCE_SCHEDULE = engine.schedule


def variant_loss(variant, *args):
    original, q, labels, anchor_sse = REFERENCE_LOSS(*args)
    if variant == "full_anchor":
        loss = original
    elif variant == "selected":
        loss = ((q - labels) ** 2).mean()
    elif variant == "selected_quarter":
        loss = ((q - labels) ** 2).mean() / 4
    else:
        raise ValueError(variant)
    return loss, q, labels, anchor_sse


def run(source, cfg, arm, seed, out, budget_override=None):
    # Each formal run executes in its own process; preflights are sequential.
    # Restore bindings even on failure, without changing the frozen source file.
    previous_loss, previous_schedule = engine.public_vector_loss, engine.schedule
    engine.public_vector_loss = lambda *a: variant_loss(arm, *a)
    engine.schedule = lambda _arm, *a: REFERENCE_SCHEDULE("every_step", *a)
    try:
        return engine.run(source, cfg, arm, seed, out, budget_override)
    finally:
        engine.public_vector_loss, engine.schedule = previous_loss, previous_schedule
