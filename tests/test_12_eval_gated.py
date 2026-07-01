import pytest
import torch

from llm_koans import koans as K


def clone_state(model):
    return {name: param.detach().clone() for name, param in model.state_dict().items()}


def test_classification_accuracy_counts_argmax_predictions():
    model = K.TinyBaseTextClassifier(vocab_size=8, d_model=4, num_classes=2)
    token_ids = torch.tensor([[1, 1], [2, 2], [3, 3]])
    labels = torch.tensor([0, 1, 0])

    with torch.no_grad():
        model.embedding.weight.zero_()
        model.encoder.weight.zero_()
        model.encoder.bias.zero_()
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([1.0, 0.0]))

    assert K.classification_accuracy(model, token_ids, labels) == pytest.approx(2 / 3)


def test_eval_gate_accepts_candidate_that_clears_threshold_and_improves():
    model = K.TinyBaseTextClassifier(vocab_size=8, d_model=4, num_classes=2)
    token_ids = torch.tensor([[1, 1], [2, 2], [3, 3]])
    labels = torch.tensor([1, 1, 1])

    with torch.no_grad():
        model.embedding.weight.zero_()
        model.encoder.weight.zero_()
        model.encoder.bias.zero_()
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([1.0, 0.0]))

    candidate = clone_state(model)
    candidate["classifier.bias"] = torch.tensor([0.0, 1.0])

    accepted = K.accept_candidate_if_improves(model, candidate, token_ids, labels, min_accuracy=1.0, min_delta=0.5)

    assert accepted is True
    assert torch.allclose(model.classifier.bias, torch.tensor([0.0, 1.0]))


def test_eval_gate_rejects_candidate_and_restores_baseline_when_validation_regresses():
    model = K.TinyBaseTextClassifier(vocab_size=8, d_model=4, num_classes=2)
    token_ids = torch.tensor([[1, 1], [2, 2], [3, 3]])
    labels = torch.tensor([0, 0, 0])

    with torch.no_grad():
        model.embedding.weight.zero_()
        model.encoder.weight.zero_()
        model.encoder.bias.zero_()
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([1.0, 0.0]))

    baseline = clone_state(model)
    candidate = clone_state(model)
    candidate["classifier.bias"] = torch.tensor([0.0, 1.0])

    accepted = K.accept_candidate_if_improves(model, candidate, token_ids, labels, min_accuracy=0.9, min_delta=0.0)

    assert accepted is False
    for name, tensor in baseline.items():
        assert torch.allclose(model.state_dict()[name], tensor)
