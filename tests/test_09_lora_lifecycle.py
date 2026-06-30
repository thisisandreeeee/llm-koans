import torch
import torch.nn.functional as F

from llm_koans import koans as K


def snapshot(model):
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def changed(before, after, name):
    return not torch.allclose(before[name], after[name])


def unchanged(before, after, name):
    return torch.allclose(before[name], after[name])


def trainable_names(model):
    return {name for name, param in model.named_parameters() if param.requires_grad}


def test_lora_adapter_finetuning_updates_adapter_not_frozen_base():
    torch.manual_seed(21)
    model = K.TinyBaseTextClassifier(vocab_size=20, d_model=8, num_classes=2)

    K.add_lora_classifier_adapter(model, rank=2, alpha=4.0)

    assert isinstance(model.classifier, K.LoRALinear)
    assert trainable_names(model) == {"classifier.A", "classifier.B"}

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.1)
    token_ids = torch.tensor([[1, 2, 3], [3, 2, 1], [4, 4, 4], [5, 6, 7]])
    labels = torch.tensor([0, 1, 0, 1])

    before = snapshot(model)
    loss = K.supervised_finetune_step(model, optimizer, token_ids, labels)
    after = snapshot(model)

    assert loss.item() > 0
    assert unchanged(before, after, "embedding.weight")
    assert unchanged(before, after, "encoder.weight")
    assert unchanged(before, after, "encoder.bias")
    assert unchanged(before, after, "classifier.weight")
    assert unchanged(before, after, "classifier.bias")
    assert changed(before, after, "classifier.B")


def test_lora_forward_is_frozen_base_plus_scaled_low_rank_delta():
    base = torch.nn.Linear(3, 2)
    with torch.no_grad():
        base.weight.copy_(torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]]))
        base.bias.copy_(torch.tensor([0.5, -0.5]))

    layer = K.LoRALinear(base, rank=2, alpha=4.0)
    with torch.no_grad():
        layer.A.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        layer.B.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))

    x = torch.tensor([[1.0, 2.0, 3.0]])

    actual = layer(x)
    expected = F.linear(x, base.weight, base.bias) + (x @ layer.A @ layer.B) * (4.0 / 2)

    assert torch.allclose(actual, expected)
    assert torch.allclose(actual, torch.tensor([[20.5, 33.5]]))


def test_lora_adapter_state_saves_and_loads_only_adapter_weights():
    torch.manual_seed(23)
    base = torch.nn.Linear(4, 3)
    source = K.LoRALinear(base, rank=2, alpha=8.0)
    target = K.LoRALinear(base, rank=2, alpha=8.0)
    with torch.no_grad():
        source.A.fill_(0.25)
        source.B.fill_(0.75)
        target.A.zero_()
        target.B.zero_()

    adapter_state = K.lora_adapter_state(source)
    K.load_lora_adapter_state(target, adapter_state)

    assert set(adapter_state) == {"A", "B"}
    assert torch.allclose(target.A, source.A)
    assert torch.allclose(target.B, source.B)


def test_merge_lora_linear_preserves_outputs_without_adapter_module():
    base = torch.nn.Linear(3, 2)
    layer = K.LoRALinear(base, rank=2, alpha=4.0)
    with torch.no_grad():
        layer.A.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        layer.B.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))

    x = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])

    merged = K.merge_lora_linear(layer)

    assert isinstance(merged, torch.nn.Linear)
    assert torch.allclose(merged(x), layer(x), atol=1e-6)
