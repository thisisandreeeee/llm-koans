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


def test_sft_chat_encoding_applies_role_template_and_eos():
    vocab = {"<user>": 1, "<assistant>": 2, "<eos>": 3, "hello": 4, "hi": 5}
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    encoded = K.encode_chat_messages(messages, vocab)

    assert torch.equal(encoded, torch.tensor([1, 4, 3, 2, 5, 3]))


def test_sft_labels_train_only_on_assistant_response_tokens():
    input_ids = torch.tensor([1, 4, 3, 2, 5, 6, 3])

    labels = K.assistant_only_labels(input_ids, assistant_token_id=2, eos_token_id=3)

    assert torch.equal(labels, torch.tensor([-100, -100, -100, -100, 5, 6, 3]))


def test_sft_step_uses_shifted_next_token_loss_and_ignores_prompt_tokens():
    torch.manual_seed(22)
    model = K.TinyCausalLM(vocab_size=8, d_model=6)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    input_ids = torch.tensor([[1, 4, 3, 2, 5, 6, 3]])
    labels = torch.tensor([[-100, -100, -100, -100, 5, 6, 3]])

    before = snapshot(model)
    loss = K.sft_step(model, optimizer, input_ids, labels)
    after = snapshot(model)

    assert loss.ndim == 0
    assert loss.item() > 0
    assert any(changed(before, after, name) for name in before)


def test_classifier_tuning_freezes_base_and_trains_only_head():
    torch.manual_seed(20)
    model = K.TinyBaseTextClassifier(vocab_size=20, d_model=8, num_classes=2)

    K.freeze_base_for_classifier_tuning(model)

    assert trainable_names(model) == {"classifier.weight", "classifier.bias"}

    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.4)
    token_ids = torch.tensor([[1, 2, 3], [3, 2, 1], [4, 4, 4], [5, 6, 7]])
    labels = torch.tensor([0, 1, 0, 1])

    before = snapshot(model)
    loss = K.supervised_finetune_step(model, optimizer, token_ids, labels)
    after = snapshot(model)

    assert loss.ndim == 0
    assert unchanged(before, after, "embedding.weight")
    assert unchanged(before, after, "encoder.weight")
    assert unchanged(before, after, "encoder.bias")
    assert changed(before, after, "classifier.weight")
    assert changed(before, after, "classifier.bias")
