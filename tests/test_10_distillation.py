import torch
import torch.nn.functional as F

from llm_koans import koans as K


def snapshot(model):
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def any_changed(before, after):
    return any(not torch.allclose(before[name], after[name]) for name in before)


def all_unchanged(before, after):
    return all(torch.allclose(before[name], after[name]) for name in before)


def test_distillation_kl_uses_soft_teacher_distribution_not_argmax_label():
    teacher_logits = torch.tensor([[3.0, 1.0, -2.0]])
    student_logits = torch.tensor([[0.0, 2.0, 0.0]])
    temperature = 2.0

    actual = K.distillation_kl_loss(student_logits, teacher_logits, temperature=temperature)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    expected = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1).mean() * temperature**2

    assert torch.allclose(actual, expected)


def test_blended_distillation_loss_combines_teacher_and_hard_labels():
    teacher_logits = torch.tensor([[3.0, 1.0, -2.0], [0.0, 2.0, 1.0]])
    student_logits = torch.tensor([[0.0, 2.0, 0.0], [1.0, 0.0, 2.0]])
    hard_labels = torch.tensor([0, 2])

    actual = K.blended_distillation_loss(student_logits, teacher_logits, hard_labels, alpha=0.7, temperature=2.0)
    expected = 0.7 * K.distillation_kl_loss(student_logits, teacher_logits, temperature=2.0)
    expected = expected + 0.3 * F.cross_entropy(student_logits, hard_labels)

    assert torch.allclose(actual, expected)


def test_distillation_step_updates_student_but_not_teacher():
    torch.manual_seed(24)
    teacher = K.TinyBaseTextClassifier(vocab_size=20, d_model=8, num_classes=3)
    student = K.TinyBaseTextClassifier(vocab_size=20, d_model=8, num_classes=3)
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.05)
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [1, 1, 1]])
    hard_labels = torch.tensor([0, 1, 2])

    teacher_before = snapshot(teacher)
    student_before = snapshot(student)
    loss = K.distillation_step(student, teacher, optimizer, token_ids, hard_labels, alpha=0.6, temperature=2.0)
    teacher_after = snapshot(teacher)
    student_after = snapshot(student)

    assert loss.ndim == 0
    assert loss.item() > 0
    assert all_unchanged(teacher_before, teacher_after)
    assert any_changed(student_before, student_after)
