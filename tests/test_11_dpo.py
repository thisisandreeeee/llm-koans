import torch
import torch.nn.functional as F

from llm_koans import koans as K


def snapshot(model):
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def any_changed(before, after):
    return any(not torch.allclose(before[name], after[name]) for name in before)


def all_unchanged(before, after):
    return all(torch.allclose(before[name], after[name]) for name in before)


def test_sequence_logprobs_gathers_target_token_log_probs_and_masks_prompt():
    logits = torch.tensor([
        [
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    ])
    target_ids = torch.tensor([[0, 1, 2]])
    mask = torch.tensor([[False, True, True]])

    actual = K.sequence_logprobs(logits, target_ids, mask)
    expected_token_logps = F.log_softmax(logits, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    expected = expected_token_logps[:, 1:].sum(dim=-1)

    assert torch.allclose(actual, expected)


def test_dpo_loss_uses_policy_improvement_over_reference_model():
    policy_chosen = torch.tensor([-1.0, -2.0])
    policy_rejected = torch.tensor([-3.0, -2.5])
    reference_chosen = torch.tensor([-1.2, -2.0])
    reference_rejected = torch.tensor([-2.2, -2.1])

    actual = K.dpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=0.2,
    )

    policy_gap = policy_chosen - policy_rejected
    reference_gap = reference_chosen - reference_rejected
    expected = -F.logsigmoid(0.2 * (policy_gap - reference_gap)).mean()

    assert torch.allclose(actual, expected)


def test_dpo_step_updates_policy_but_not_reference():
    torch.manual_seed(25)
    policy = K.TinyCausalLM(vocab_size=10, d_model=6)
    reference = K.TinyCausalLM(vocab_size=10, d_model=6)
    reference.load_state_dict(policy.state_dict())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1)

    chosen_ids = torch.tensor([[1, 2, 3, 4], [1, 5, 6, 7]])
    rejected_ids = torch.tensor([[1, 2, 3, 8], [1, 5, 6, 9]])
    chosen_mask = torch.tensor([[False, False, True, True], [False, False, True, True]])
    rejected_mask = torch.tensor([[False, False, True, True], [False, False, True, True]])

    policy_before = snapshot(policy)
    reference_before = snapshot(reference)
    loss = K.dpo_step(policy, reference, optimizer, chosen_ids, rejected_ids, chosen_mask, rejected_mask, beta=0.2)
    policy_after = snapshot(policy)
    reference_after = snapshot(reference)

    assert loss.ndim == 0
    assert all_unchanged(reference_before, reference_after)
    assert any_changed(policy_before, policy_after)
