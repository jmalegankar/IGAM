"""Contract tests for the RecurrentCell interface.

Every cell in CELL_FACTORIES (see conftest.py) is run through the same
battery of contract checks. Add a new cell to that registry and the
entire suite runs against it for free.

Tests are split by what they cover:
  - TestInitState         — init_state shapes, device, dtype, discovery
  - TestStep              — step output shape, episode_start reset
  - TestForwardSequence   — sequence stacking, side-output key stability
  - TestHelpers           — apply_episode_mask, detach_state
  - TestGradientFlow      — brief training reduces loss (sanity)
"""

from __future__ import annotations

import torch
from torch import Tensor

from memrl.cell import apply_episode_mask, detach_state


# Test config — kept small for speed; the contract tests don't need big sizes.
INPUT_SIZE = 12
HIDDEN_SIZE = 16
BATCH_SIZE = 4
SEQ_LEN = 5


def _make_cell(factory):
    return factory(INPUT_SIZE, HIDDEN_SIZE)


# --- init_state ------------------------------------------------------------


class TestInitState:
    def test_returns_dict_of_tensors_with_batch_dim(self, cell_name_and_factory):
        name, factory = cell_name_and_factory
        cell = _make_cell(factory)
        s = cell.init_state(BATCH_SIZE)
        assert isinstance(s, dict), f"{name}: init_state must return a dict"
        # Memoryless is stateless by design: an empty dict is its contract.
        if name != "Memoryless":
            assert len(s) > 0, f"{name}: empty state dict"
        for k, v in s.items():
            assert isinstance(v, Tensor), f"{name}: state[{k!r}] not a Tensor"
            assert v.shape[0] == BATCH_SIZE, f"{name}: state[{k!r}] batch dim {v.shape[0]} != {BATCH_SIZE}"

    def test_device_inferred_when_none(self, cell_name_and_factory):
        """init_state(B) with no device should infer from cell parameters."""
        name, factory = cell_name_and_factory
        cell = _make_cell(factory)
        s = cell.init_state(BATCH_SIZE)
        expected_device = next(cell.parameters()).device
        for k, v in s.items():
            assert v.device == expected_device, \
                f"{name}: state[{k!r}] on {v.device}, expected {expected_device}"

    def test_explicit_device(self, cell_name_and_factory, device):
        name, factory = cell_name_and_factory
        cell = _make_cell(factory).to(device)
        s = cell.init_state(BATCH_SIZE, device=device)
        for v in s.values():
            assert v.device == device, f"{name}: explicit device not honored"

    def test_dtype_propagation(self, cell_name_and_factory):
        name, factory = cell_name_and_factory
        cell = _make_cell(factory)
        s = cell.init_state(BATCH_SIZE, dtype=torch.float64)
        for k, v in s.items():
            assert v.dtype == torch.float64, \
                f"{name}: state[{k!r}] dtype {v.dtype}, expected float64"

    def test_shape_discovery_with_batch_one(self, cell_name_and_factory):
        """init_state(batch_size=1) is the rollout buffer's shape-discovery path."""
        name, factory = cell_name_and_factory
        cell = _make_cell(factory)
        s = cell.init_state(1)
        for k, v in s.items():
            assert v.shape[0] == 1, f"{name}: state[{k!r}] batch dim != 1"
            assert v.ndim >= 1, f"{name}: state[{k!r}] is a scalar"
            assert all(d > 0 for d in v.shape[1:]), \
                f"{name}: state[{k!r}] has empty trailing dim"


# --- step ------------------------------------------------------------------


class TestStep:
    def test_output_shape_and_finite(self, cell_name_and_factory):
        name, factory = cell_name_and_factory
        cell = _make_cell(factory)
        s = cell.init_state(BATCH_SIZE)
        x = torch.randn(BATCH_SIZE, INPUT_SIZE)
        y, _, _ = cell.step(x, s)
        assert y.shape == (BATCH_SIZE, HIDDEN_SIZE), \
            f"{name}: y.shape {y.shape} != ({BATCH_SIZE}, {HIDDEN_SIZE})"
        assert torch.isfinite(y).all(), f"{name}: y has NaN/Inf"

    def test_state_keys_and_shapes_preserved(self, cell_name_and_factory):
        name, factory = cell_name_and_factory
        cell = _make_cell(factory)
        s = cell.init_state(BATCH_SIZE)
        x = torch.randn(BATCH_SIZE, INPUT_SIZE)
        _, s_new, _ = cell.step(x, s)
        assert set(s_new.keys()) == set(s.keys()), \
            f"{name}: state keys changed from {set(s.keys())} to {set(s_new.keys())}"
        for k in s:
            assert s_new[k].shape == s[k].shape, \
                f"{name}: state[{k!r}] shape changed {s[k].shape} → {s_new[k].shape}"

    def test_episode_start_resets_to_fresh_init(self, cell_name_and_factory):
        """Calling step with episode_start=True should match starting from init_state."""
        name, factory = cell_name_and_factory
        cell = _make_cell(factory)
        x = torch.randn(BATCH_SIZE, INPUT_SIZE)

        s_arbitrary = {k: torch.randn_like(v) for k, v in cell.init_state(BATCH_SIZE).items()}
        es_all = torch.ones(BATCH_SIZE, dtype=torch.bool)
        y_reset, _, _ = cell.step(x, s_arbitrary, episode_start=es_all)

        s_fresh = cell.init_state(BATCH_SIZE)
        y_fresh, _, _ = cell.step(x, s_fresh)

        assert torch.allclose(y_reset, y_fresh, atol=1e-5), \
            f"{name}: episode_start=all-True doesn't match starting from init_state"

    def test_partial_episode_start_only_affects_marked_rows(self, cell_name_and_factory):
        """Mask=[True, False, True, False]: rows 0,2 reset; rows 1,3 keep their state."""
        name, factory = cell_name_and_factory
        cell = _make_cell(factory)
        x = torch.randn(BATCH_SIZE, INPUT_SIZE)

        # Build a state with non-zero rows.
        s_nonzero = {k: torch.randn_like(v) for k, v in cell.init_state(BATCH_SIZE).items()}
        mask = torch.tensor([True, False, True, False])

        # Fork RNG: the two paths each call `step` once, which may consume
        # random samples (SHM does). They must consume from the same RNG state
        # to produce comparable outputs.
        rng_state = torch.random.get_rng_state()
        y_partial, _, _ = cell.step(x, s_nonzero, episode_start=mask)

        # Independently: reset rows 0,2 by hand, leave 1,3 alone, run step with no mask.
        torch.random.set_rng_state(rng_state)
        s_zero = cell.init_state(BATCH_SIZE)
        s_manual = {k: v.clone() for k, v in s_nonzero.items()}
        for k in s_manual:
            s_manual[k][0] = s_zero[k][0]
            s_manual[k][2] = s_zero[k][2]
        y_manual, _, _ = cell.step(x, s_manual)

        # Both should match exactly.
        assert torch.allclose(y_partial, y_manual, atol=1e-5), \
            f"{name}: partial episode_start reset doesn't match manual reset"


# --- forward_sequence ------------------------------------------------------


class TestForwardSequence:
    def test_output_stacks_over_time(self, cell_name_and_factory):
        name, factory = cell_name_and_factory
        cell = _make_cell(factory)
        x_seq = torch.randn(SEQ_LEN, BATCH_SIZE, INPUT_SIZE)
        s0 = cell.init_state(BATCH_SIZE)
        y_seq, _, _ = cell.forward_sequence(x_seq, s0)
        assert y_seq.shape == (SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE), \
            f"{name}: y_seq.shape {y_seq.shape}"
        assert torch.isfinite(y_seq).all(), f"{name}: y_seq has NaN/Inf"

    def test_matches_explicit_step_loop(self, cell_name_and_factory):
        """forward_sequence MUST be equivalent to stepping manually.

        This is the contract for any future parallel-scan override of
        `forward_sequence` — and a regression guard for the default loop.
        """
        name, factory = cell_name_and_factory
        cell = _make_cell(factory)
        x_seq = torch.randn(SEQ_LEN, BATCH_SIZE, INPUT_SIZE)
        s0 = cell.init_state(BATCH_SIZE)

        # Fork RNG so both paths see the SAME stream of random numbers.
        # Necessary for cells with internal stochasticity (e.g., SHM samples
        # a random θ row per step); deterministic cells are unaffected.
        rng_state = torch.random.get_rng_state()

        # Sequence path.
        y_seq_fwd, s_final_fwd, _ = cell.forward_sequence(x_seq, s0)

        # Manual unroll — restore RNG so it gets the same random samples.
        torch.random.set_rng_state(rng_state)
        s = cell.init_state(BATCH_SIZE)
        ys_manual = []
        for t in range(SEQ_LEN):
            y_t, s, _ = cell.step(x_seq[t], s)
            ys_manual.append(y_t)
        y_seq_manual = torch.stack(ys_manual, dim=0)

        assert torch.allclose(y_seq_fwd, y_seq_manual, atol=1e-5), \
            f"{name}: forward_sequence diverges from manual step loop"
        for k in s_final_fwd:
            assert torch.allclose(s_final_fwd[k], s[k], atol=1e-5), \
                f"{name}: forward_sequence state[{k!r}] diverges from manual"

    def test_side_outputs_stacked_with_stable_keys(self, cell_name_and_factory):
        name, factory = cell_name_and_factory
        cell = _make_cell(factory)
        x_seq = torch.randn(SEQ_LEN, BATCH_SIZE, INPUT_SIZE)
        s0 = cell.init_state(BATCH_SIZE)
        _, _, side_seq = cell.forward_sequence(x_seq, s0)
        # Whatever side outputs the cell emits, they must stack over T.
        for k, v in side_seq.items():
            assert v.shape[0] == SEQ_LEN, \
                f"{name}: side[{k!r}].shape[0] = {v.shape[0]}, expected {SEQ_LEN}"
            assert v.shape[1] == BATCH_SIZE, \
                f"{name}: side[{k!r}] batch dim = {v.shape[1]}"
            assert torch.isfinite(v).all(), f"{name}: side[{k!r}] has NaN/Inf"


# --- helpers ---------------------------------------------------------------


class TestHelpers:
    def test_apply_episode_mask_all_false_is_identity(self, cell_name_and_factory):
        """All-False mask: helper should return the input state object (fast path)."""
        name, factory = cell_name_and_factory
        cell = _make_cell(factory)
        s = cell.init_state(BATCH_SIZE)
        mask = torch.zeros(BATCH_SIZE, dtype=torch.bool)
        s_out = apply_episode_mask(s, mask)
        assert s_out is s, \
            f"{name}: all-False apply_episode_mask should return input object, not a copy"

    def test_apply_episode_mask_zeros_only_selected_rows(self, cell_name_and_factory):
        name, factory = cell_name_and_factory
        cell = _make_cell(factory)
        s = {k: torch.randn_like(v) for k, v in cell.init_state(BATCH_SIZE).items()}
        # Ensure non-trivial values so the comparison is meaningful.
        for v in s.values():
            assert v.abs().sum() > 0
        mask = torch.tensor([True, False, True, False])
        s_masked = apply_episode_mask(s, mask)
        for k, v_orig in s.items():
            v_masked = s_masked[k]
            assert (v_masked[0] == 0).all(), f"{name}: state[{k!r}][0] not zeroed"
            assert (v_masked[2] == 0).all(), f"{name}: state[{k!r}][2] not zeroed"
            assert torch.equal(v_masked[1], v_orig[1]), f"{name}: state[{k!r}][1] changed"
            assert torch.equal(v_masked[3], v_orig[3]), f"{name}: state[{k!r}][3] changed"

    def test_detach_state_strips_requires_grad(self, cell_name_and_factory):
        name, factory = cell_name_and_factory
        cell = _make_cell(factory)
        s = cell.init_state(BATCH_SIZE)
        x = torch.randn(BATCH_SIZE, INPUT_SIZE, requires_grad=True)
        _, s_new, _ = cell.step(x, s)
        s_det = detach_state(s_new)
        for k, v in s_det.items():
            assert not v.requires_grad, f"{name}: state[{k!r}] still requires grad after detach"
            # Shape must be preserved.
            assert v.shape == s_new[k].shape


# --- gradient flow ---------------------------------------------------------


class TestGradientFlow:
    def test_brief_training_reduces_loss(self, cell_name_and_factory):
        """Sanity check: 50 Adam steps on a fixed-target regression reduces loss."""
        name, factory = cell_name_and_factory
        cell = _make_cell(factory)
        opt = torch.optim.Adam(cell.parameters(), lr=3e-3)

        # Fixed input + target — cell must memorize the mapping.
        T = 4
        x_seq = torch.randn(T, BATCH_SIZE, INPUT_SIZE)
        target = torch.randn(T, BATCH_SIZE, HIDDEN_SIZE)

        initial_loss = None
        final_loss = None
        for step in range(50):
            s = cell.init_state(BATCH_SIZE)
            y_seq, _, _ = cell.forward_sequence(x_seq, s)
            loss = (y_seq - target).pow(2).mean()
            if step == 0:
                initial_loss = loss.item()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cell.parameters(), max_norm=1.0)
            opt.step()
            final_loss = loss.item()

        assert final_loss < initial_loss * 0.9, \
            f"{name}: loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_all_parameters_receive_gradient(self, cell_name_and_factory):
        """Every learnable parameter must receive a non-None gradient."""
        name, factory = cell_name_and_factory
        cell = _make_cell(factory)

        T = 4
        x_seq = torch.randn(T, BATCH_SIZE, INPUT_SIZE)
        s = cell.init_state(BATCH_SIZE)
        y_seq, _, _ = cell.forward_sequence(x_seq, s)
        loss = y_seq.pow(2).mean()
        loss.backward()

        for n, p in cell.named_parameters():
            assert p.grad is not None, f"{name}: parameter {n!r} got no gradient"
            assert torch.isfinite(p.grad).all(), f"{name}: parameter {n!r} has NaN/Inf grad"
