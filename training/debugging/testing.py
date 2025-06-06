import torch
from collections import OrderedDict
from typing import Tuple, List, Union

def _tensors_close(a: torch.Tensor,
                   b: torch.Tensor,
                   rtol: float,
                   atol: float) -> bool:
    """
    Choose exact== for integer / bool tensors and allclose for floats.
    """
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return False                      # let caller handle non-tensor case
    if a.dtype.is_floating_point or a.dtype.is_complex:
        return torch.allclose(a, b, rtol=rtol, atol=atol)
    else:
        return torch.equal(a, b)


def _compare_state_dicts(d1: "OrderedDict[str, Union[torch.Tensor, object]]",
                         d2: "OrderedDict[str, Union[torch.Tensor, object]]",
                         rtol: float,
                         atol: float,
                         max_mismatches: int = 10) -> List[str]:
    """
    Return a list of human-readable mismatch messages (empty == identical).
    Stops after `max_mismatches` to avoid flooding.
    """
    mismatches: List[str] = []
    keys1, keys2 = set(d1.keys()), set(d2.keys())

    # Missing / extra keys
    for miss in keys1 ^ keys2:
        if len(mismatches) >= max_mismatches:
            return mismatches
        origin = "first" if miss in keys1 else "second"
        mismatches.append(f"Key '{miss}' present only in the {origin} dict.")

    # Compare common keys
    for k in keys1 & keys2:
        if len(mismatches) >= max_mismatches:
            break
        v1, v2 = d1[k], d2[k]

        # Tensor branch
        if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
            if v1.shape != v2.shape or v1.dtype != v2.dtype:
                mismatches.append(
                    f"Tensor '{k}' differs in shape/dtype: "
                    f"{tuple(v1.shape)},{v1.dtype} vs {tuple(v2.shape)},{v2.dtype}"
                )
            elif not _tensors_close(v1, v2, rtol, atol):
                mismatches.append(f"Tensor values differ at '{k}': {v1!r} vs {v2!r}.")
                # mismatches.append(f"  {v1}\n\n------------------------\n\n vs\n\n {v2}")
        # Non-tensor branch
        elif isinstance(v1, dict) and isinstance(v2, dict):
            # Recursion for nested dicts
            sub_mismatches = _compare_state_dicts(v1, v2, rtol, atol)
            if sub_mismatches:
                mismatches.append(f"Nested dict '{k}' mismatch:")
                for msg in sub_mismatches:
                    mismatches.append("  " + msg)
        elif isinstance(v1, list) and isinstance(v2, list):
            if len(v1) != len(v2):
                mismatches.append(f"List '{k}' differs in length: {len(v1)} vs {len(v2)}")
            else:
                for i, (v1i, v2i) in enumerate(zip(v1, v2)):
                    if isinstance(v1i, torch.Tensor) and isinstance(v2i, torch.Tensor):
                        if not _tensors_close(v1i, v2i, rtol, atol):
                            mismatches.append(f"List '{k}' differs at index {i}: {v1i} vs {v2i}")
                    elif isinstance(v1i, dict) and isinstance(v2i, dict):
                        sub_mismatches = _compare_state_dicts(v1i, v2i, rtol, atol)
                        if sub_mismatches:
                            mismatches.append(f"List '{k}' mismatch at index {i}:")
                            for msg in sub_mismatches:
                                mismatches.append("  " + msg)
                    elif v1i != v2i:
                        mismatches.append(f"List '{k}' differs at index {i}: {v1i} vs {v2i}")
        else:
            if v1 != v2:
                mismatches.append(f"Value mismatch at '{k}': {v1!r} vs {v2!r}")
    return mismatches


def compare_models_and_optimizers(model_a: torch.nn.Module,
                                  model_b: torch.nn.Module,
                                  optim_a: torch.optim.Optimizer,
                                  optim_b: torch.optim.Optimizer,
                                  *,
                                  rtol: float = 1e-5,
                                  atol: float = 1e-8,
                                  verbose: bool = True) -> bool:
    """
    Return `True` if both models and both optimizers have identical (all-close)
    states.  Otherwise print / return mismatches.

    Parameters
    ----------
    rtol, atol : float
        Tolerances passed to `torch.allclose` for floating-point tensors.
    verbose : bool
        If True, prints mismatch details to stdout.

    Notes
    -----
    * Uses `state_dict()` (fast, ordered, includes buffers).
    * Stops reporting after 10 discrepancies per object to stay readable.
    """
    m_mismatch = _compare_state_dicts(model_a.state_dict(), model_b.state_dict(), rtol, atol)
    if m_mismatch:
        print("Model mismatch(es):")
        for msg in m_mismatch: print("  •", msg)
    
    # print optimizer state dict keys
    o_mismatch = _compare_state_dicts(optim_a, optim_b, rtol, atol)
    
    if o_mismatch:
        print("Optimizer mismatch(es):")
        for msg in o_mismatch: 
            print("  •", msg)
        print(f'Step mismatch: {optim_a["state"][0]["step"]} vs {optim_b["state"][0]["step"]}')
        print(f"Learning rate mismatch: {optim_a['param_groups'][0]['lr']} vs {optim_b['param_groups'][0]['lr']}")
    if verbose:
        if not m_mismatch and not o_mismatch:
            print("✓ Models *and* optimizers match.")

    # Return True only when *nothing* mismatched
    return not m_mismatch and not o_mismatch