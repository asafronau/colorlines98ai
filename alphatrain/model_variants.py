"""Rebuild any policy net from its state_dict: the standard ResNet (alphatrain/model.py), the D4-equivariant p4m
trunk (model_p4m.py, HISTORY 278) or the color-equivariant slot trunk (model_c7.py, HISTORY 279). One place for
the loaders (evaluate.load_model, inference_cpp/export_ts.load_model, scripts/model_shape)."""
from alphatrain.model import PolicyNet, head_kwargs_from_state
from alphatrain.model_c7 import PolicyNetC7, c7_kwargs_from_state, is_c7_state
from alphatrain.model_p4m import PolicyNetP4M, is_p4m_state, p4m_kwargs_from_state


def strip_state(state):
    """Drop torch.compile prefixes and dead value-head keys of old dual-head checkpoints."""
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    return {k: v for k, v in state.items() if not k.startswith('value_')}


def net_from_state(state):
    """Return (net in eval mode with `state` loaded, description, train_path_b shape flags). p4m and c7 nets come back
    frozen (plain convolutions with the assembled weights): inference only."""
    state = strip_state(state)
    if is_p4m_state(state):
        kw = p4m_kwargs_from_state(state)
        net = PolicyNetP4M(**kw)
        net.load_state_dict(state)
        net.train(False)
        net.freeze()
        return (net, f"p4m {kw['num_blocks']}b x {kw['group_channels']}g/{kw['expand']}e",
                f"--trunk p4m --num-blocks {kw['num_blocks']} --group-channels {kw['group_channels']} "
                f"--p4m-expand {kw['expand']}")
    if is_c7_state(state):
        kw = c7_kwargs_from_state(state)
        net = PolicyNetC7(**kw)
        net.load_state_dict(state)
        net.train(False)
        net.freeze()
        return (net, f"c7 {kw['num_blocks']}b x {kw['slot_channels']}k/{kw['shared_channels']}s",
                f"--trunk c7 --num-blocks {kw['num_blocks']} --slot-channels {kw['slot_channels']} "
                f"--shared-channels {kw['shared_channels']}")
    ch = state['stem.0.weight'].shape[0]
    nb = sum(1 for k in state if k.endswith('.conv1.weight') and k.startswith('blocks.'))
    net = PolicyNet(in_channels=state['stem.0.weight'].shape[1], num_blocks=nb, channels=ch,
                    **head_kwargs_from_state(state))
    net.load_state_dict(state)
    net.train(False)
    return net, f'{nb}b x {ch}ch', f'--num-blocks {nb} --channels {ch}'
