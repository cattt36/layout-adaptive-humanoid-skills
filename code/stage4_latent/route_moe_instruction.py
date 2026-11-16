#!/usr/bin/env python3
"""Route a text/ASR command to one of the three MOE Student action experts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
try:
    import torch
except ModuleNotFoundError as exc:
    raise RuntimeError("This script requires PyTorch") from exc
from train_joint_semantic_router import Router, DEFAULT_ACTIONS, explicit_action_cue
from motion_latent_z import (
    CHECKPOINT_FORMAT_VERSION,
    _normalize_command_text,
    hashed_text_vector,
)

LATENT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = LATENT_ROOT / 'moe_router_v3' / 'joint_semantic_router.pt'
# Only the v4 tree satisfies the deployment-observable latent contract.
DEFAULT_EXPERT_ROOT = LATENT_ROOT / 'latent_15_v4'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument('--query', required=True)
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--profile', choices=('Z0','Z1','Z2','Z3','Z4'), default='Z4',
                   help='Latent expert ablation to deploy; Z4 is the recommended full expert.')
    p.add_argument('--min-confidence', type=float, default=0.60,
                   help='Reject learned-only routes below this top probability.')
    p.add_argument('--min-margin', type=float, default=0.15,
                   help='Reject learned-only routes when top-1 minus top-2 is smaller.')
    p.add_argument('--allow-learned-fallback', action='store_true',
                   help='Allow cue-free learned routing after confidence checks; disabled for safety by default.')
    p.add_argument('--allow-ambiguous', action='store_true',
                   help='Offline-only override that disables confidence/margin rejection.')
    p.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    p.add_argument('--expert-root', type=Path, default=None, help='Expert root directory; defaults to the sibling latent_15_v4 directory.')
    args=p.parse_args()
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError('--min-confidence must be in [0, 1]')
    if not 0.0 <= args.min_margin <= 1.0:
        raise ValueError('--min-margin must be in [0, 1]')
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Router checkpoint not found: {checkpoint_path}')
    ckpt=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    actions=tuple(ckpt.get('actions',DEFAULT_ACTIONS))
    device_name = (
        'cuda' if args.device == 'auto' and torch.cuda.is_available()
        else 'cpu' if args.device == 'auto' else args.device
    )
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('--device cuda was requested, but CUDA is unavailable')
    device=torch.device(device_name)
    model=Router(int(ckpt['text_dim']),int(ckpt['hidden_dim']),int(ckpt['latent_dim']),len(actions)).to(device)
    model.load_state_dict(ckpt['model_state']); model.eval()
    norm = _normalize_command_text(args.query)
    x=hashed_text_vector(norm,int(ckpt["text_dim"]))
    with torch.no_grad():
        z, logits=model(torch.from_numpy(x[None,:]).to(device))
        probs=torch.softmax(logits,dim=-1).cpu().numpy()[0]
        z=z.cpu().numpy()[0]
    learned_probs = probs.copy()
    learned_action = actions[int(np.argmax(learned_probs))]
    # Explicit command ontology is applied after the learned router.  This is
    # important for ASR/user wording: the hashed n-gram model is useful as a
    # fallback, but it can confuse the closely related three-pole and
    # five-pole tasks when it has seen few English paraphrases.
    # Keep training-time evaluation and deployment routing on exactly the
    # same bilingual ontology implementation.
    cue = explicit_action_cue(args.query)
    if cue in actions:
        idx = actions.index(cue)
        probs = probs * 0.10
        probs[idx] = 0.90 + 0.10 * probs[idx]

    order=np.argsort(probs)[::-1][:max(1,min(args.top_k,len(actions)))]
    top_probability = float(probs[int(order[0])])
    second_probability = float(probs[int(order[1])]) if len(order) > 1 else 0.0
    margin = top_probability - second_probability
    explicit_route = cue in actions
    accepted = bool(
        explicit_route
        or args.allow_ambiguous
        or (args.allow_learned_fallback and
            top_probability >= args.min_confidence
            and margin >= args.min_margin
        )
    )
    selected_action = actions[int(order[0])] if accepted else None
    expert_root = (Path(args.expert_root) if args.expert_root else DEFAULT_EXPERT_ROOT).expanduser().resolve()
    expert_dir = expert_root / selected_action / args.profile if selected_action else None
    expert_checkpoint = expert_dir / 'best_motion_latent.pt' if expert_dir else None
    expert_protocol = None
    if expert_checkpoint and expert_checkpoint.is_file():
        expert_protocol = torch.load(
            expert_checkpoint, map_location='cpu', weights_only=False
        ).get('checkpoint_format_version')
    expert_exists = bool(
        expert_checkpoint
        and expert_checkpoint.is_file()
        and expert_protocol == CHECKPOINT_FORMAT_VERSION
    )
    if accepted and not expert_exists:
        accepted = False
    result={
        'format':'moe_instruction_route_v3',
        'query':args.query, 'normalized_query':norm,
        'selected_action':selected_action,
        'accepted': accepted,
        'rejection_reason': (
            None if accepted
            else 'expert_checkpoint_missing_or_incompatible' if selected_action and not expert_exists
            else 'no_explicit_action_cue_or_low_confidence'
        ),
        'top_probability': top_probability,
        'top_margin': margin,
        'min_confidence': args.min_confidence,
        'min_margin': args.min_margin,
        'learned_fallback_enabled': args.allow_learned_fallback,
        'learned_prediction':learned_action,
        'explicit_cue':cue,
        'selected_expert_dir':str(expert_dir) if expert_dir else None,
        'selected_expert_checkpoint':str(expert_checkpoint) if expert_checkpoint else None,
        'required_expert_protocol': CHECKPOINT_FORMAT_VERSION,
        'found_expert_protocol': expert_protocol,
        'profile': args.profile,
        'expert_exists':expert_exists,
        'router_checkpoint':str(checkpoint_path),
        'routing_basis': 'explicit_command_ontology' if cue in actions else 'learned_router_fallback',
        'probabilities_before_ontology': {a:float(learned_probs[i]) for i,a in enumerate(actions)},
        'probabilities_after_ontology': {a:float(probs[i]) for i,a in enumerate(actions)},
        'results':[{'rank':i+1,'action':actions[int(j)],'probability':float(probs[int(j)])} for i,j in enumerate(order)]
    }
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
