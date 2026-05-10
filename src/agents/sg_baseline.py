"""StochasticGoose-style action-learning agent ported to modern arc_agi SDK.

Verbatim port of ref/ARC3-solution/custom_agents/action.py with adaptations:
  - Uses modern arcengine.FrameData (frame: list[list[list[int]]] of palette indices)
    via arc_agi.Arcade — old SG used a legacy HTTP client.
  - Tracks levels via FrameData.levels_completed instead of legacy `score` int.
  - available_actions are list[int] (action ids), not list[GameAction].
  - No tensorboard / no visualization side-channels — that's the trainer's job.
  - CPU-friendly: no torch.cuda calls assumed.
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from arcengine import FrameData, GameAction, GameState

from src.perception.segmenter import StaticPriorBuilder

LOG = logging.getLogger(__name__)


class ActionModel(nn.Module):
    """SG's CNN: predicts per-action P(frame will change next).

    Input:  (B, 16, 64, 64) one-hot palette index frames.
    Output: (B, 5 + 4096) — logits for ACTION1-5 + per-pixel ACTION6 click logits.
    """

    def __init__(self, input_channels: int = 16, grid_size: int = 64) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.num_action_types = 5

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.action_pool = nn.MaxPool2d(4, 4)
        self.action_fc = nn.Linear(256 * 16 * 16, 512)
        self.action_head = nn.Linear(512, self.num_action_types)

        self.coord_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, kernel_size=1)
        self.coord_conv4 = nn.Conv2d(32, 1, kernel_size=1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        feat = F.relu(self.conv4(x))                              # (B, 256, 64, 64)

        a = self.action_pool(feat)                                # (B, 256, 16, 16)
        a = a.view(a.size(0), -1)                                 # (B, 65536)
        a = F.relu(self.action_fc(a))
        a = self.dropout(a)
        action_logits = self.action_head(a)                       # (B, 5)

        c = F.relu(self.coord_conv1(feat))
        c = F.relu(self.coord_conv2(c))
        c = F.relu(self.coord_conv3(c))
        coord_logits = self.coord_conv4(c).view(c.size(0), -1)    # (B, 4096)

        return torch.cat([action_logits, coord_logits], dim=1)    # (B, 5+4096)


@dataclass
class TrainStats:
    main_loss: float = 0.0
    total_loss: float = 0.0
    action_entropy: float = 0.0
    coord_entropy: float = 0.0
    accuracy: float = 0.0


class SGBaselineAgent:
    """SG-style agent: stateless w.r.t. the env runtime; you drive it.

    Lifecycle (caller's responsibility, see scripts/train_sg.py):
      1. agent = SGBaselineAgent(grid_size=64, device='cpu')
      2. frame = env.reset()
      3. loop:
          action, data = agent.choose_action(frame)
          new_frame = env.step(action, data=data)
          agent.observe(prev_frame_tensor, action_idx, new_frame)  # adds experience
          if step % train_freq == 0: agent.train_step()
          frame = new_frame
    """

    ACTION_LIST = [
        GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
        GameAction.ACTION4, GameAction.ACTION5,
    ]
    NUM_COLOURS = 16
    GRID = 64
    NUM_COORDS = 64 * 64

    def __init__(
        self,
        device: str = "cpu",
        lr: float = 1e-4,
        batch_size: int = 64,
        buffer_size: int = 200_000,
        train_frequency: int = 5,
        action_entropy_coef: float = 1e-4,
        coord_entropy_coef: float = 1e-5,
        use_coord_prior: bool = True,
        prior_mask_after_frames: int = 20,
        prior_background_downweight: float = 0.1,
        enable_segment_prior: bool = False,
        prior_min_segment_size: int = 3,
        seed: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.train_frequency = train_frequency
        self.action_entropy_coef = action_entropy_coef
        self.coord_entropy_coef = coord_entropy_coef
        self.use_coord_prior = use_coord_prior

        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.model: Optional[ActionModel] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.experience_buffer: deque = deque(maxlen=buffer_size)
        self.experience_hashes: set = set()

        self.segmenter = StaticPriorBuilder(
            mask_after_frames=prior_mask_after_frames,
            background_downweight=prior_background_downweight,
            enable_segment_prior=enable_segment_prior,
            min_segment_size=prior_min_segment_size,
        ) if use_coord_prior else None

        self.current_levels_completed = -1
        self.action_counter = 0
        self._init_model()

    def _init_model(self) -> None:
        self.model = ActionModel(self.NUM_COLOURS, self.GRID).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    # ---- public API ----

    def maybe_reset_for_new_level(self, frame: FrameData, *, reset_model: bool = True) -> bool:
        """Detect level-up and (optionally) wipe model+buffer (SG's default).

        First-frame seed: when current_levels_completed == -1 (initial state) we
        only adopt the value silently — no model reset, no buffer wipe — so the
        flag actually controls behavior on *real* level transitions only."""
        if frame.levels_completed == self.current_levels_completed:
            return False
        if self.current_levels_completed == -1:
            self.current_levels_completed = frame.levels_completed
            return False
        LOG.info(
            "level transition %d -> %d at action %d",
            self.current_levels_completed, frame.levels_completed, self.action_counter,
        )
        self.experience_buffer.clear()
        self.experience_hashes.clear()
        if reset_model:
            self._init_model()
        if self.segmenter is not None:
            self.segmenter.reset()
        self.current_levels_completed = frame.levels_completed
        return True

    def _frame_2d(self, frame: FrameData) -> np.ndarray:
        """Last subframe as a (64, 64) int64 palette grid."""
        arr = np.array(frame.frame, dtype=np.int64)
        if arr.ndim == 3:
            arr = arr[-1]
        assert arr.shape == (self.GRID, self.GRID), f"got {arr.shape}"
        return arr

    def frame_to_tensor(self, frame: FrameData) -> torch.Tensor:
        """Convert FrameData.frame (list[N_subframes][64][64] palette idx) -> (16, 64, 64) float.

        Pure: does NOT update segmenter stats. Use choose_action for the canonical
        per-step observation path."""
        arr = self._frame_2d(frame)
        t = torch.zeros(self.NUM_COLOURS, self.GRID, self.GRID, dtype=torch.float32)
        t.scatter_(0, torch.from_numpy(arr).unsqueeze(0), 1)
        return t.to(self.device)

    def choose_action(self, frame: FrameData) -> tuple[GameAction, Optional[dict], int]:
        """Return (action, data_dict_or_None, unified_action_idx).

        unified_action_idx = 0..4 for ACTION1-5; 5 + (y*64 + x) for ACTION6."""
        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET, None, -1

        arr_2d = self._frame_2d(frame)
        if self.segmenter is not None:
            self.segmenter.observe(arr_2d)
            prior = self.segmenter.click_prior(arr_2d)
        else:
            prior = None

        t = torch.zeros(self.NUM_COLOURS, self.GRID, self.GRID, dtype=torch.float32)
        t.scatter_(0, torch.from_numpy(arr_2d).unsqueeze(0), 1)
        x_in = t.to(self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x_in).squeeze(0)
        action_idx, coords, coord_idx = self._sample(
            logits, frame.available_actions or [], coord_prior=prior,
        )

        if action_idx < 5:
            return self.ACTION_LIST[action_idx], None, action_idx
        y, x = coords
        return GameAction.ACTION6, {"x": int(x), "y": int(y)}, 5 + int(coord_idx)

    def observe(self, prev_state_t: torch.Tensor, prev_action_idx: int, new_frame: FrameData) -> None:
        """Append (prev_state, prev_action, frame_changed) to the experience buffer."""
        if prev_action_idx < 0:        # came from a RESET; nothing to learn
            return
        new_state_t = self.frame_to_tensor(new_frame)
        prev_np = prev_state_t.cpu().numpy().astype(bool)
        new_np = new_state_t.cpu().numpy().astype(bool)
        h = self._exp_hash(prev_np, prev_action_idx)
        if h in self.experience_hashes:
            return
        self.experience_hashes.add(h)
        changed = not np.array_equal(prev_np, new_np)
        self.experience_buffer.append({
            "state": prev_np,
            "action_idx": prev_action_idx,
            "reward": 1.0 if changed else 0.0,
        })

    def train_step(self) -> Optional[TrainStats]:
        if len(self.experience_buffer) < self.batch_size:
            return None
        idx = self.rng.choice(len(self.experience_buffer), self.batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in idx]

        states = torch.stack([torch.from_numpy(e["state"]).float() for e in batch]).to(self.device)
        action_idx = torch.tensor([e["action_idx"] for e in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([e["reward"] for e in batch], dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        logits = self.model(states)                                              # (B, 4101)
        selected = logits.gather(1, action_idx.unsqueeze(1)).squeeze(1)
        main = F.binary_cross_entropy_with_logits(selected, rewards)

        probs = torch.sigmoid(logits)
        a_ent = probs[:, :5].mean()
        c_ent = probs[:, 5:].mean()
        loss = main - self.action_entropy_coef * a_ent - self.coord_entropy_coef * c_ent
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            acc = ((torch.sigmoid(selected) > 0.5) == (rewards > 0.5)).float().mean()
        return TrainStats(
            main_loss=main.item(),
            total_loss=loss.item(),
            action_entropy=a_ent.item(),
            coord_entropy=c_ent.item(),
            accuracy=acc.item(),
        )

    # ---- internals ----

    def _sample(
        self,
        logits: torch.Tensor,
        available_actions: list[int],
        coord_prior: Optional[np.ndarray] = None,
    ) -> tuple[int, Optional[tuple[int, int]], Optional[int]]:
        action_logits = logits[:5].clone()
        coord_logits = logits[5:].clone()

        # available_actions is list[int] of action ids 1..7 (0 = RESET, never in this list mid-game)
        valid_simple = {a for a in available_actions if 1 <= a <= 5}
        action6_ok = 6 in available_actions
        if valid_simple or action6_ok:
            mask = torch.full_like(action_logits, float("-inf"))
            for a in valid_simple:
                mask[a - 1] = 0.0
            action_logits = action_logits + mask
            if not action6_ok:
                coord_logits = coord_logits + torch.full_like(coord_logits, float("-inf"))

        action_p = torch.sigmoid(action_logits)
        coord_p = torch.sigmoid(coord_logits) / self.NUM_COORDS

        if coord_prior is not None and action6_ok:
            prior_t = torch.from_numpy(coord_prior.reshape(-1)).to(coord_p.device).to(coord_p.dtype)
            coord_p = coord_p * prior_t

        all_p = torch.cat([action_p, coord_p])
        s = all_p.sum()
        if not torch.isfinite(s) or s.item() <= 0:
            # fallback: uniform over valid simple actions; ACTION1 if nothing
            choices = sorted(valid_simple) or [1]
            return int(choices[0] - 1), None, None
        all_p = (all_p / s).cpu().numpy()
        sel = int(self.rng.choice(len(all_p), p=all_p))
        if sel < 5:
            return sel, None, None
        coord_idx = sel - 5
        return 5, (coord_idx // self.GRID, coord_idx % self.GRID), coord_idx

    def _exp_hash(self, frame_bool: np.ndarray, action_idx: int) -> str:
        return hashlib.md5(frame_bool.tobytes() + str(action_idx).encode()).hexdigest()
