"""Kaggle submission agent — Agent subclass matching the inversion sample shape.

This is a structural port of the inversion StochasticGoose sample notebook
agent (LB 0.25). Differences from the sample:
  - Click prior: static-mask + bg-downweight + 4-conn segment equalization,
    multiplied into the sigmoid coord probabilities before sampling.
    (Sample has no prior; coord_probs are flat-sigmoid.)
  - All segment logic is inlined here so this file can be %%writefile'd into
    /kaggle/working/my_agent.py without needing src/perception imports.

To use:
  - In a Kaggle notebook, ship this file as /kaggle/working/my_agent.py,
    then copy into /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py
  - Register in agents/__init__.py: AVAILABLE_AGENTS = {"myagent": MyAgent}
  - Run via main.py --agent myagent
"""

import hashlib
import logging
import os
import random
import time
import traceback
from collections import deque
from datetime import datetime
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState


# ============================================================================
# Inlined: 4-conn segmentation + click prior (mirrors src/perception/segmenter.py)
# ============================================================================

GRID = 64
NUM_COLOURS = 16


def label_4conn_same_color(arr_2d: np.ndarray) -> tuple[np.ndarray, int]:
    H, W = arr_2d.shape
    labels = np.zeros((H, W), dtype=np.int32)
    next_label = 0
    stack: list[tuple[int, int]] = []
    for sy in range(H):
        for sx in range(W):
            if labels[sy, sx]:
                continue
            next_label += 1
            color = arr_2d[sy, sx]
            labels[sy, sx] = next_label
            stack.append((sy, sx))
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and not labels[ny, nx] and arr_2d[ny, nx] == color:
                        labels[ny, nx] = next_label
                        stack.append((ny, nx))
    return labels, next_label


class StaticPriorBuilder:
    """Per-game running stats + per-frame click prior. Inlined from segmenter.py."""

    def __init__(
        self,
        mask_after_frames: int = 20,
        background_downweight: float = 0.1,
        enable_segment_prior: bool = True,
        min_segment_size: int = 3,
    ) -> None:
        self.mask_after_frames = mask_after_frames
        self.background_downweight = background_downweight
        self.enable_segment_prior = enable_segment_prior
        self.min_segment_size = min_segment_size
        self.reset()

    def reset(self) -> None:
        self.change_count = np.zeros((GRID, GRID), dtype=np.int32)
        self.color_count = np.zeros(NUM_COLOURS, dtype=np.int64)
        self.last_frame: Optional[np.ndarray] = None
        self.n_frames = 0

    def observe(self, frame_2d: np.ndarray) -> None:
        if frame_2d.shape != (GRID, GRID):
            return
        if self.last_frame is not None:
            self.change_count += (frame_2d != self.last_frame).astype(np.int32)
        self.last_frame = frame_2d.copy()
        self.n_frames += 1
        self.color_count += np.bincount(frame_2d.ravel(), minlength=NUM_COLOURS).astype(np.int64)

    def click_prior(self, frame_2d: np.ndarray) -> np.ndarray:
        prior = np.ones((GRID, GRID), dtype=np.float32)
        if self.n_frames >= self.mask_after_frames:
            static = self.change_count == 0
            prior[static] = 0.0
        if self.color_count.sum() > 0 and self.background_downweight < 1.0:
            bg_color = int(self.color_count.argmax())
            prior[frame_2d == bg_color] *= self.background_downweight
        if self.enable_segment_prior:
            labels, n = label_4conn_same_color(frame_2d)
            if n > 0:
                sizes = np.bincount(labels.ravel(), minlength=n + 1)
                per_label_w = np.zeros(n + 1, dtype=np.float32)
                valid = sizes >= self.min_segment_size
                per_label_w[valid] = 1.0 / sizes[valid]
                prior = prior * per_label_w[labels]
        return prior


# ============================================================================
# ActionModel CNN — same as inversion sample
# ============================================================================

class ActionModel(nn.Module):
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
        conv_features = F.relu(self.conv4(x))

        action_features = self.action_pool(conv_features)
        action_features = action_features.view(action_features.size(0), -1)
        action_features = F.relu(self.action_fc(action_features))
        action_features = self.dropout(action_features)
        action_logits = self.action_head(action_features)

        coord_features = F.relu(self.coord_conv1(conv_features))
        coord_features = F.relu(self.coord_conv2(coord_features))
        coord_features = F.relu(self.coord_conv3(coord_features))
        coord_logits = self.coord_conv4(coord_features)
        coord_logits = coord_logits.view(coord_logits.size(0), -1)

        return torch.cat([action_logits, coord_logits], dim=1)


# ============================================================================
# MyAgent — Agent subclass, gateway-compatible (matches sample's contract)
# ============================================================================

class MyAgent(Agent):
    """CNN bandit + segment-prior coord sampler. Resets model per level."""

    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed % (2**32 - 1))
        self.start_time = time.time()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[{self.game_id}] device: {self.device}")

        self.current_score = -1
        self.logger = logging.getLogger(f"MyAgent_{self.game_id}")

        self.grid_size = GRID
        self.num_coordinates = self.grid_size * self.grid_size
        self.num_colours = NUM_COLOURS
        self.action_model: Optional[ActionModel] = None
        self.optimizer: Optional[optim.Optimizer] = None

        self.experience_buffer: deque = deque(maxlen=200_000)
        self.experience_hashes: set = set()
        self.batch_size = 64
        self.train_frequency = 5

        self.prev_frame: Optional[np.ndarray] = None
        self.prev_action_idx: Optional[int] = None

        self.action_list = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                            GameAction.ACTION4, GameAction.ACTION5]

        # Our prior — disabled until we've seen mask_after_frames observations
        self.prior = StaticPriorBuilder(enable_segment_prior=True)

    def append_frame(self, frame: FrameData) -> None:
        self.frames.append(frame)
        if len(self.frames) > self._MAX_FRAMES:
            self.frames = self.frames[-self._MAX_FRAMES:]
        if frame.guid:
            self.guid = frame.guid
        if hasattr(self, "recorder") and not self.is_playback:
            import json
            self.recorder.record(__import__('json').loads(frame.model_dump_json()))

    def _get_level(self, frame: FrameData) -> int:
        return getattr(frame, 'score', None) or frame.levels_completed

    def _has_time_elapsed(self) -> bool:
        return (time.time() - self.start_time) >= 8 * 3600 - 5 * 60

    def is_done(self, frames, latest_frame) -> bool:
        try:
            return any([
                latest_frame.state is GameState.WIN,
                self._has_time_elapsed(),
            ])
        except Exception as e:
            print(f"[{self.game_id}] is_done crashed: {e}")
            traceback.print_exc()
            return True

    def _frame_to_tensor(self, frame_data: FrameData) -> torch.Tensor:
        frame = np.array(frame_data.frame, dtype=np.int64)
        frame = frame[-1]  # last layer
        if frame.shape != (self.grid_size, self.grid_size):
            raise ValueError(f"Bad frame shape {frame.shape}")
        tensor = torch.zeros(self.num_colours, self.grid_size, self.grid_size, dtype=torch.float32)
        tensor.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        return tensor.to(self.device)

    def _exp_hash(self, frame_bool: np.ndarray, action_idx: int) -> str:
        return hashlib.md5(frame_bool.tobytes() + str(action_idx).encode()).hexdigest()

    def _train_step(self) -> None:
        if len(self.experience_buffer) < self.batch_size:
            return
        idxs = np.random.choice(len(self.experience_buffer), self.batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in idxs]
        states = torch.stack([torch.from_numpy(e['state']).float().to(self.device) for e in batch])
        action_idxs = torch.tensor([e['action_idx'] for e in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([e['reward'] for e in batch], dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        logits = self.action_model(states)
        selected = logits.gather(1, action_idxs.unsqueeze(1)).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(selected, rewards)

        # entropy regularizer (matches sample)
        probs = torch.sigmoid(logits)
        loss = loss - 0.0001 * probs[:, :5].mean() - 0.00001 * probs[:, 5:].mean()
        loss.backward()
        self.optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _sample_action(self, logits: torch.Tensor, available_actions, frame_2d: np.ndarray):
        """Sample from 5 + 4096 logits. Apply segment prior to coord_probs."""
        action_logits = logits[:5]
        coord_logits = logits[5:]

        # Mask unavailable actions (gateway sends raw ints [1..6])
        if available_actions:
            mask = torch.full_like(action_logits, float('-inf'))
            action6_avail = False
            for a in available_actions:
                aid = a.value if hasattr(a, 'value') else int(a)
                if 1 <= aid <= 5:
                    mask[aid - 1] = 0.0
                elif aid == 6:
                    action6_avail = True
            action_logits = action_logits + mask
            if not action6_avail:
                coord_logits = coord_logits + torch.full_like(coord_logits, float('-inf'))

        action_probs = torch.sigmoid(action_logits)
        coord_probs = torch.sigmoid(coord_logits)

        # === our addition: multiply prior into coord_probs ===
        prior = self.prior.click_prior(frame_2d)        # (64, 64) float32
        prior_t = torch.from_numpy(prior.reshape(-1)).to(coord_probs.device).to(coord_probs.dtype)
        coord_probs = coord_probs * prior_t

        # match sample's scaling: coord-side probabilities split into per-pixel mass
        coord_probs_scaled = coord_probs / self.num_coordinates

        all_p = torch.cat([action_probs, coord_probs_scaled])
        total = all_p.sum()
        if total.item() == 0:
            # nothing has weight (prior killed everything + masks too) — fall back to action_probs only
            all_p = torch.cat([action_probs, torch.zeros_like(coord_probs_scaled)])
            total = all_p.sum()
            if total.item() == 0:
                # last resort: pick any available simple action uniformly
                idx = int(np.argmax((action_logits > float('-inf')).cpu().numpy()))
                return idx, None, None
        all_p = (all_p / total).cpu().numpy()
        sel = int(np.random.choice(len(all_p), p=all_p))
        if sel < 5:
            return sel, None, None
        coord_idx = sel - 5
        return 5, (coord_idx // self.grid_size, coord_idx % self.grid_size), coord_idx

    def choose_action(self, frames, latest_frame) -> GameAction:
        try:
            # Level transition → reset model + buffer
            current = self._get_level(latest_frame)
            if current != self.current_score:
                self.experience_buffer.clear()
                self.experience_hashes.clear()
                self.action_model = ActionModel(input_channels=self.num_colours,
                                                grid_size=self.grid_size).to(self.device)
                self.optimizer = optim.Adam(self.action_model.parameters(), lr=0.0001)
                self.prev_frame = None
                self.prev_action_idx = None
                self.current_score = current
                self.prior.reset()
                print(f"[{self.game_id}] level changed → reset; lvl={current}")

            if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                self.prev_frame = None
                self.prev_action_idx = None
                action = GameAction.RESET
                action.reasoning = "game needs reset"
                return action

            frame_2d = np.array(latest_frame.frame, dtype=np.int64)[-1]
            self.prior.observe(frame_2d)

            cur_tensor = self._frame_to_tensor(latest_frame)

            # Record transition for prev action
            if self.prev_frame is not None and self.prev_action_idx is not None:
                exp_h = self._exp_hash(self.prev_frame, self.prev_action_idx)
                if exp_h not in self.experience_hashes:
                    cur_bool = cur_tensor.cpu().numpy().astype(bool)
                    changed = not np.array_equal(self.prev_frame, cur_bool)
                    self.experience_buffer.append({
                        'state': self.prev_frame,
                        'action_idx': self.prev_action_idx,
                        'reward': 1.0 if changed else 0.0,
                    })
                    self.experience_hashes.add(exp_h)

            available = getattr(latest_frame, 'available_actions', None)
            with torch.no_grad():
                logits = self.action_model(cur_tensor.unsqueeze(0)).squeeze(0)
                action_idx, coords, coord_idx = self._sample_action(logits, available, frame_2d)

            if action_idx < 5:
                action = self.action_list[action_idx]
                action.reasoning = f"{action.name}"
            else:
                action = GameAction.ACTION6
                y, x = coords
                action.set_data({"x": int(x), "y": int(y)})
                action.reasoning = f"ACTION6 at ({x},{y})"

            self.prev_frame = cur_tensor.cpu().numpy().astype(bool)
            self.prev_action_idx = action_idx if action_idx < 5 else (5 + coord_idx)

            if self.action_counter % self.train_frequency == 0:
                self._train_step()

            return action

        except Exception as e:
            print(f"[{self.game_id}] choose_action crashed at step {self.action_counter}: {type(e).__name__}: {e}")
            traceback.print_exc()
            action = random.choice(self.action_list[:5])
            action.reasoning = f"fallback after error: {e}"
            return action
