from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from vidur.config import OnlineLpGlobalSchedulerConfig
from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.scheduler.utils.memory_planner import MemoryPlanner


@dataclass
class LPRequest:
    request: Request
    j: int
    t_arr: int
    s: int
    o: int


def reward_components(
    alpha: float,
    beta: float,
    gamma: float,
    sigma: float,
    o: int,
    T: int,
    tau: int,
    t_arr: int,
    goodput_window: int,
    goodput_weight: float,
) -> Tuple[float, float, float, float, float, float]:
    thr = max(0, min(o, T - tau + 1))
    comp_saved = max(0, T - (tau + o - 1))
    ft_saved = max(0, T - tau + 1)
    job_const = (beta + gamma) * (T - t_arr + 1)
    goodput_bonus = goodput_weight if (tau - t_arr) < goodput_window else 0.0
    r_report = sigma + alpha * thr + beta * comp_saved + gamma * ft_saved + goodput_bonus
    r_decision = r_report
    return thr, comp_saved, ft_saved, job_const, r_decision, r_report


class OnlineLPSimulator:
    def __init__(self, T: int, B: np.ndarray, M: np.ndarray) -> None:
        self.T = T
        self.G = len(B)
        self.B = B.astype(float)
        self.M = M.astype(float)
        self.thru_occ = np.zeros((self.G, T), dtype=float)
        self.mem_occ = np.zeros((self.G, T), dtype=float)

    def feasible_if_start(self, req: LPRequest, g: int, tau: int) -> bool:
        if tau < req.t_arr or tau > self.T:
            return False
        k0 = tau - 1
        k_end = min(self.T, tau + req.o - 1) - 1
        if k_end < k0:
            return False
        ks = np.arange(k0, k_end + 1, dtype=int)
        thetas = ks - k0
        if np.any(self.thru_occ[g, ks] + 1 > self.B[g]):
            return False
        mem_add = req.s + thetas + 1.0
        if np.any(self.mem_occ[g, ks] + mem_add > self.M[g]):
            return False
        return True

    def feasible_mask_all_g(
        self, req: LPRequest, tau: int
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if tau < req.t_arr or tau > self.T:
            return False, None, None, None
        k0 = tau - 1
        k_end = min(self.T, tau + req.o - 1) - 1
        if k_end < k0:
            return False, None, None, None
        idxs = np.arange(k0, k_end + 1, dtype=int)
        thetas = idxs - k0
        mem_vals = req.s + thetas + 1.0
        thru_ok = (self.thru_occ[:, idxs] + 1.0 <= self.B[:, None]).all(axis=1)
        mem_ok = (self.mem_occ[:, idxs] + mem_vals[None, :] <= self.M[:, None]).all(axis=1)
        feas_mask = thru_ok & mem_ok
        return True, idxs, mem_vals, feas_mask

    def start_job(
        self,
        req: LPRequest,
        g: int,
        tau: int,
        idxs: Optional[np.ndarray] = None,
        mem_vals: Optional[np.ndarray] = None,
    ) -> None:
        if idxs is None or mem_vals is None:
            k0 = tau - 1
            k_end = min(self.T, tau + req.o - 1) - 1
            idxs = np.arange(k0, k_end + 1, dtype=int)
            thetas = idxs - k0
            mem_vals = req.s + thetas + 1.0
        self.thru_occ[g, idxs] += 1.0
        self.mem_occ[g, idxs] += mem_vals

    def reset_state(self) -> None:
        self.thru_occ.fill(0.0)
        self.mem_occ.fill(0.0)


@dataclass
class Column:
    j: int
    g: int
    tau: int
    r_norm: float
    thru_idx: np.ndarray
    mem_vals: np.ndarray


class DualAHDPolicy:
    def __init__(
        self,
        sim: OnlineLPSimulator,
        grad_steps: int,
        step_size: float,
        batch_size: int,
        mem_scale_factor: float,
        backlog_gain: float,
        backlog_slack: float,
        length_penalty: float,
        reward_normalizer: float,
    ) -> None:
        self.sim = sim
        self.T = sim.T
        self.G = sim.G
        self.grad_steps = grad_steps
        self.step_size = step_size
        self.batch_size = max(1, int(batch_size))
        self.p = np.zeros((2, self.G, self.T), dtype=float)
        self.history: List[Column] = []
        self._sgd_calls = 0
        self.rng = np.random.default_rng(0)
        base_mem_scale = 1.0 / max(1.0, float(self.sim.M.max()))
        self.mem_scale = base_mem_scale * float(mem_scale_factor)
        self.reward_normalizer = max(1.0, float(reward_normalizer))
        self.backlog_gain = max(0.0, float(backlog_gain))
        self.backlog_slack = max(0.0, float(backlog_slack))
        self.length_penalty = max(0.0, float(length_penalty))

    def reset(self) -> None:
        self.p.fill(0.0)
        self.history.clear()
        self._sgd_calls = 0

    def _column_inner_vec(self, p_vec: np.ndarray, col: Column) -> float:
        sizeGT = self.G * self.T
        flat_thru = col.g * self.T + col.thru_idx
        flat_mem = sizeGT + (col.g * self.T + col.thru_idx)
        thru_part = p_vec[flat_thru].sum()
        mem_part = (p_vec[flat_mem] * (self.mem_scale * col.mem_vals)).sum()
        return float(thru_part + mem_part)

    def column_inner(self, g: int, idxs: np.ndarray, mem_vals: np.ndarray) -> float:
        thru_part = self.p[0, g, idxs].sum()
        mem_part = (self.p[1, g, idxs] * (self.mem_scale * mem_vals)).sum()
        return float(thru_part + mem_part)

    def record_sample(
        self,
        req: LPRequest,
        g: int,
        tau: int,
        idxs: np.ndarray,
        mem_vals: np.ndarray,
        r_norm: float,
    ) -> None:
        self.history.append(
            Column(
                j=req.j,
                g=g,
                tau=tau,
                r_norm=r_norm,
                thru_idx=idxs.copy(),
                mem_vals=mem_vals.astype(float, copy=True),
            )
        )

    def resolve_prices(self, d_vec_flat: np.ndarray) -> None:
        if not self.history:
            self.p.fill(0.0)
            return

        sizeGT = self.G * self.T
        p_vec = np.concatenate([self.p[0].reshape(-1), self.p[1].reshape(-1)])

        total_samples = len(self.history)
        steps = max(1, self.grad_steps)
        batch_size = min(self.batch_size, total_samples)
        scale = 1.0 / float(batch_size)

        for _ in range(steps):
            if batch_size == total_samples:
                sample_idx = np.arange(total_samples, dtype=int)
            else:
                sample_idx = self.rng.choice(total_samples, size=batch_size, replace=False)

            grad = d_vec_flat.copy()
            for idx in sample_idx:
                col = self.history[idx]
                margin = col.r_norm - self._column_inner_vec(p_vec, col)
                if margin <= 0.0:
                    continue
                flat_thru = col.g * self.T + col.thru_idx
                flat_mem = sizeGT + (col.g * self.T + col.thru_idx)
                grad[flat_thru] -= scale
                grad[flat_mem] -= scale * (self.mem_scale * col.mem_vals)
            p_vec = np.maximum(0.0, p_vec - self.step_size * grad)

        self.p[0] = p_vec[:sizeGT].reshape(self.G, self.T)
        self.p[1] = p_vec[sizeGT:].reshape(self.G, self.T)
        self._sgd_calls += 1


class OnlineLPGlobalScheduler(BaseGlobalScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._lp_config: OnlineLpGlobalSchedulerConfig = (
            self._config.cluster_config.global_scheduler_config
        )
        self._policy_name = self._lp_config.policy.lower().strip()
        if self._policy_name != "dual_ahd":
            raise ValueError(
                f"Unsupported Online LP policy: {self._policy_name}. Only dual_ahd is supported."
            )
        self._time_step_seconds = max(1e-6, float(self._lp_config.time_step_seconds))
        self._horizon = max(1, int(self._lp_config.horizon))

        self._B, self._M = self._build_capacities()
        self._sim = OnlineLPSimulator(self._horizon, self._B, self._M)

        self._waiting: List[LPRequest] = []
        self._waiting_ids: Set[int] = set()
        self._recorded_jobs: Set[int] = set()
        self._jobs_seen = 0
        self._current_step = 1
        self._current_time_seconds = 0.0

        self._policy = DualAHDPolicy(
            self._sim,
            grad_steps=self._lp_config.grad_steps,
            step_size=self._lp_config.step_size,
            batch_size=self._lp_config.batch_size,
            mem_scale_factor=self._lp_config.mem_scale_factor,
            backlog_gain=self._lp_config.backlog_gain,
            backlog_slack=self._lp_config.backlog_slack,
            length_penalty=self._lp_config.length_penalty,
            reward_normalizer=self._lp_config.reward_normalizer,
        )
        self._max_waiting = max(0, int(self._lp_config.max_waiting))

    def set_time(self, time_seconds: float) -> None:
        self._current_time_seconds = max(self._current_time_seconds, time_seconds)

    def has_pending_requests(self) -> bool:
        return bool(self._request_queue) or bool(self._waiting)

    def _select_waiting(self) -> List[LPRequest]:
        if self._max_waiting <= 0 or len(self._waiting) <= self._max_waiting:
            return self._waiting
        return self._waiting[: self._max_waiting]

    def _build_capacities(self) -> Tuple[np.ndarray, np.ndarray]:
        batch_size_cap = int(self._config.cluster_config.replica_scheduler_config.batch_size_cap)
        block_size = int(self._config.cluster_config.replica_scheduler_config.block_size)
        num_blocks_override = self._config.cluster_config.replica_scheduler_config.num_blocks
        max_tokens = getattr(self._config.request_generator_config, "max_tokens", None)
        if not max_tokens:
            max_tokens = 1

        B = []
        M = []
        for replica in self._replicas.values():
            memory_planner = MemoryPlanner(self._config.cluster_config.replica_config, replica)
            max_batch_size = min(memory_planner.get_max_batch_size(), batch_size_cap)
            B.append(max_batch_size)

            if num_blocks_override:
                num_blocks = int(num_blocks_override)
            else:
                max_blocks_per_sequence = max(1, max_tokens // block_size)
                num_blocks = max_blocks_per_sequence * memory_planner.get_max_request_slots()
            M.append(num_blocks * block_size)

        return np.array(B, dtype=float), np.array(M, dtype=float)

    def _to_step(self, time_seconds: float) -> int:
        return max(1, int(math.floor(time_seconds / self._time_step_seconds)) + 1)

    def _make_lp_request(self, request: Request) -> LPRequest:
        step = self._to_step(request.arrived_at)
        if step > self._horizon:
            step = self._horizon
        return LPRequest(
            request=request,
            j=request.id,
            t_arr=step,
            s=request.num_prefill_tokens,
            o=request.num_decode_tokens,
        )

    def _estimate_total_jobs(self) -> int:
        if self._lp_config.total_jobs > 0:
            return int(self._estimate_job_bound(self._lp_config.total_jobs))
        num_requests = getattr(self._config.request_generator_config, "num_requests", None)
        if num_requests:
            return int(self._estimate_job_bound(num_requests))
        return max(1, self._jobs_seen + len(self._waiting))

    @staticmethod
    def _estimate_job_bound(value: int) -> int:
        return max(1, int(value))

    def _enqueue_arrivals(self, arrivals: List[LPRequest]) -> None:
        if not arrivals:
            return
        for req in arrivals:
            if req.j in self._waiting_ids:
                continue
            self._waiting.append(req)
            self._waiting_ids.add(req.j)
        self._jobs_seen += len(arrivals)

    def _schedule_dual_ahd(self, step: int) -> List[Tuple[int, Request]]:
        if not self._waiting:
            return []

        total_jobs = float(self._estimate_total_jobs())
        remaining_est = max(1.0, float(self._sim.B.sum()))
        # remaining_est = max(1.0, total_jobs - float(self._jobs_seen))
        remain_thru = np.clip(self._sim.B[:, None] - self._sim.thru_occ, 0.0, None)
        remain_mem = np.clip(self._sim.M[:, None] - self._sim.mem_occ, 0.0, None)
        d_vec_flat = np.concatenate(
            [
                (remain_thru / remaining_est).flatten(),
                ((remain_mem / remaining_est) * self._policy.mem_scale).flatten(),
            ]
        )
        d_vec_flat /= self._policy.reward_normalizer

        backlog_ratio = float(len(self._waiting)) / max(1.0, total_jobs)
        d_vec_flat *= 1.0 + self._policy.backlog_gain * backlog_ratio
        eff_eps = -self._policy.backlog_slack * backlog_ratio
        self._policy.resolve_prices(d_vec_flat)

        waiting = self._select_waiting()
        candidates: List[
            Tuple[float, LPRequest, int, int, float, float, np.ndarray, np.ndarray]
        ] = []
        for req in list(waiting):
            valid, idxs, mem_vals, feas_mask = self._sim.feasible_mask_all_g(req, step)
            if not valid or idxs is None or mem_vals is None or feas_mask is None:
                continue

            _, _, _, _, r_decision, r_report = reward_components(
                self._lp_config.alpha,
                self._lp_config.beta,
                self._lp_config.gamma,
                self._lp_config.sigma,
                req.o,
                self._horizon,
                step,
                req.t_arr,
                self._lp_config.goodput_window,
                self._lp_config.goodput_weight,
            )
            r_norm = min(r_decision / self._policy.reward_normalizer, 1.0)

            margins = np.full(self._sim.G, -np.inf, dtype=float)
            len_penalty = self._policy.length_penalty * float(req.o)
            for g in range(self._sim.G):
                if not feas_mask[g]:
                    continue
                margins[g] = r_norm - self._policy.column_inner(g, idxs, mem_vals) - len_penalty
                if margins[g] > eff_eps:
                    candidates.append(
                        (margins[g], req, g, step, r_decision, r_report, idxs, mem_vals)
                    )

            if req.j not in self._recorded_jobs:
                g_star = int(np.argmax(margins))
                self._policy.record_sample(req, g_star, step, idxs, mem_vals, r_norm)
                self._recorded_jobs.add(req.j)

        if not candidates:
            if backlog_ratio > 0 and self._policy.backlog_slack > 0:
                self._policy.p *= 0.5
            return []

        candidates.sort(key=lambda x: (-x[0], x[1].t_arr, x[1].j, x[3], x[2]))

        started_ids: Set[int] = set()
        mappings: List[Tuple[int, Request]] = []
        for (
            _,
            req_cand,
            g_cand,
            tau_cand,
            _,
            _,
            idxs_cand,
            mem_vals_cand,
        ) in candidates:
            if req_cand.j not in self._waiting_ids or req_cand.j in started_ids:
                continue
            if not self._sim.feasible_if_start(req_cand, g_cand, tau_cand):
                continue
            self._sim.start_job(req_cand, g_cand, tau_cand, idxs_cand, mem_vals_cand)
            mappings.append((g_cand, req_cand.request))
            started_ids.add(req_cand.j)

        if started_ids:
            self._waiting = [req for req in self._waiting if req.j not in started_ids]
            self._waiting_ids.difference_update(started_ids)

        return mappings

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        if not self._request_queue and not self._waiting:
            return []

        self._current_step = max(
            self._current_step, self._to_step(self._current_time_seconds)
        )

        new_requests = self._request_queue
        self._request_queue = []

        arrivals_by_step: Dict[int, List[LPRequest]] = {}
        for request in new_requests:
            lp_req = self._make_lp_request(request)
            arrivals_by_step.setdefault(lp_req.t_arr, []).append(lp_req)

        max_step = max([self._current_step] + list(arrivals_by_step.keys()))

        mappings: List[Tuple[int, Request]] = []
        for step in range(self._current_step, max_step + 1):
            arrivals = arrivals_by_step.get(step, [])
            self._enqueue_arrivals(arrivals)

            mappings.extend(self._schedule_dual_ahd(step))

        self._current_step = max_step
        return mappings
