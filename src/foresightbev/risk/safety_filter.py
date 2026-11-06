"""Stage 6 safety filter: deterministic gates, selection and fallback.

This is an engineered rule set. It carries no formal safety guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

from foresightbev.contracts import (
    DecisionStatus, ReasonCode as R, SafetyDecision, TrajectoryCandidate, TrajectoryRisk,
)


@dataclass
class SafetyConfig:
    max_event_score: float = 0.10
    max_perception_age_s: float = 0.5
    min_healthy_cameras: int = 1
    lane_change_cost: float = 0.25
    lateral_cost: float = 0.05
    risk_cost: float = 2.0
    stopped_speed: float = 0.5
    hold_after_fallback_s: float = 1.0
    hold_speed_on_coverage_loss: bool = False


@dataclass
class PerceptionHealth:
    available: bool
    age_s: float
    healthy_cameras: int
    total_cameras: int


class SafetyFilter:
    """Selection is deterministic. The only state is the time of the last fallback.

    For ``hold_after_fallback_s`` after a fallback, plans that accelerate are
    inadmissible, so a brief drop in evidence cannot immediately undo a stop.
    """

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()
        self.last_fallback: float | None = None

    def _value(self, risk: TrajectoryRisk) -> float:
        return risk.calibrated_probability if risk.calibrated_probability is not None else risk.event_score

    def admissible(self, risk: TrajectoryRisk) -> list[R]:
        problems = []
        if risk.static_collision:
            problems.append(R.NOMINAL_STATIC_COLLISION)
        if self._value(risk) > self.config.max_event_score:
            problems.append(R.NOMINAL_EVENT_SCORE_EXCEEDED)
        if risk.unknown_corridor_violation:
            problems.append(R.NOMINAL_UNKNOWN_CORRIDOR)
        return problems

    def _cost(self, candidate: TrajectoryCandidate, risk: TrajectoryRisk, desired: float, home: float) -> tuple:
        cfg = self.config
        speed = float(candidate.poses[:, 3].mean())
        lateral = abs(candidate.terminal_lateral - home)
        cost = (max(desired - speed, 0.0) / max(desired, 0.1) + cfg.lateral_cost * lateral
                + (cfg.lane_change_cost if lateral > 0.5 else 0.0) + cfg.risk_cost * self._value(risk))
        return round(cost, 9), candidate.candidate_id

    def _fallback(self, candidates, by_id, reasons: list[R], nominal_id, constraints) -> SafetyDecision:
        comfort = next((c for c in candidates if c.candidate_id == "comfort_stop"), None)
        emergency = next((c for c in candidates if c.candidate_id == "emergency_stop"), None)
        if comfort is None or emergency is None:
            raise ValueError("candidate set must contain comfort_stop and emergency_stop")
        risk = by_id.get("comfort_stop")
        moving = comfort.poses[0, 3] > self.config.stopped_speed
        emergency_needed = moving and risk is not None and (risk.static_collision or self._value(risk) > self.config.max_event_score)
        chosen = emergency if emergency_needed else comfort
        code = R.FALLBACK_EMERGENCY_STOP if emergency_needed else R.FALLBACK_COMFORT_STOP
        return SafetyDecision(DecisionStatus.FALLBACK, chosen.candidate_id, chosen, reasons + [code],
                              chosen.candidate_id, nominal_id, constraints)

    def _holding(self, t: float) -> bool:
        return self.last_fallback is not None and t - self.last_fallback < self.config.hold_after_fallback_s

    def decide(self, candidates: list[TrajectoryCandidate], risks: list[TrajectoryRisk], health: PerceptionHealth,
               desired_speed: float, home_lane: float = 0.0) -> SafetyDecision:
        decision = self._decide(candidates, risks, health, desired_speed, home_lane)
        if decision.status == DecisionStatus.FALLBACK and candidates:
            self.last_fallback = candidates[0].t_ref
        return decision

    def _decide(self, candidates, risks, health, desired_speed, home_lane) -> SafetyDecision:
        cfg = self.config
        by_id = {r.candidate_id: r for r in risks}
        nominal = next((c for c in candidates if c.is_nominal), None)
        nominal_id = nominal.candidate_id if nominal else None
        constraints = {"max_event_score": cfg.max_event_score, "max_perception_age_s": cfg.max_perception_age_s}

        # Missing evidence is never a clear road: these gates run before any risk logic.
        if not health.available:
            return self._fallback(candidates, {}, [R.PERCEPTION_MISSING], nominal_id, constraints)
        if health.age_s > cfg.max_perception_age_s:
            return self._fallback(candidates, {}, [R.PERCEPTION_STALE], nominal_id, constraints)
        if health.healthy_cameras < max(cfg.min_healthy_cameras, 1):
            return self._fallback(candidates, {}, [R.CAMERA_COVERAGE_LOST], nominal_id, constraints)

        coverage = [R.CAMERA_COVERAGE_LOST] if health.healthy_cameras < health.total_cameras else []
        holding = bool(candidates) and self._holding(candidates[0].t_ref)

        hold_speed = holding or (bool(coverage) and cfg.hold_speed_on_coverage_loss)

        def accelerates(c: TrajectoryCandidate) -> bool:
            return hold_speed and float(c.poses[-1, 3]) > float(c.poses[0, 3]) + 0.1

        if nominal is not None and nominal.candidate_id in by_id:
            problems = self.admissible(by_id[nominal.candidate_id])
            if accelerates(nominal) and not problems:
                problems = [R.POST_FALLBACK_HOLD if holding else R.COVERAGE_SPEED_HOLD]
            if not problems:
                return SafetyDecision(DecisionStatus.PASS, nominal.candidate_id, nominal,
                                      coverage + [R.NOMINAL_ADMISSIBLE], None, nominal_id, constraints)
        else:
            problems = [R.NOMINAL_EVENT_SCORE_EXCEEDED]

        options = [c for c in candidates if c.candidate_id in by_id and not self.admissible(by_id[c.candidate_id])]
        options = [c for c in options if not c.is_nominal and not c.is_fallback and not accelerates(c)]
        if options:
            # Minimal intervention: stay close to what the nominal plan intended.
            reference = nominal.terminal_lateral if nominal is not None else home_lane
            best = min(options, key=lambda c: self._cost(c, by_id[c.candidate_id], desired_speed, reference))
            return SafetyDecision(DecisionStatus.MODIFY, best.candidate_id, best, coverage + problems, None,
                                  nominal_id, constraints)
        return self._fallback(candidates, by_id, coverage + problems + [R.NO_ADMISSIBLE_CANDIDATE], nominal_id, constraints)
