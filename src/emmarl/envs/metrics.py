"""Metrics tracking for FireSim environment."""

from __future__ import annotations

from dataclasses import dataclass, field

from emmarl.envs.agent import AgentState, AgentStatus


@dataclass
class EpisodeMetrics:
    """Tracks metrics over an episode for visualization."""

    steps: list[int] = field(default_factory=list)
    status_counts: dict[AgentStatus, list[int]] = field(default_factory=dict)
    avg_health: list[float] = field(default_factory=list)
    avg_stamina: list[float] = field(default_factory=list)
    active_incidents: list[int] = field(default_factory=list)
    resolved_incidents: list[int] = field(default_factory=list)
    alive_agents: list[int] = field(default_factory=list)
    responders: list[int] = field(default_factory=list)
    civilians: list[int] = field(default_factory=list)

    def record(
        self,
        step: int,
        agent_states: dict[str, AgentState],
        active_incidents: int,
        resolved_incidents: int,
    ) -> None:
        """Record metrics for current step."""
        self.steps.append(step)

        total_health = 0.0
        total_stamina = 0.0
        alive_count = 0
        responder_count = 0
        civilian_count = 0

        for status in AgentStatus:
            self.status_counts.setdefault(status, []).append(0)

        for agent_id, state in agent_states.items():
            status = state.agent_status
            self.status_counts[status][-1] += 1
            total_health += state.health
            total_stamina += state.stamina
            if state.is_alive():
                alive_count += 1
            if state.agent_status not in (
                AgentStatus.DECEASED,
                AgentStatus.EVACUATED,
            ):
                if "civilian" in agent_id:
                    civilian_count += 1
                else:
                    responder_count += 1

        num_agents = len(agent_states) if agent_states else 1
        self.avg_health.append(total_health / num_agents)
        self.avg_stamina.append(total_stamina / num_agents)
        self.active_incidents.append(active_incidents)
        self.resolved_incidents.append(resolved_incidents)
        self.alive_agents.append(alive_count)
        self.responders.append(responder_count)
        self.civilians.append(civilian_count)
