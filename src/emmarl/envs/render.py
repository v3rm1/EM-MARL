"""Rendering system for FireSim environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx

from emmarl.envs.agent import AgentState, AgentType
from emmarl.envs.map import EmergencyMap, TerrainType, ZoneType
from emmarl.envs.metrics import EpisodeMetrics


class RenderMode(Enum):
    """Rendering modes."""

    MAP = auto()
    GRAPH = auto()
    BOTH = auto()


@dataclass
class RenderConfig:
    """Configuration for rendering."""

    figure_size: tuple[int, int] = (14, 10)
    map_dpi: int = 100
    graph_dpi: int = 100
    show_grid: bool = True
    show_legend: bool = True
    agent_marker_size: int = 100
    zone_alpha: float = 0.3
    incident_alpha: float = 0.6
    show_metrics: bool = True
    metrics_height: float = 0.25


@dataclass
class GraphFilter:
    """Filter configuration for agent graph."""

    agent_types: list[AgentType] | None = None
    min_resource_threshold: float | None = None
    resource_type: str | None = None
    show_collaboration: bool = True
    show_hierarchy: bool = True
    show_proximity: bool = True
    proximity_radius: float = 100.0
    min_stamina: float | None = None
    show_alive_only: bool = True


class FireSimRenderer:
    """Renderer for FireSim environment."""

    def __init__(
        self,
        config: RenderConfig | None = None,
    ) -> None:
        """Initialize renderer."""
        self.config = config or RenderConfig()
        self._fig: plt.Figure | None = None
        self._ax_map: plt.Axes | None = None
        self._ax_graph: plt.Axes | None = None
        self._ax_metrics1: plt.Axes | None = None
        self._ax_metrics2: plt.Axes | None = None

        self._agent_colors = {
            AgentType.MEDIC: "#2ecc71",
            AgentType.FIRE_FORCE: "#e74c3c",
            AgentType.POLICE: "#3498db",
            AgentType.CIVILIAN: "#f39c12",
        }

        self._zone_colors = {
            ZoneType.SAFE: "#27ae60",
            ZoneType.FIRE: "#c0392b",
            ZoneType.FLOODED: "#2980b9",
            ZoneType.COLLAPSED: "#7f8c8d",
            ZoneType.HAZMAT: "#8e44ad",
            ZoneType.CROWDED: "#e67e22",
            ZoneType.MEDICAL_EMERGENCY: "#1abc9c",
            ZoneType.ROAD: "#34495e",
            ZoneType.BUILDING: "#95a5a6",
        }

        self._terrain_colors = {
            TerrainType.OPEN: "#90ee90",
            TerrainType.FOREST: "#228b22",
            TerrainType.GRASS: "#7cfc00",
            TerrainType.URBAN: "#808080",
            TerrainType.ROAD: "#2c3e50",
            TerrainType.WATER: "#3498db",
            TerrainType.BUILDING: "#696969",
            TerrainType.BURNED: "#4a3728",
        }

    def render(
        self,
        emergency_map: EmergencyMap,
        agent_states: dict[str, AgentState],
        agent_types: dict[str, AgentType],
        agent_configs: dict[str, Any],
        mode: RenderMode = RenderMode.BOTH,
        graph_filter: GraphFilter | None = None,
        interactive: bool = False,
        episode_metrics: EpisodeMetrics | None = None,
    ) -> np.ndarray:
        """Render the environment.

        Args:
            emergency_map: The map to render
            agent_states: Dict of agent states
            agent_types: Dict of agent types
            agent_configs: Dict of agent configs
            mode: Rendering mode
            graph_filter: Filter for graph visualization
            interactive: If True, update existing figure instead of creating new one

        Returns:
            Rendered image as numpy array
        """
        has_metrics_data = (
            episode_metrics is not None and len(episode_metrics.steps) > 0
        )
        show_metrics = self.config.show_metrics and has_metrics_data

        needs_new_figure = False

        if not interactive or self._fig is None:
            needs_new_figure = True
        elif self._fig is not None:
            current_has_metrics = self._ax_metrics1 is not None
            if current_has_metrics != show_metrics:
                needs_new_figure = True

        if needs_new_figure:
            if self._fig is not None:
                plt.close(self._fig)

            if mode == RenderMode.BOTH:
                if show_metrics:
                    self._fig = plt.figure(figsize=self.config.figure_size)
                    gs = self._fig.add_gridspec(
                        2, 2, height_ratios=[1, 0.4], hspace=0.3, wspace=0.2
                    )
                    self._ax_map = self._fig.add_subplot(gs[0, 0])
                    self._ax_graph = self._fig.add_subplot(gs[0, 1])
                    self._ax_metrics1 = self._fig.add_subplot(gs[1, 0])
                    self._ax_metrics2 = self._fig.add_subplot(gs[1, 1])
                else:
                    self._fig, (self._ax_map, self._ax_graph) = plt.subplots(
                        1, 2, figsize=self.config.figure_size
                    )
                    self._ax_metrics1 = None
                    self._ax_metrics2 = None
            elif mode == RenderMode.MAP:
                self._fig, self._ax_map = plt.subplots(figsize=(8, 8))
                self._ax_graph = None
                self._ax_metrics1 = None
                self._ax_metrics2 = None
            elif mode == RenderMode.GRAPH:
                if show_metrics:
                    self._fig = plt.figure(figsize=(8, 10))
                    gs = self._fig.add_gridspec(
                        2, 1, height_ratios=[1, 0.4], hspace=0.3
                    )
                    self._ax_graph = self._fig.add_subplot(gs[0, 0])
                    self._ax_map = None
                    self._ax_metrics1 = self._fig.add_subplot(gs[1, 0])
                    self._ax_metrics2 = None
                else:
                    self._fig, self._ax_graph = plt.subplots(figsize=(8, 8))
                    self._ax_map = None
                    self._ax_metrics1 = None
                    self._ax_metrics2 = None
        else:
            if (
                mode == RenderMode.BOTH
                and self._ax_map is not None
                and self._ax_graph is not None
            ):
                self._ax_map.cla()
                self._ax_graph.cla()
                if show_metrics and self._ax_metrics1 is not None:
                    self._ax_metrics1.cla()
                    self._ax_metrics2.cla()
            elif mode == RenderMode.MAP and self._ax_map is not None:
                self._ax_map.cla()
            elif mode == RenderMode.GRAPH and self._ax_graph is not None:
                self._ax_graph.cla()
                if show_metrics and self._ax_metrics1 is not None:
                    self._ax_metrics1.cla()

        if mode == RenderMode.BOTH or mode == RenderMode.MAP:
            self._render_map(emergency_map, agent_states, agent_types)
        if mode == RenderMode.BOTH or mode == RenderMode.GRAPH:
            self._render_graph(
                emergency_map, agent_states, agent_types, agent_configs, graph_filter
            )
        if show_metrics:
            self._render_metrics(episode_metrics)

        if mode == RenderMode.BOTH or mode == RenderMode.MAP:
            plt.tight_layout()

        if interactive:
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            return None
        else:
            self._fig.canvas.draw()
            try:
                img = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                from io import BytesIO

                buf = BytesIO()
                self._fig.savefig(buf, format="png", dpi=self.config.map_dpi)
                buf.seek(0)
                img = plt.imread(buf)
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                elif img.shape[2] == 4:
                    img = img[:, :, :3]
                img = (img * 255).astype(np.uint8)
            return img

    def _render_map(
        self,
        emergency_map: EmergencyMap,
        agent_states: dict[str, AgentState],
        agent_types: dict[str, AgentType],
    ) -> None:
        """Render the map view."""
        self._ax_map.set_xlim(0, emergency_map.width)
        self._ax_map.set_ylim(0, emergency_map.height)
        self._ax_map.set_aspect("equal")
        self._ax_map.set_title("FireSim - Emergency Map (Grid-Based)")
        self._ax_map.set_xlabel("X Position")
        self._ax_map.set_ylabel("Y Position")

        if self.config.show_grid:
            self._ax_map.grid(True, alpha=0.3)

        self._render_terrain(emergency_map)
        self._render_zones(emergency_map)
        self._render_incidents(emergency_map)
        self._render_agents(agent_states, agent_types)

        if self.config.show_legend:
            self._add_map_legend()

    def _render_terrain(self, emergency_map: EmergencyMap) -> None:
        """Render grid terrain."""
        if emergency_map.terrain is None:
            return

        terrain = emergency_map.terrain
        cell_size = terrain.cell_size

        for gy in range(terrain.height):
            for gx in range(terrain.width):
                terrain_type = TerrainType(terrain.terrain[gy, gx])
                color = self._terrain_colors.get(terrain_type, "#90ee90")

                x = gx * cell_size
                y = gy * cell_size

                rect = patches.Rectangle(
                    (x, y),
                    cell_size,
                    cell_size,
                    linewidth=0.5,
                    edgecolor="#333333",
                    facecolor=color,
                    alpha=0.6,
                )
                self._ax_map.add_patch(rect)

    def _render_zones(self, emergency_map: EmergencyMap) -> None:
        """Render zones on the map."""
        for zone in emergency_map.zones:
            color = self._zone_colors.get(zone.zone_type, "#95a5a6")
            (x_min, y_min), (x_max, y_max) = zone.bounds

            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor=color,
                facecolor=color,
                alpha=self.config.zone_alpha * zone.intensity,
            )
            self._ax_map.add_patch(rect)

            self._ax_map.text(
                zone.position[0],
                zone.position[1],
                zone.zone_id,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

    def _render_incidents(self, emergency_map: EmergencyMap) -> None:
        """Render incidents on the map."""
        for incident in emergency_map.incidents:
            if not incident.active:
                continue

            color = self._zone_colors.get(incident.incident_type, "#95a5a6")
            severity = incident.severity

            circle = patches.Circle(
                incident.position,
                radius=20 + severity * 30,
                linewidth=2,
                edgecolor=color,
                facecolor=color,
                alpha=self.config.incident_alpha * severity,
            )
            self._ax_map.add_patch(circle)

            self._ax_map.text(
                incident.position[0],
                incident.position[1],
                f"{incident.incident_id}\n{incident.severity:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
            )

    def _render_agents(
        self,
        agent_states: dict[str, AgentState],
        agent_types: dict[str, AgentType],
    ) -> None:
        """Render agents on the map."""
        for agent_id, state in agent_states.items():
            if agent_id not in agent_types:
                continue

            agent_type = agent_types[agent_id]
            color = self._agent_colors.get(agent_type, "#95a5a6")

            if not state.is_alive():
                marker = "x"
                alpha = 0.5
            elif state.is_exhausted():
                marker = "o"
                alpha = 0.5
            else:
                marker = "o"
                alpha = 1.0

            size = self.config.agent_marker_size
            if state.is_running:
                size *= 1.5

            self._ax_map.scatter(
                state.position[0],
                state.position[1],
                c=color,
                s=size,
                marker=marker,
                alpha=alpha,
                edgecolors="black",
                linewidths=1,
                zorder=10,
            )

            self._ax_map.annotate(
                agent_id,
                state.position,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=6,
                zorder=11,
            )

            if state.get_speed() > 0.1:
                self._ax_map.annotate(
                    "",
                    xy=(
                        state.position[0] + state.velocity[0] * 3,
                        state.position[1] + state.velocity[1] * 3,
                    ),
                    xytext=state.position,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1),
                    zorder=9,
                )

    def _add_map_legend(self) -> None:
        """Add legend for map elements."""
        legend_elements = []

        for agent_type, color in self._agent_colors.items():
            legend_elements.append(
                patches.Patch(facecolor=color, edgecolor="black", label=agent_type.name)
            )

        self._ax_map.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=8,
        )

    def _render_graph(
        self,
        emergency_map: EmergencyMap,
        agent_states: dict[str, AgentState],
        agent_types: dict[str, AgentType],
        agent_configs: dict[str, Any],
        graph_filter: GraphFilter | None = None,
    ) -> None:
        """Render agent relationship graph."""
        self._ax_graph.set_title("FireSim - Agent Relationships")
        self._ax_graph.axis("off")

        graph_filter = graph_filter or GraphFilter()

        G = self._build_agent_graph(
            emergency_map, agent_states, agent_types, agent_configs, graph_filter
        )

        if len(G.nodes) == 0:
            self._ax_graph.text(
                0.5,
                0.5,
                "No agents to display",
                ha="center",
                va="center",
                transform=self._ax_graph.transAxes,
            )
            return

        pos = self._compute_graph_layout(G, agent_states)

        node_colors = [
            self._agent_colors.get(agent_types.get(node, ""), "#95a5a6")
            for node in G.nodes()
        ]

        node_sizes = []
        for node in G.nodes():
            state = agent_states.get(node)
            if state:
                base_size = 300
                size = base_size + (100 - state.stamina) * 2
                node_sizes.append(size)
            else:
                node_sizes.append(300)

        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            ax=self._ax_graph,
            alpha=0.8,
            edgecolors="black",
            linewidths=1,
        )

        nx.draw_networkx_labels(
            G,
            pos,
            labels={n: n for n in G.nodes()},
            font_size=7,
            ax=self._ax_graph,
        )

        edge_colors = []
        edge_widths = []
        for u, v, data in G.edges(data=True):
            edge_type = data.get("type", "proximity")
            if edge_type == "collaboration":
                edge_colors.append("#27ae60")
                edge_widths.append(2.0)
            elif edge_type == "hierarchy":
                edge_colors.append("#9b59b6")
                edge_widths.append(1.5)
            else:
                edge_colors.append("#bdc3c7")
                edge_widths.append(1.0)

        nx.draw_networkx_edges(
            G,
            pos,
            edge_color=edge_colors,
            width=edge_widths,
            ax=self._ax_graph,
            alpha=0.6,
            arrows=True,
            arrowsize=15,
            connectionstyle="arc3,rad=0.1",
        )

        self._add_graph_legend(graph_filter)

    def _build_agent_graph(
        self,
        emergency_map: EmergencyMap,
        agent_states: dict[str, AgentState],
        agent_types: dict[str, AgentType],
        agent_configs: dict[str, Any],
        graph_filter: GraphFilter,
    ) -> nx.DiGraph:
        """Build agent relationship graph."""
        G = nx.DiGraph()

        alive_agents = {
            aid: state
            for aid, state in agent_states.items()
            if not graph_filter.show_alive_only or state.is_alive()
        }

        if graph_filter.agent_types:
            alive_agents = {
                aid: state
                for aid, state in alive_agents.items()
                if agent_types.get(aid) in graph_filter.agent_types
            }

        if graph_filter.min_stamina is not None:
            alive_agents = {
                aid: state
                for aid, state in alive_agents.items()
                if state.stamina >= graph_filter.min_stamina
            }

        for agent_id in alive_agents:
            G.add_node(
                agent_id,
                agent_type=agent_types.get(agent_id),
                state=agent_states[agent_id],
            )

        if graph_filter.show_collaboration:
            self._add_collaboration_edges(G, alive_agents, agent_types, agent_configs)

        if graph_filter.show_hierarchy:
            self._add_hierarchy_edges(G, alive_agents, agent_types)

        if graph_filter.show_proximity:
            self._add_proximity_edges(G, alive_agents, graph_filter.proximity_radius)

        return G

    def _add_collaboration_edges(
        self,
        G: nx.DiGraph,
        agent_states: dict[str, AgentState],
        agent_types: dict[str, AgentType],
        agent_configs: dict[str, Any],
    ) -> None:
        """Add collaboration edges based on recent interactions."""
        responders = {
            aid: at for aid, at in agent_types.items() if at != AgentType.CIVILIAN
        }

        for responder_id in responders:
            if responder_id not in G:
                continue
            responder_state = agent_states.get(responder_id)
            if not responder_state:
                continue

            for other_id, other_state in agent_states.items():
                if other_id == responder_id:
                    continue
                if other_id not in G:
                    continue

                dist = np.sqrt(
                    (responder_state.position[0] - other_state.position[0]) ** 2
                    + (responder_state.position[1] - other_state.position[1]) ** 2
                )

                if dist < 50:
                    G.add_edge(
                        responder_id,
                        other_id,
                        type="collaboration",
                        weight=1.0 - dist / 50,
                    )

    def _add_hierarchy_edges(
        self,
        G: nx.DiGraph,
        agent_states: dict[str, AgentState],
        agent_types: dict[str, AgentType],
    ) -> None:
        """Add hierarchy edges (command structure)."""
        medics = [a for a, t in agent_types.items() if t == AgentType.MEDIC]
        firefighters = [a for a, t in agent_types.items() if t == AgentType.FIRE_FORCE]
        police = [a for a, t in agent_types.items() if t == AgentType.POLICE]

        if medics:
            for m in medics[1:]:
                if m in G and medics[0] in G:
                    G.add_edge(medics[0], m, type="hierarchy", weight=0.8)

        if firefighters:
            for f in firefighters[1:]:
                if f in G and firefighters[0] in G:
                    G.add_edge(firefighters[0], f, type="hierarchy", weight=0.8)

        if police:
            for p in police[1:]:
                if p in G and police[0] in G:
                    G.add_edge(police[0], p, type="hierarchy", weight=0.8)

    def _add_proximity_edges(
        self,
        G: nx.DiGraph,
        agent_states: dict[str, AgentState],
        radius: float,
    ) -> None:
        """Add proximity edges between nearby agents."""
        agents = list(agent_states.keys())

        for i, agent_a in enumerate(agents):
            if agent_a not in G:
                continue

            state_a = agent_states[agent_a]

            for agent_b in agents[i + 1 :]:
                if agent_b not in G:
                    continue

                state_b = agent_states[agent_b]

                dist = np.sqrt(
                    (state_a.position[0] - state_b.position[0]) ** 2
                    + (state_a.position[1] - state_b.position[1]) ** 2
                )

                if dist < radius and dist > 0:
                    G.add_edge(
                        agent_a,
                        agent_b,
                        type="proximity",
                        weight=1.0 - dist / radius,
                    )

    def _compute_graph_layout(
        self, G: nx.DiGraph, agent_states: dict[str, AgentState]
    ) -> dict:
        """Compute graph layout based on map positions."""
        if len(G.nodes) == 0:
            return {}

        try:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        except Exception:
            pos = nx.circular_layout(G)

        return pos

    def _add_graph_legend(self, graph_filter: GraphFilter) -> None:
        """Add legend for graph edges."""
        legend_elements = [
            patches.Patch(facecolor="#2ecc71", edgecolor="black", label="Medic"),
            patches.Patch(facecolor="#e74c3c", edgecolor="black", label="FireForce"),
            patches.Patch(facecolor="#3498db", edgecolor="black", label="Police"),
            patches.Patch(facecolor="#f39c12", edgecolor="black", label="Civilian"),
        ]

        if graph_filter.show_collaboration:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="#27ae60",
                    linewidth=2,
                    label="Collaboration",
                )
            )

        if graph_filter.show_hierarchy:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="#9b59b6",
                    linewidth=1.5,
                    label="Hierarchy",
                )
            )

        if graph_filter.show_proximity:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="#bdc3c7",
                    linewidth=1,
                    label="Proximity",
                )
            )

        self._ax_graph.legend(
            handles=legend_elements,
            loc="upper left",
            fontsize=7,
            framealpha=0.9,
        )

    def _render_metrics(self, episode_metrics: EpisodeMetrics) -> None:
        """Render metrics plots under the network graph."""
        if episode_metrics is None or len(episode_metrics.steps) == 0:
            return

        steps = episode_metrics.steps
        self._render_status_counts(steps, episode_metrics)
        self._render_health_stamina(steps, episode_metrics)

    def _render_status_counts(
        self, steps: list[int], episode_metrics: EpisodeMetrics
    ) -> None:
        """Render agent status counts over time."""
        if self._ax_metrics1 is None:
            return

        self._ax_metrics1.set_title("Agent Status Over Time", fontsize=10)
        self._ax_metrics1.set_xlabel("Step", fontsize=8)
        self._ax_metrics1.set_ylabel("Count", fontsize=8)

        status_colors = {
            "HEALTHY": "#2ecc71",
            "INJURED": "#f39c12",
            "AFFECTED": "#e67e22",
            "CRITICAL": "#e74c3c",
            "DECEASED": "#7f8c8d",
            "SURVIVED": "#3498db",
            "EVACUATED": "#1abc9c",
            "RESCUED": "#9b59b6",
        }

        for status, counts in episode_metrics.status_counts.items():
            if status.name in status_colors and any(c > 0 for c in counts):
                self._ax_metrics1.plot(
                    steps,
                    counts,
                    label=status.name,
                    color=status_colors[status.name],
                    linewidth=1.5,
                )

        self._ax_metrics1.legend(loc="upper right", fontsize=7)
        self._ax_metrics1.grid(True, alpha=0.3)

    def _render_health_stamina(
        self, steps: list[int], episode_metrics: EpisodeMetrics
    ) -> None:
        """Render health, stamina, and incident metrics over time."""
        if self._ax_metrics2 is None:
            return

        self._ax_metrics2.set_title("Health, Stamina & Incidents", fontsize=10)
        self._ax_metrics2.set_xlabel("Step", fontsize=8)
        self._ax_metrics2.set_ylabel("Value", fontsize=8)

        self._ax_metrics2.plot(
            steps,
            episode_metrics.avg_health,
            label="Avg Health",
            color="#e74c3c",
            linewidth=1.5,
        )
        self._ax_metrics2.plot(
            steps,
            episode_metrics.avg_stamina,
            label="Avg Stamina",
            color="#3498db",
            linewidth=1.5,
        )
        self._ax_metrics2.plot(
            steps,
            episode_metrics.active_incidents,
            label="Active Incidents",
            color="#f39c12",
            linewidth=1.5,
            linestyle="--",
        )
        self._ax_metrics2.plot(
            steps,
            episode_metrics.resolved_incidents,
            label="Resolved Incidents",
            color="#2ecc71",
            linewidth=1.5,
            linestyle="--",
        )

        self._ax_metrics2.legend(loc="upper right", fontsize=7)
        self._ax_metrics2.grid(True, alpha=0.3)

    def close(self) -> None:
        """Close the rendering figure."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax_map = None
            self._ax_graph = None
            self._ax_metrics1 = None
            self._ax_metrics2 = None

    def save(self, filepath: str, **kwargs) -> None:
        """Save rendered image to file."""
        if self._fig is not None:
            self._fig.savefig(filepath, **kwargs)
