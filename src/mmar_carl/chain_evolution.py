"""
ChainEvolver — evolutionary search over ``ReasoningChain`` configurations.

Given a base chain, a labelled dataset, and a scoring metric, ``ChainEvolver``
runs an evolutionary loop: it generates a population of mutated chain variants,
evaluates each one on the dataset, retains the top scorers ("elitism"), and
breeds the next generation by mutating selected parents. After ``generations``
rounds it returns the best chain seen along with a per-generation fitness
history.

This is a coarse but useful tool for prompt / model / parallelism sweeps:
"which step prompts, model choices, and max_workers value give the best
mean dataset score?" — without writing N hand-rolled variants.

Mutation kinds (each toggleable via the corresponding pool argument on
``ChainMutator``):

* ``prompt_rewrite`` — append a randomly-chosen suffix to one LLM step's
  ``aim`` text. Pool: ``aim_suffix_pool``.
* ``model_swap`` — set ``llm_config.model`` on one LLM step. Pool: ``model_pool``.
* ``temperature_swap`` — set ``llm_config.temperature`` on one LLM step.
  Pool: ``temperature_pool``.
* ``max_workers`` — change the chain-level ``max_workers``. Pool:
  ``max_workers_pool``.

All mutations are non-destructive: the returned chain is built fresh from
``chain.to_dict()`` so the original chain is never modified.
"""

from __future__ import annotations

import asyncio
import math
import random
import warnings
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from .dataset_evaluator import DatasetEvaluator
from .logging_utils import log_info, log_warning
from .metrics import MetricBase
from .models.dataset import AbstractDataset, DatasetEvaluationReport, ThresholdStrategy
from .models.enums import StepType

__all__ = [
    "ChainEvolver",
    "ChainMutator",
    "EvolutionCostEstimate",
    "EvolutionResult",
    "GenerationStats",
    "IndividualMetrics",
    "MutationKind",
    "format_runs_pareto",
]


class IndividualMetrics(BaseModel):
    """Per-individual runtime metrics captured during one generation.

    Surfaces the speed/quality trade-off: a chain variant that scores high
    but takes 3× longer than the rest of the population is now visible in
    the history directly. Paired by index with
    :attr:`GenerationStats.population_scores`.
    """

    score: float = Field(description="Fitness score for this individual (mean across dataset).")
    wall_time_s: float = Field(
        default=0.0, description="Total wall-clock time spent evaluating this individual."
    )
    total_tokens: int = Field(
        default=0, description="Total tokens (prompt + completion) consumed across all cases."
    )
    llm_calls: int = Field(
        default=0,
        description=(
            "Total number of LLM steps invoked across all cases (number of "
            "step-runs that recorded token usage). Useful for budgeting and "
            "estimating per-call latency."
        ),
    )
    mutation_kind: Optional[str] = Field(
        default=None,
        description=(
            "Name of the :class:`MutationKind` that produced this individual "
            "from its parent (e.g. ``'prompt_rewrite'``, ``'model_swap'``). "
            "None for generation-0 seeds, elitism clones, or when the mutator "
            "couldn't apply any mutation. Stored as the string value of the "
            "enum so pydantic round-trips cleanly."
        ),
    )
    parent_score: Optional[float] = Field(
        default=None,
        description=(
            "Fitness score of the parent that was mutated to produce this "
            "individual. None for generation-0 seeds and elitism clones. "
            "When set, ``score - parent_score`` is the mutation's "
            "score delta — used by :meth:`EvolutionResult.format_mutation_effectiveness`."
        ),
    )
    scores_by_metric: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-metric mean scores across the dataset, keyed by metric "
            "name. Populated when :class:`ChainEvolver` is configured "
            "with multiple metrics; "
            "empty when only a single metric is used (the legacy path). "
            "``score`` always carries the composite fitness used for "
            "selection — derived from this dict via the evolver's "
            "``fitness_fn``."
        ),
    )


class MutationKind(str, Enum):
    PROMPT_REWRITE = "prompt_rewrite"
    MODEL_SWAP = "model_swap"
    TEMPERATURE_SWAP = "temperature_swap"
    MAX_WORKERS = "max_workers"
    DELETE_STEP = "delete_step"
    INSERT_STEP = "insert_step"


class GenerationStats(BaseModel):
    """Per-generation summary returned by :meth:`ChainEvolver.evolve`."""

    generation: int = Field(description="0-indexed generation number")
    best_score: float
    mean_score: float
    population_scores: list[float] = Field(
        default_factory=list,
        description="Score per individual (ordered by descending score).",
    )
    population_metrics: list[IndividualMetrics] = Field(
        default_factory=list,
        description=(
            "Per-individual {score, wall_time_s, total_tokens, llm_calls} — "
            "paired by index with ``population_scores``. Surfaces the "
            "speed/quality trade-off across the population."
        ),
    )
    best_chain_spec: dict[str, Any] = Field(
        default_factory=dict,
        description="Serialized form of the best chain in this generation (via chain.to_dict()).",
    )


class EvolutionResult(BaseModel):
    """Return value of :meth:`ChainEvolver.evolve`."""

    best_chain_spec: dict[str, Any] = Field(
        description="Serialized best chain across all generations."
    )
    best_score: float
    best_generation: int
    history: list[GenerationStats] = Field(default_factory=list)

    def _repr_markdown_(self) -> str:
        """Rich-display protocol for Jupyter.

        Markdown summary: headline score + generation count + a
        text-format score-evolution chart so the user sees the
        convergence trajectory by typing ``result`` at the prompt.
        """
        n_gen = len(self.history)
        lines = [
            f"**EvolutionResult** — best score: **{self.best_score:.3f}** "
            f"@ generation {self.best_generation} · "
            f"{n_gen} generation{'s' if n_gen != 1 else ''} completed",
            "",
        ]
        if n_gen >= 1:
            lines.append("```text")
            lines.append(self.format_score_evolution())
            lines.append("```")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Score-evolution visualization
    # ------------------------------------------------------------------

    def format_score_evolution(
        self,
        *,
        format: str = "text",
        height: int = 10,
        width: int = 50,
        png_path: Optional[str] = None,
    ) -> str:
        """Render a per-generation score chart with best / mean / worst lines.

        Answers the key evolution question: "is it converging, stagnating,
        or oscillating?" — without needing to scrape ``history`` manually.

        * ``format="text"`` (default): Unicode line chart with three series
          overlaid. ``height`` rows tall, ``width`` columns wide. Best line
          uses ``█``, mean uses ``▒``, worst uses ``░``. Y-axis labelled
          with min/max score; X-axis labelled with generation indices.
          Legend at the bottom.
        * ``format="png"``: matplotlib line chart with shaded inter-quartile
          band (q1..q3) between worst-min and best-max envelopes. Requires
          ``mmar-carl[viz]`` extra; raises ``ImportError`` otherwise.
          Writes to ``png_path`` (required) and returns the absolute path.

        Empty history returns a one-line placeholder rather than an empty
        chart.
        """
        if not self.history:
            return "(no generations in history — nothing to chart)"

        # Extract series: (best, mean, worst) per generation.
        gens = [stats.generation for stats in self.history]
        bests = [stats.best_score for stats in self.history]
        means = [stats.mean_score for stats in self.history]
        worsts = [
            min(stats.population_scores) if stats.population_scores else stats.best_score
            for stats in self.history
        ]

        # Replace +/-inf with finite proxies so we can chart them.
        def _finitize(values: list[float]) -> list[float]:
            return [v if math.isfinite(v) else 0.0 for v in values]

        bests_f = _finitize(bests)
        means_f = _finitize(means)
        worsts_f = _finitize(worsts)

        if format == "text":
            return self._format_score_evolution_text(
                gens, bests_f, means_f, worsts_f, height, width
            )
        if format == "png":
            return self._format_score_evolution_png(
                gens, bests, means, worsts, png_path
            )
        raise ValueError(
            f"Unknown format {format!r}. Use 'text' or 'png'."
        )

    @staticmethod
    def _format_score_evolution_text(
        gens: list[int],
        bests: list[float],
        means: list[float],
        worsts: list[float],
        height: int,
        width: int,
    ) -> str:
        all_values = bests + means + worsts
        y_min = min(all_values)
        y_max = max(all_values)
        if y_max == y_min:
            y_max = y_min + 1.0  # Avoid div-by-zero on perfectly flat history

        n_gens = len(gens)
        # Column index per generation — evenly spaced across the width.
        if n_gens == 1:
            cols = [width // 2]
        else:
            cols = [int(i * (width - 1) / (n_gens - 1)) for i in range(n_gens)]

        def _row_for(value: float) -> int:
            """Y position 0 = top (max), height-1 = bottom (min)."""
            frac = (value - y_min) / (y_max - y_min)
            return int((1.0 - frac) * (height - 1))

        # Build the canvas. Higher-priority markers overwrite lower ones:
        # worst < mean < best, so paint in that order.
        canvas: list[list[str]] = [[" "] * width for _ in range(height)]
        for col, w in zip(cols, worsts):
            canvas[_row_for(w)][col] = "░"
        for col, m in zip(cols, means):
            canvas[_row_for(m)][col] = "▒"
        for col, b in zip(cols, bests):
            canvas[_row_for(b)][col] = "█"

        # Y-axis labels (max top, min bottom)
        lines = []
        for row_idx, row in enumerate(canvas):
            if row_idx == 0:
                y_label = f"{y_max:>6.3f} "
            elif row_idx == height - 1:
                y_label = f"{y_min:>6.3f} "
            else:
                y_label = " " * 7
            lines.append(y_label + "│" + "".join(row))

        # X-axis tick row
        axis = " " * 7 + "└" + "─" * width
        lines.append(axis)
        # X-axis labels — gen index at each gen's column
        x_label_row = list(" " * (8 + width))
        for col, g in zip(cols, gens):
            label = str(g)
            for i, ch in enumerate(label):
                pos = 8 + col + i
                if pos < len(x_label_row):
                    x_label_row[pos] = ch
        lines.append("".join(x_label_row).rstrip())
        # X-axis title
        lines.append(" " * 8 + "generation")
        # Legend
        lines.append("")
        lines.append("legend:  █ best   ▒ mean   ░ worst")
        return "\n".join(lines)

    @staticmethod
    def _format_score_evolution_png(
        gens: list[int],
        bests: list[float],
        means: list[float],
        worsts: list[float],
        png_path: Optional[str],
    ) -> str:
        if png_path is None:
            raise ValueError("format='png' requires png_path=<destination>.")
        from ._optional_deps import require_matplotlib  # noqa: PLC0415

        plt = require_matplotlib()
        import os  # noqa: PLC0415

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(gens, bests, label="best", marker="o", linewidth=2)
        ax.plot(gens, means, label="mean", marker="s", linestyle="--")
        ax.plot(gens, worsts, label="worst", marker="^", linestyle=":")
        # Shaded envelope between worst and best for quick "spread" reading
        ax.fill_between(gens, worsts, bests, alpha=0.15, label="spread")
        ax.set_xlabel("generation")
        ax.set_ylabel("score")
        ax.set_title("Score evolution per generation")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(gens)

        abs_path = os.path.abspath(png_path)
        fig.savefig(abs_path, bbox_inches="tight")
        plt.close(fig)
        return abs_path

    # ------------------------------------------------------------------
    # Spend-vs-quality Pareto
    # ------------------------------------------------------------------

    def format_spend_vs_quality(
        self,
        *,
        format: str = "text",
        height: int = 10,
        width: int = 50,
        cost_per_1k: Optional[float] = None,
        png_path: Optional[str] = None,
    ) -> str:
        """Per-generation cumulative-spend vs best-so-far score chart.

        Answers "is more evolution worth it, or am I past the knee of the
        diminishing-returns curve?" by tracing cumulative token (or USD)
        spend on the x-axis against best-so-far score on the y-axis. Each
        generation contributes one point; the curve typically climbs
        steeply then plateaus — the elbow is the "stop spending" signal.

        Uses `GenerationStats.population_metrics[i].total_tokens` to
        compute spend, accumulated generation-by-generation.

        Args:
            format: ``"text"`` (default, Unicode chart) or ``"png"``
                (matplotlib scatter+line; requires ``mmar-carl[viz]``).
            height: Chart rows (text only).
            width: Chart cols (text only).
            cost_per_1k: Optional uniform price-per-1k-tokens — sufficient
                when the chain uses a single model. When provided, the
                x-axis flips to USD with proper cost labels. Per-step model
                attribution isn't stored on ``IndividualMetrics``, so a
                full pricing dict isn't supported here; for multi-model
                cost analysis, use the chain-level ``format_cost_by_model``.
            png_path: Required when ``format='png'``.

        Returns:
            Rendered string (text mode) or absolute PNG path (png mode).
            Empty history returns a one-line placeholder.
        """
        if not self.history:
            return "(no generations in history — nothing to chart)"

        gens = [stats.generation for stats in self.history]
        cum_tokens: list[int] = []
        best_so_far: list[float] = []
        running_tokens = 0
        running_best = float("-inf")
        for stats in self.history:
            gen_tokens = sum(m.total_tokens for m in stats.population_metrics)
            running_tokens += gen_tokens
            running_best = max(running_best, stats.best_score)
            cum_tokens.append(running_tokens)
            best_so_far.append(running_best if math.isfinite(running_best) else 0.0)

        if cost_per_1k is not None:
            x_values = [t / 1000.0 * cost_per_1k for t in cum_tokens]
            x_label = "cumulative cost (USD)"
            x_format = "${:.4f}"
        else:
            x_values = [float(t) for t in cum_tokens]
            x_label = "cumulative tokens"
            x_format = "{:,.0f}"

        if format == "text":
            return self._format_spend_vs_quality_text(
                gens, x_values, best_so_far, x_label, x_format, height, width,
            )
        if format == "png":
            return self._format_spend_vs_quality_png(
                gens, x_values, best_so_far, x_label, png_path,
            )
        raise ValueError(
            f"Unknown format {format!r}. Use 'text' or 'png'."
        )

    @staticmethod
    def _format_spend_vs_quality_text(
        gens: list[int],
        x_values: list[float],
        y_values: list[float],
        x_label: str,
        x_format: str,
        height: int,
        width: int,
    ) -> str:
        x_min = min(x_values)
        x_max = max(x_values)
        y_min = min(y_values)
        y_max = max(y_values)
        if x_max == x_min:
            x_max = x_min + 1.0
        if y_max == y_min:
            y_max = y_min + 1.0

        canvas: list[list[str]] = [[" "] * width for _ in range(height)]
        for gen_idx, (x, y) in enumerate(zip(x_values, y_values)):
            col = int(((x - x_min) / (x_max - x_min)) * (width - 1))
            row = int((1.0 - (y - y_min) / (y_max - y_min)) * (height - 1))
            canvas[row][col] = "█"
            if gen_idx == len(x_values) - 1:
                canvas[row][col] = "◉"

        lines: list[str] = []
        for row_idx, row in enumerate(canvas):
            if row_idx == 0:
                y_label = f"{y_max:>6.3f} "
            elif row_idx == height - 1:
                y_label = f"{y_min:>6.3f} "
            else:
                y_label = " " * 7
            lines.append(y_label + "│" + "".join(row))

        lines.append(" " * 7 + "└" + "─" * width)
        min_str = x_format.format(x_min)
        max_str = x_format.format(x_max)
        x_axis_label = (
            " " * 8 + min_str + " " * max(0, width - len(min_str) - len(max_str)) + max_str
        )
        lines.append(x_axis_label)
        lines.append(" " * 8 + x_label)
        lines.append("")
        lines.append(
            f"legend:  █ generation midpoint   ◉ latest generation "
            f"({len(gens)} total)"
        )
        # Plateau detection — marginal gain of the last generation
        # (best_so_far[-1] − best_so_far[-2]) as a fraction of total gain
        # (best_so_far[-1] − best_so_far[0]). Below 5% = essentially flat.
        if len(y_values) >= 3:
            total_gain = y_values[-1] - y_values[0]
            marginal = y_values[-1] - y_values[-2]
            if total_gain > 0 and marginal / total_gain < 0.05:
                lines.append(
                    "⚠ plateau detected: last generation added <5% of the total "
                    "score gain — consider stopping evolution."
                )
        return "\n".join(lines)

    @staticmethod
    def _format_spend_vs_quality_png(
        gens: list[int],
        x_values: list[float],
        y_values: list[float],
        x_label: str,
        png_path: Optional[str],
    ) -> str:
        if png_path is None:
            raise ValueError("format='png' requires png_path=<destination>.")
        from ._optional_deps import require_matplotlib  # noqa: PLC0415

        plt = require_matplotlib()
        import os  # noqa: PLC0415

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_values, y_values, marker="o", linewidth=2, label="best-so-far")
        ax.scatter(
            [x_values[-1]], [y_values[-1]],
            s=160, marker="*", color="#EF4444", zorder=5,
            label=f"latest (gen {gens[-1]})",
        )
        for g, x, y in zip(gens, x_values, y_values):
            ax.annotate(
                f"g{g}", xy=(x, y), xytext=(5, 5),
                textcoords="offset points", fontsize=8, alpha=0.7,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel("best-so-far score")
        ax.set_title("Evolution spend vs quality")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        abs_path = os.path.abspath(png_path)
        fig.savefig(abs_path, bbox_inches="tight")
        plt.close(fig)
        return abs_path

    # ------------------------------------------------------------------
    # Per-individual Pareto front
    # ------------------------------------------------------------------

    def format_pareto(
        self,
        *,
        format: str = "text",
        height: int = 10,
        width: int = 50,
        cost_per_1k: Optional[float] = None,
        png_path: Optional[str] = None,
    ) -> str:
        """Per-individual Pareto front of score vs cost across all generations.

        Each individual ever evaluated by this evolution is one point on a
        ``(cost, score)`` plane. Pareto-dominant individuals (those where
        you can't get a higher score at lower cost) are marked distinctly.

        Different from :meth:`format_spend_vs_quality`:
        - That method aggregates per *generation* (cumulative spend, best-so-far).
        - This method plots every *individual* on the same axes so the
          user can see "which specific mutation produced an
          out-of-the-curve cheap-and-good chain?".

        Args:
            format: ``"text"`` (Unicode scatter) or ``"png"`` (matplotlib;
                requires ``mmar-carl[viz]``).
            height, width: Text canvas dimensions.
            cost_per_1k: Uniform price-per-1k-tokens. When provided, x-axis
                becomes USD; otherwise raw tokens.
            png_path: Required for ``format="png"``.

        Returns:
            Text chart or absolute path to PNG. Empty history returns a
            one-line placeholder.
        """
        if not self.history:
            return "(no generations in history — nothing to chart)"

        # Flatten all individuals across all generations into (cost, score, gen).
        points: list[tuple[float, float, int]] = []
        for stats in self.history:
            for m in stats.population_metrics:
                tokens = m.total_tokens
                cost = (tokens / 1000.0) * cost_per_1k if cost_per_1k is not None else float(tokens)
                if not math.isfinite(m.score):
                    continue
                points.append((cost, m.score, stats.generation))

        if not points:
            return "(no finite-score individuals — nothing to chart)"

        # Pareto-dominance: a point is dominated if some other point has
        # cost <= ours and score > ours (or cost < ours and score >= ours).
        pareto_indices: set[int] = set()
        for i, (c_i, s_i, _) in enumerate(points):
            dominated = False
            for j, (c_j, s_j, _) in enumerate(points):
                if i == j:
                    continue
                if (c_j <= c_i and s_j > s_i) or (c_j < c_i and s_j >= s_i):
                    dominated = True
                    break
            if not dominated:
                pareto_indices.add(i)

        if format == "text":
            return self._format_pareto_text(
                points, pareto_indices, height, width, cost_per_1k,
            )
        if format == "png":
            return self._format_pareto_png(
                points, pareto_indices, png_path, cost_per_1k,
            )
        raise ValueError(
            f"Unknown format {format!r}. Use 'text' or 'png'."
        )

    @staticmethod
    def _format_pareto_text(
        points: list[tuple[float, float, int]],
        pareto_indices: set[int],
        height: int,
        width: int,
        cost_per_1k: Optional[float],
    ) -> str:
        x_values = [c for c, _, _ in points]
        y_values = [s for _, s, _ in points]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        if x_max == x_min:
            x_max = x_min + 1.0
        if y_max == y_min:
            y_max = y_min + 1.0

        canvas: list[list[str]] = [[" "] * width for _ in range(height)]
        # Paint dominated points first, then Pareto-front points on top so
        # they're visible even when overlapping.
        for i, (c, s, _) in enumerate(points):
            if i in pareto_indices:
                continue
            col = int(((c - x_min) / (x_max - x_min)) * (width - 1))
            row = int((1.0 - (s - y_min) / (y_max - y_min)) * (height - 1))
            canvas[row][col] = "·"
        for i, (c, s, _) in enumerate(points):
            if i not in pareto_indices:
                continue
            col = int(((c - x_min) / (x_max - x_min)) * (width - 1))
            row = int((1.0 - (s - y_min) / (y_max - y_min)) * (height - 1))
            canvas[row][col] = "★"

        if cost_per_1k is not None:
            x_label = "cost (USD)"
            x_format = "${:.4f}"
        else:
            x_label = "tokens"
            x_format = "{:,.0f}"

        lines: list[str] = []
        for row_idx, row in enumerate(canvas):
            if row_idx == 0:
                y_label = f"{y_max:>6.3f} "
            elif row_idx == height - 1:
                y_label = f"{y_min:>6.3f} "
            else:
                y_label = " " * 7
            lines.append(y_label + "│" + "".join(row))
        lines.append(" " * 7 + "└" + "─" * width)
        min_str = x_format.format(x_min)
        max_str = x_format.format(x_max)
        x_axis_line = (
            " " * 8 + min_str + " " * max(0, width - len(min_str) - len(max_str)) + max_str
        )
        lines.append(x_axis_line)
        lines.append(" " * 8 + x_label)
        lines.append("")
        lines.append(
            f"legend:  ★ Pareto-dominant   · dominated   "
            f"({len(pareto_indices)} / {len(points)} on front)"
        )
        return "\n".join(lines)

    @staticmethod
    def _format_pareto_png(
        points: list[tuple[float, float, int]],
        pareto_indices: set[int],
        png_path: Optional[str],
        cost_per_1k: Optional[float],
    ) -> str:
        if png_path is None:
            raise ValueError("format='png' requires png_path=<destination>.")
        from ._optional_deps import require_matplotlib  # noqa: PLC0415

        plt = require_matplotlib()
        import os  # noqa: PLC0415

        dom_x = [c for i, (c, _, _) in enumerate(points) if i not in pareto_indices]
        dom_y = [s for i, (_, s, _) in enumerate(points) if i not in pareto_indices]
        pf_x = [c for i, (c, _, _) in enumerate(points) if i in pareto_indices]
        pf_y = [s for i, (_, s, _) in enumerate(points) if i in pareto_indices]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            dom_x, dom_y,
            color="#9CA3AF", alpha=0.6, s=40,
            label=f"dominated ({len(dom_x)})",
        )
        ax.scatter(
            pf_x, pf_y,
            facecolors="none", edgecolors="#EF4444", linewidth=2, s=120,
            marker="o", label=f"Pareto front ({len(pf_x)})",
        )
        # Connect Pareto-front points sorted by cost so the staircase is visible.
        if pf_x:
            pf_sorted = sorted(zip(pf_x, pf_y), key=lambda p: p[0])
            ax.step(
                [p[0] for p in pf_sorted],
                [p[1] for p in pf_sorted],
                where="post", color="#EF4444", alpha=0.4, linestyle="--",
            )
        ax.set_xlabel("cost (USD)" if cost_per_1k is not None else "tokens")
        ax.set_ylabel("score")
        ax.set_title("Pareto front: individuals across generations")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        abs_path = os.path.abspath(png_path)
        fig.savefig(abs_path, bbox_inches="tight")
        plt.close(fig)
        return abs_path

    # ------------------------------------------------------------------
    # Mutation effectiveness bar chart
    # ------------------------------------------------------------------

    def format_mutation_effectiveness(
        self,
        *,
        format: str = "text",
        bar_width: int = 30,
        png_path: Optional[str] = None,
    ) -> str:
        """Per-mutation-kind mean score improvement bar chart.

        Groups every individual produced by mutation (i.e. has both
        ``mutation_kind`` and ``parent_score`` set on its
        :class:`IndividualMetrics`) by mutation kind, computes the mean
        ``score - parent_score`` delta per kind, and renders a horizontal
        bar chart so the user sees which mutations actually move the
        needle for this task.

        Generation-0 seeds and elitism clones are skipped automatically
        (their ``parent_score`` is None). The chart caption includes the
        sample size per kind so the user can judge statistical weight.

        Args:
            format: ``"text"`` (default) or ``"png"`` (requires
                ``mmar-carl[viz]``).
            bar_width: Width of the bar column in characters (text only).
            png_path: Required for ``format="png"``.

        Returns:
            Rendered string (text) or absolute PNG path. Empty when no
            mutated individuals exist yet (gen-0 only / elitism-only runs).
        """
        # Aggregate (mutation_kind → list of deltas).
        deltas_by_kind: dict[str, list[float]] = {}
        for stats in self.history:
            for m in stats.population_metrics:
                if m.mutation_kind is None or m.parent_score is None:
                    continue
                if not math.isfinite(m.score) or not math.isfinite(m.parent_score):
                    continue
                deltas_by_kind.setdefault(m.mutation_kind, []).append(
                    m.score - m.parent_score
                )

        if not deltas_by_kind:
            return (
                "(no mutated individuals with measured parent scores — "
                "evolution has only seeded gen-0 or used elitism so far)"
            )

        # Compute mean deltas and sample sizes.
        rows = [
            (kind, sum(deltas) / len(deltas), len(deltas))
            for kind, deltas in deltas_by_kind.items()
        ]
        rows.sort(key=lambda r: r[1], reverse=True)  # best mean first

        if format == "text":
            return self._format_mutation_effectiveness_text(rows, bar_width)
        if format == "png":
            return self._format_mutation_effectiveness_png(rows, png_path)
        raise ValueError(
            f"Unknown format {format!r}. Use 'text' or 'png'."
        )

    @staticmethod
    def _format_mutation_effectiveness_text(
        rows: list[tuple[str, float, int]],
        bar_width: int,
    ) -> str:
        # Symmetric scale so positive (good) and negative (bad) deltas are
        # both visible with bars going out from a center line.
        max_abs = max(abs(mean) for _, mean, _ in rows)
        if max_abs == 0:
            max_abs = 1.0  # avoid divide-by-zero on all-zero deltas

        kind_width = max(
            len("mutation"), max(len(kind) for kind, _, _ in rows)
        )
        sample_width = max(len("n"), max(len(str(n)) for _, _, n in rows))

        lines = [
            f"{'mutation':<{kind_width}}  {'mean Δ score':>13}  "
            f"{'n':>{sample_width}}  bar"
        ]
        lines.append(
            "-" * (kind_width + 2 + 13 + 2 + sample_width + 2 + bar_width + 1)
        )
        center = bar_width // 2
        for kind, mean_delta, n in rows:
            cells = int((abs(mean_delta) / max_abs) * center)
            cells = max(cells, 1) if mean_delta != 0 else 0
            if mean_delta >= 0:
                bar = " " * center + "█" * cells
            else:
                bar = " " * (center - cells) + "█" * cells + " " * center
            lines.append(
                f"{kind:<{kind_width}}  {mean_delta:>+13.4f}  "
                f"{n:>{sample_width}}  |{bar}|"
            )
        lines.append(
            "-" * (kind_width + 2 + 13 + 2 + sample_width + 2 + bar_width + 1)
        )
        lines.append("legend:  bars left of center = negative Δ (mutation hurt)")
        lines.append("         bars right of center = positive Δ (mutation helped)")
        lines.append("         n = sample size; ranked by mean Δ descending")
        return "\n".join(lines)

    @staticmethod
    def _format_mutation_effectiveness_png(
        rows: list[tuple[str, float, int]],
        png_path: Optional[str],
    ) -> str:
        if png_path is None:
            raise ValueError("format='png' requires png_path=<destination>.")
        from ._optional_deps import require_matplotlib  # noqa: PLC0415

        plt = require_matplotlib()
        import os  # noqa: PLC0415

        kinds = [kind for kind, _, _ in rows]
        means = [mean for _, mean, _ in rows]
        ns = [n for _, _, n in rows]
        # Green for positive, red for negative.
        colors = ["#22C55E" if m >= 0 else "#EF4444" for m in means]

        fig, ax = plt.subplots(figsize=(8, max(3, len(rows) * 0.6)))
        bars = ax.barh(kinds, means, color=colors, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="#374151", linewidth=1)
        # Sample-size annotation per bar.
        for bar, n in zip(bars, ns):
            width = bar.get_width()
            ax.annotate(
                f"n={n}", xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5 if width >= 0 else -5, 0),
                textcoords="offset points",
                ha="left" if width >= 0 else "right", va="center", fontsize=9,
            )
        ax.set_xlabel("mean Δ score (child − parent)")
        ax.set_title("Mutation effectiveness")
        ax.grid(True, axis="x", alpha=0.3)
        ax.invert_yaxis()  # best mutation at top

        abs_path = os.path.abspath(png_path)
        fig.savefig(abs_path, bbox_inches="tight")
        plt.close(fig)
        return abs_path

    def to_lineage_mermaid(
        self,
        *,
        score_decimals: int = 2,
        score_tolerance: float = 1e-6,
    ) -> str:
        """Mermaid tree of every generation's individuals + parent edges.

        Each individual becomes a node labeled
        ``G<gen>·#<idx>\\n<mutation_kind>\\n<score>``, organised into one
        ``subgraph`` per generation. Parent edges are recovered
        heuristically: for every individual in generation N>0 whose
        :class:`IndividualMetrics` records a ``parent_score``, we look
        up the prior-generation individual whose score matches within
        ``score_tolerance``. Seeds and elitism clones (no
        ``parent_score``) have no incoming edge — they sit at the root
        of their lineage.

        The best chain across all generations (``best_score`` at
        ``best_generation``) is rendered with a gold border so the
        user can find the winning lineage instantly.

        Args:
            score_decimals: Decimals shown in node labels (default 2).
            score_tolerance: Absolute tolerance used when matching a
                child's ``parent_score`` against the prior generation's
                scores. Default 1e-6 — generally enough since both
                values come from the same float-typed metric output.

        Returns:
            A Mermaid ``flowchart TD`` block (with one ``subgraph`` per
            generation). Empty history returns a one-line placeholder.
        """
        if not self.history:
            return "%% (no history — nothing to chart)"

        lines: list[str] = ["flowchart TD"]
        # Node id factory: G<gen>I<idx> — must be Mermaid-safe.
        def _nid(gen: int, idx: int) -> str:
            return f"G{gen}I{idx}"

        best_nid: str | None = None

        for gen_stats in self.history:
            gen = gen_stats.generation
            lines.append(f"    subgraph gen{gen}[\"Generation {gen}\"]")
            metrics_list = gen_stats.population_metrics or []
            scores = gen_stats.population_scores or [
                m.score for m in metrics_list
            ]
            n = max(len(scores), len(metrics_list))
            for idx in range(n):
                im = metrics_list[idx] if idx < len(metrics_list) else None
                score = (
                    im.score if im is not None
                    else (scores[idx] if idx < len(scores) else 0.0)
                )
                mutation = (
                    im.mutation_kind if im and im.mutation_kind else "seed"
                )
                node_id = _nid(gen, idx)
                # Mermaid ``<br/>`` line break — same as the heatmap fix
                # we shipped earlier; ``\n`` would render as the literal
                # two-character escape in viewers.
                label = f"#{idx}<br/>{mutation}<br/>{score:.{score_decimals}f}"
                # Replace any double quotes in mutation names so the
                # label string parses cleanly.
                safe = label.replace('"', "'")
                lines.append(f'        {node_id}["{safe}"]')
                # Track which node is the best so we can style it after
                # the subgraphs close.
                if (
                    gen == self.best_generation
                    and abs(score - self.best_score) <= score_tolerance
                    and best_nid is None  # only highlight the first match
                ):
                    best_nid = node_id
            lines.append("    end")

        # Parent-edge recovery using ``parent_score`` matching.
        # Build a per-generation score → node_id lookup once.
        score_index: dict[int, list[tuple[float, str]]] = {}
        for gen_stats in self.history:
            gen = gen_stats.generation
            metrics_list = gen_stats.population_metrics or []
            scores = gen_stats.population_scores or [
                m.score for m in metrics_list
            ]
            score_index[gen] = [
                (
                    metrics_list[idx].score
                    if idx < len(metrics_list)
                    else scores[idx],
                    _nid(gen, idx),
                )
                for idx in range(max(len(scores), len(metrics_list)))
            ]

        for gen_stats in self.history:
            gen = gen_stats.generation
            if gen == 0:
                continue  # seeds have no parents in this history view
            prior = score_index.get(gen - 1, [])
            for idx, im in enumerate(gen_stats.population_metrics or []):
                if im.parent_score is None:
                    continue  # elitism clone or seed clone
                # Find the closest matching parent score in gen-1.
                best_match: str | None = None
                best_delta = float("inf")
                for parent_score, parent_id in prior:
                    delta = abs(parent_score - im.parent_score)
                    if delta < best_delta:
                        best_delta = delta
                        best_match = parent_id
                if best_match is None or best_delta > score_tolerance:
                    continue
                child_id = _nid(gen, idx)
                kind = (im.mutation_kind or "mutate").replace('"', "'")
                lines.append(f"    {best_match} -->|{kind}| {child_id}")

        # Best-chain highlight: gold border + bold-ish stroke.
        if best_nid is not None:
            lines.append(
                f"    style {best_nid} fill:#FCD34D,color:#000,"
                "stroke:#B45309,stroke-width:3px,rx:8"
            )
        lines.append(
            f"    %% lineage tree — best: {best_nid or '?'} "
            f"@ generation {self.best_generation} score={self.best_score:.{score_decimals}f}"
        )
        return "\n".join(lines)


class EvolutionCostEstimate(BaseModel):
    """Pre-flight cost projection for :meth:`ChainEvolver.evolve`.

    Multiplies a single ``chain.estimate_cost(...)`` row by the actual
    number of chain executions the evolution loop will perform:
    ``smoke_check + (population_size × generations × len(dataset))``.
    """

    population_size: int
    generations: int
    cases_per_evaluation: int = Field(description="``len(dataset)`` at estimate time")
    smoke_check_enabled: bool
    total_chain_runs: int = Field(
        description="Total chain executions: smoke (1 or 0) + pop * gens * cases"
    )
    per_chain_total_tokens: int = Field(
        description="Token total per single chain run (from base_chain.estimate_cost)"
    )
    per_chain_cost_usd: float = Field(
        description="USD cost per single chain run (from base_chain.estimate_cost)"
    )
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    pricing_missing_models: list[str] = Field(
        default_factory=list,
        description="Models referenced in the chain that lack a pricing entry.",
    )

    def format_summary(self) -> str:
        """Human-readable summary suitable for printing before a run."""
        lines = [
            f"ChainEvolver cost projection — {self.total_chain_runs} chain runs:",
            f"  population_size:      {self.population_size}",
            f"  generations:          {self.generations}",
            f"  cases per evaluation: {self.cases_per_evaluation}",
            f"  smoke check:          {'enabled' if self.smoke_check_enabled else 'disabled'}",
            f"  per-chain tokens:     {self.per_chain_total_tokens:,}",
            f"  per-chain cost:       ${self.per_chain_cost_usd:.4f}",
            f"  total tokens:         {self.total_tokens:,} "
            f"(in: {self.total_input_tokens:,} / out: {self.total_output_tokens:,})",
            f"  total cost:           ${self.total_cost_usd:.4f}",
        ]
        if self.pricing_missing_models:
            lines.append(
                f"  ⚠ missing pricing:    {', '.join(sorted(set(self.pricing_missing_models)))}"
            )
            lines.append(
                "    (cost is a lower bound; supply a `pricing` dict that covers these models for an accurate projection)"
            )
        return "\n".join(lines)


class ChainMutator:
    """Apply one random mutation to a :class:`ReasoningChain`.

    Pools default to empty; mutation kinds whose pool is empty are skipped.
    If *all* pools are empty, :meth:`mutate` returns an exact clone of the
    chain (useful for seeding without diversification).

    Parameters
    ----------
    model_pool:
        Sample destination for ``MutationKind.MODEL_SWAP``.
    temperature_pool:
        Sample destination for ``MutationKind.TEMPERATURE_SWAP``.
    max_workers_pool:
        Sample destination for ``MutationKind.MAX_WORKERS`` — values may be
        ``int`` or the string ``"auto"``.
    aim_suffix_pool:
        Suffixes to append to a chosen LLM step's ``aim`` text.
    enabled_kinds:
        Optional explicit subset of mutation kinds; defaults to all kinds
        whose pool is non-empty.
    """

    def __init__(
        self,
        *,
        model_pool: Optional[list[str]] = None,
        temperature_pool: Optional[list[float]] = None,
        max_workers_pool: Optional[list[int | str]] = None,
        aim_suffix_pool: Optional[list[str]] = None,
        step_template_pool: Optional[list[dict[str, Any]]] = None,
        allow_step_deletion: bool = False,
        enabled_kinds: Optional[list[MutationKind]] = None,
    ) -> None:
        self.model_pool = list(model_pool or [])
        self.temperature_pool = list(temperature_pool or [])
        self.max_workers_pool = list(max_workers_pool or [])
        self.aim_suffix_pool = list(aim_suffix_pool or [])
        # structural mutations: template pool feeds INSERT_STEP;
        # allow_step_deletion gates DELETE_STEP (a destructive op — keep
        # opt-in so existing evolutions don't pick it up by surprise
        # when they had no idea it was about to land).
        self.step_template_pool = list(step_template_pool or [])
        self.allow_step_deletion = bool(allow_step_deletion)

        if enabled_kinds is None:
            enabled_kinds = []
            if self.aim_suffix_pool:
                enabled_kinds.append(MutationKind.PROMPT_REWRITE)
            if self.model_pool:
                enabled_kinds.append(MutationKind.MODEL_SWAP)
            if self.temperature_pool:
                enabled_kinds.append(MutationKind.TEMPERATURE_SWAP)
            if self.max_workers_pool:
                enabled_kinds.append(MutationKind.MAX_WORKERS)
            if self.step_template_pool:
                enabled_kinds.append(MutationKind.INSERT_STEP)
            if self.allow_step_deletion:
                enabled_kinds.append(MutationKind.DELETE_STEP)
        self.enabled_kinds = list(enabled_kinds)

    # ------------------------------------------------------------------

    def mutate(self, chain: Any, rng: random.Random) -> Any:
        """Return a new chain with one randomly chosen mutation applied.

        Backward-compatible wrapper around :meth:`mutate_with_kind`. Callers
        that need to know which mutation was applied (for lineage tracking
        in evolution effectiveness charts) should use
        :meth:`mutate_with_kind` instead.

        If no mutations are enabled or applicable, returns a fresh clone of
        the chain (round-tripped through ``to_dict()`` / ``from_dict()``).
        """
        chain_out, _ = self.mutate_with_kind(chain, rng)
        return chain_out

    def mutate_with_kind(
        self, chain: Any, rng: random.Random
    ) -> "tuple[Any, Optional[MutationKind]]":
        """Return ``(new_chain, applied_mutation_kind)``.

        ``applied_mutation_kind`` is ``None`` when no mutation was applied
        (no pools configured, or no LLM steps to mutate). Otherwise it's
        the :class:`MutationKind` of the mutation that actually changed
        the chain — useful for grouping individuals by what produced them
        and computing per-mutation effectiveness.
        """
        from .chain import ReasoningChain  # noqa: PLC0415

        spec = chain.to_dict()
        if not self.enabled_kinds:
            return ReasoningChain.from_dict(spec, use_typed_steps=True), None

        kinds = list(self.enabled_kinds)
        rng.shuffle(kinds)
        for kind in kinds:
            # Structural mutations (DELETE_STEP / INSERT_STEP) might
            # produce an invalid DAG — bad dep numbers, cycles, etc.
            # Snapshot the spec first so a failed mutation rolls back
            # cleanly and the loop moves on to the next kind.
            snapshot = _json_clone(spec)
            try:
                applied = self._apply_mutation(spec, kind, rng)
            except Exception:
                applied = False
            if not applied:
                spec = snapshot
                continue
            try:
                return ReasoningChain.from_dict(spec, use_typed_steps=True), kind
            except Exception:
                # Roll back — try the next mutation kind.
                spec = snapshot
                continue

        # No applicable mutation (e.g. no LLM steps in the chain).
        return ReasoningChain.from_dict(spec, use_typed_steps=True), None

    # ------------------------------------------------------------------

    def _apply_mutation(
        self, spec: dict[str, Any], kind: MutationKind, rng: random.Random
    ) -> bool:
        """Mutate ``spec`` in-place. Return True if mutation was applied."""
        if kind is MutationKind.MAX_WORKERS:
            if not self.max_workers_pool:
                return False
            spec["max_workers"] = rng.choice(self.max_workers_pool)
            return True

        # Step-level mutations need at least one LLM step.
        llm_step_indexes = self._llm_step_indexes(spec)
        if not llm_step_indexes:
            return False
        idx = rng.choice(llm_step_indexes)
        step = spec["steps"][idx]

        if kind is MutationKind.PROMPT_REWRITE:
            if not self.aim_suffix_pool:
                return False
            suffix = rng.choice(self.aim_suffix_pool)
            current_aim = step.get("aim") or step.get("step_config", {}).get("aim", "")
            new_aim = (current_aim + " " + suffix).strip() if current_aim else suffix
            step["aim"] = new_aim
            # Keep step_config.aim in sync if present (some serializers nest it).
            if isinstance(step.get("step_config"), dict) and "aim" in step["step_config"]:
                step["step_config"]["aim"] = new_aim
            return True

        if kind is MutationKind.MODEL_SWAP:
            if not self.model_pool:
                return False
            self._ensure_llm_config_dict(step)
            step["llm_config"]["model"] = rng.choice(self.model_pool)
            return True

        if kind is MutationKind.TEMPERATURE_SWAP:
            if not self.temperature_pool:
                return False
            self._ensure_llm_config_dict(step)
            step["llm_config"]["temperature"] = rng.choice(self.temperature_pool)
            return True

        if kind is MutationKind.DELETE_STEP:
            return self._apply_delete_step(spec, rng)

        if kind is MutationKind.INSERT_STEP:
            return self._apply_insert_step(spec, rng)

        return False  # pragma: no cover — unreachable given the enum

    # ------------------------------------------------------------------
    # Structural mutations
    # ------------------------------------------------------------------

    def _apply_delete_step(self, spec: dict[str, Any], rng: random.Random) -> bool:
        """Delete a leaf step (one whose number no other step depends on).

        Refuses to delete when:
          * fewer than two steps remain after the delete, OR
          * every step is depended on by something else (no leaves).

        Leaf step ⇒ safe to drop without breaking any dependency edge.
        """
        steps = spec.get("steps") or []
        if len(steps) < 2:
            return False
        # Build the set of all referenced dep numbers.
        referenced: set[int] = set()
        for s in steps:
            for d in (s.get("dependencies") or []):
                referenced.add(int(d))
        leaves = [
            i for i, s in enumerate(steps)
            if int(s.get("number", -1)) not in referenced
        ]
        if not leaves:
            return False
        victim_idx = rng.choice(leaves)
        del steps[victim_idx]
        return True

    def _apply_insert_step(self, spec: dict[str, Any], rng: random.Random) -> bool:
        """Insert a step from the template pool before the chain's last step.

        The inserted step inherits the **dependencies** of the last
        step (so it has access to the same upstream data), and the
        last step's dependencies are rewritten to a single dep on the
        new step — making the inserted step a verification /
        refinement gate before the existing final output.
        """
        if not self.step_template_pool:
            return False
        steps = spec.get("steps") or []
        if not steps:
            return False
        template = rng.choice(self.step_template_pool)
        new_step = _json_clone(template)
        # Strip any user-supplied number / dependencies — we compute
        # them so the resulting spec stays valid.
        new_step.pop("number", None)
        # Number = max(existing) + 1 so the new step's id can't
        # collide with anything already in the chain.
        existing_numbers = [int(s.get("number", 0)) for s in steps]
        new_number = max(existing_numbers) + 1
        new_step["number"] = new_number
        # Pick the last step *by number*, not by list position, so the
        # mutation is robust against partially-out-of-order specs.
        last_idx = max(
            range(len(steps)), key=lambda i: int(steps[i].get("number", 0)),
        )
        last_step = steps[last_idx]
        prior_deps = list(last_step.get("dependencies") or [])
        new_step["dependencies"] = prior_deps
        last_step["dependencies"] = [new_number]
        # Insert just before the last step in list order so the spec
        # is also easy to read.
        steps.insert(last_idx, new_step)
        return True

    @staticmethod
    def _llm_step_indexes(spec: dict[str, Any]) -> list[int]:
        return [
            i
            for i, step in enumerate(spec.get("steps", []))
            if str(step.get("step_type")) in (StepType.LLM.value, "llm")
        ]

    @staticmethod
    def _ensure_llm_config_dict(step: dict[str, Any]) -> None:
        cfg = step.get("llm_config")
        if not isinstance(cfg, dict):
            step["llm_config"] = {}


def _json_clone(obj: Any) -> Any:
    """Deep-copy via JSON round-trip — safe for the JSON-friendly chain spec."""
    import copy as _copy  # noqa: PLC0415
    return _copy.deepcopy(obj)


# ----------------------------------------------------------------------------
# Evolver
# ----------------------------------------------------------------------------


ContextFactory = Callable[[Any], Any]  # DataCase -> ReasoningContext


class ChainEvolver:
    """Evolutionary search over ``ReasoningChain`` variants.

    Parameters
    ----------
    base_chain:
        Starting chain. The original is never mutated; clones are produced
        via ``chain.to_dict()`` / ``from_dict()``.
    dataset:
        ``AbstractDataset`` of test cases used as the fitness landscape.
    metric:
        Per-case scorer applied via :class:`DatasetEvaluator`. Mean score
        across cases is the chain's fitness.
    mutator:
        ``ChainMutator`` controlling the mutation operators. If ``None``,
        the evolver runs as a vanilla "evaluate-and-rank" loop (no mutations
        beyond a clone), which is rarely useful but supported.
    population_size:
        Number of individuals per generation (>= 1).
    generations:
        How many generations to run. ``generations=1`` evaluates the initial
        population once and returns the best — no breeding.
    elitism:
        Top-K individuals carried forward unchanged to the next generation
        (clamped to ``population_size``).
    rng:
        Optional :class:`random.Random` for deterministic runs. Defaults to a
        fresh ``random.Random()`` with no fixed seed.
    selection_strategy:
        Strategy passed to :class:`DatasetEvaluator`. Defaults to
        ``ThresholdStrategy(threshold=0.0)`` (selects nothing — we only care
        about the mean score, not problem-case selection).
    """

    def __init__(
        self,
        base_chain: Any,
        dataset: AbstractDataset,
        metric: "MetricBase | list[MetricBase]",
        *,
        fitness_fn: Optional[Callable[[dict[str, float]], float]] = None,
        mutator: Optional[ChainMutator] = None,
        population_size: int = 6,
        generations: int = 3,
        elitism: int = 2,
        rng: Optional[random.Random] = None,
        selection_strategy: Optional[Any] = None,
        smoke_check: bool = True,
        max_concurrent_individuals: int = 1,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        if population_size < 1:
            raise ValueError("population_size must be >= 1")
        if generations < 1:
            raise ValueError("generations must be >= 1")
        if elitism < 0:
            raise ValueError("elitism must be >= 0")
        if max_concurrent_individuals < 1:
            raise ValueError("max_concurrent_individuals must be >= 1")

        # multi-objective evolution: accept either a single metric
        # (legacy) or a list of metrics with an optional fitness_fn that
        # composes the per-metric scores into a single fitness float.
        if isinstance(metric, list):
            if not metric:
                raise ValueError("metric list must contain at least one MetricBase")
            self.metrics: list[MetricBase] = list(metric)
            seen: set[str] = set()
            for m in self.metrics:
                if m.name in seen:
                    raise ValueError(
                        f"duplicate metric name {m.name!r} — metric names must be unique"
                    )
                seen.add(m.name)
        else:
            self.metrics = [metric]
        # Legacy attribute for code paths that read ``self.metric`` —
        # always points at the first (or only) metric.
        self.metric: MetricBase = self.metrics[0]

        if fitness_fn is None:
            # Default fitness: mean of all metric scores. Identity when
            # there's only one metric, matching the legacy single-metric
            # path.
            def _mean_fitness(scores: dict[str, float]) -> float:
                return sum(scores.values()) / len(scores) if scores else 0.0
            self.fitness_fn: Callable[[dict[str, float]], float] = _mean_fitness
        else:
            self.fitness_fn = fitness_fn

        self.base_chain = base_chain
        self.dataset = dataset
        self.mutator = mutator
        self.population_size = population_size
        self.generations = generations
        self.elitism = min(elitism, population_size)
        self.rng = rng or random.Random()
        self.selection_strategy = selection_strategy or ThresholdStrategy(threshold=0.0)
        self.smoke_check = smoke_check
        self.max_concurrent_individuals = max_concurrent_individuals
        self.checkpoint_path = checkpoint_path

    # ------------------------------------------------------------------

    def estimate_cost(
        self,
        context_factory: ContextFactory,
        *,
        pricing: Optional[dict[str, tuple[float, float]]] = None,
        default_output_tokens: int = 512,
        char_per_token: int = 4,
    ) -> EvolutionCostEstimate:
        """Project total token usage and USD cost for an :meth:`evolve` call.

        Multiplies a single ``base_chain.estimate_cost(...)`` row by the
        number of chain executions the loop will make:
        ``(1 if smoke_check else 0) + population_size × generations ×
        len(dataset)``. Mutated children may use different models /
        ``max_tokens`` values, so this is a coarse projection grounded in
        the *base* chain — for fine-grained per-individual cost, wrap the
        ``ChainMutator`` and call ``estimate_cost`` on each mutated chain.

        Args:
            context_factory: Same callable passed to :meth:`evolve`. Used
                to build a context against the first dataset case for the
                underlying chain estimate.
            pricing: Optional ``{model: (input_per_1k_usd,
                output_per_1k_usd)}`` map — same shape as
                :meth:`ReasoningChain.estimate_cost`.
            default_output_tokens: Forwarded to the underlying chain estimate.
            char_per_token: Forwarded to the underlying chain estimate.

        Returns:
            :class:`EvolutionCostEstimate` with per-run + total fields and
            a list of any models that lacked pricing entries.

        Raises:
            RuntimeError: If the dataset is empty (cannot estimate).
        """
        cases = list(self.dataset)
        if not cases:
            raise RuntimeError(
                "ChainEvolver.estimate_cost: dataset is empty — no cases to estimate against."
            )

        first_ctx = context_factory(cases[0])
        per_chain = self.base_chain.estimate_cost(
            first_ctx,
            pricing=pricing,
            default_output_tokens=default_output_tokens,
            char_per_token=char_per_token,
        )

        n_cases = len(cases)
        n_full_runs = self.population_size * self.generations * n_cases
        n_smoke = 1 if self.smoke_check else 0
        total_runs = n_smoke + n_full_runs

        return EvolutionCostEstimate(
            population_size=self.population_size,
            generations=self.generations,
            cases_per_evaluation=n_cases,
            smoke_check_enabled=self.smoke_check,
            total_chain_runs=total_runs,
            per_chain_total_tokens=per_chain.total_tokens,
            per_chain_cost_usd=per_chain.total_cost_usd,
            total_input_tokens=per_chain.total_input_tokens * total_runs,
            total_output_tokens=per_chain.total_output_tokens * total_runs,
            total_tokens=per_chain.total_tokens * total_runs,
            total_cost_usd=per_chain.total_cost_usd * total_runs,
            pricing_missing_models=[
                row.model
                for row in per_chain.steps
                if row.pricing_missing and row.model is not None
            ],
        )

    # ------------------------------------------------------------------

    async def evolve(self, context_factory: ContextFactory) -> EvolutionResult:
        """Run the evolutionary loop and return the best chain found.

        When ``checkpoint_path`` was passed to the constructor, the evolver:
          (a) writes a JSON checkpoint after every completed generation;
          (b) on start, if the checkpoint file already exists, *resumes*
              from it — skipping the smoke check, restoring the RNG state,
              and continuing from `generation = last_completed + 1`.
        Delete the file to force a fresh run.
        """
        resumed_state = self._load_checkpoint() if self.checkpoint_path else None
        if resumed_state is not None:
            history = resumed_state["history"]
            overall_best_spec = resumed_state["overall_best_spec"]
            overall_best_score = resumed_state["overall_best_score"]
            overall_best_gen = resumed_state["overall_best_gen"]
            self.rng.setstate(resumed_state["rng_state"])
            population = resumed_state["population"]
            self._pending_lineage = resumed_state["pending_lineage"]
            start_gen = resumed_state["next_generation"]
            log_info(
                f"ChainEvolver: resumed from checkpoint at "
                f"{self.checkpoint_path} — continuing from gen {start_gen}"
            )
        else:
            if self.smoke_check:
                await self._run_smoke_check(context_factory)
            population = self._seed_population()
            overall_best_spec: dict[str, Any] = self.base_chain.to_dict()
            overall_best_score: float = float("-inf")
            overall_best_gen: int = 0
            history: list[GenerationStats] = []
            start_gen = 0
        flat_signal_warned: bool = False

        for gen in range(start_gen, self.generations):
            # Snapshot the lineage populated by the most-recent
            # _seed_population / _next_generation call. Same index ordering
            # as `population`.
            lineage = getattr(self, "_pending_lineage", [(None, None)] * len(population))
            scored: list[tuple[float, Any, IndividualMetrics]] = []
            if self.max_concurrent_individuals == 1:
                # Sequential — preserves the original behaviour, deterministic
                # for users with seeded ``random.Random``.
                for idx, ind in enumerate(population):
                    score, metrics = await self._evaluate(ind, context_factory)
                    self._attach_lineage(metrics, lineage, idx)
                    scored.append((score, ind, metrics))
            else:
                # Concurrent evaluation across population — bounded by a
                # semaphore so we don't overwhelm rate-limited LLM APIs.
                evaluated_results = await self._evaluate_population_concurrent(
                    population, context_factory
                )
                for idx, (ind, (score, metrics)) in enumerate(
                    zip(population, evaluated_results)
                ):
                    self._attach_lineage(metrics, lineage, idx)
                    scored.append((score, ind, metrics))

            # Descending by score
            scored.sort(key=lambda t: t[0], reverse=True)
            best_score, best_chain, _ = scored[0]
            scores_only = [s for s, _, _ in scored]
            metrics_only = [m for _, _, m in scored]
            mean_score = sum(scores_only) / len(scores_only) if scores_only else 0.0

            stats = GenerationStats(
                generation=gen,
                best_score=best_score,
                mean_score=mean_score,
                population_scores=scores_only,
                population_metrics=metrics_only,
                best_chain_spec=best_chain.to_dict(),
            )
            history.append(stats)
            log_info(
                f"ChainEvolver: gen {gen} — best={best_score:.4f} mean={mean_score:.4f} "
                f"(n={len(scored)})"
            )

            # Loud no-fitness-signal warning. If every individual scored the
            # same value (especially 0.0 or -inf), selection is random and
            # subsequent generations waste tokens. Emit a single UserWarning
            # the FIRST time this is observed; let the run continue in case
            # the user genuinely wants a constant metric (e.g. for testing).
            if not flat_signal_warned and self._scores_are_flat(scores_only):
                self._warn_no_fitness_signal(gen, scores_only)
                flat_signal_warned = True

            if best_score > overall_best_score:
                overall_best_score = best_score
                overall_best_spec = best_chain.to_dict()
                overall_best_gen = gen

            if gen + 1 < self.generations:
                population = self._next_generation(scored)

            # Persist checkpoint after EACH completed generation so a Ctrl-C
            # mid-run can be picked up exactly where it left off. Snapshot
            # is taken AFTER _next_generation so the resumed run starts the
            # next gen with the right population state.
            if self.checkpoint_path:
                next_pop = population if gen + 1 < self.generations else []
                next_lineage = (
                    getattr(self, "_pending_lineage", [])
                    if gen + 1 < self.generations
                    else []
                )
                self._save_checkpoint(
                    history=history,
                    overall_best_spec=overall_best_spec,
                    overall_best_score=overall_best_score,
                    overall_best_gen=overall_best_gen,
                    next_population=next_pop,
                    next_lineage=next_lineage,
                    completed_gen=gen,
                )

        return EvolutionResult(
            best_chain_spec=overall_best_spec,
            best_score=overall_best_score,
            best_generation=overall_best_gen,
            history=history,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _scores_are_flat(scores: list[float]) -> bool:
        """Return True if all scores in the population are essentially identical.

        Treats ``-inf`` / ``nan`` as flat when the whole population shares it
        (every individual failed identically — no selection signal).
        """
        if len(scores) <= 1:
            return False
        # All identical (including all -inf or all NaN)?
        first = scores[0]
        if all(math.isnan(s) for s in scores):
            return True
        if all(math.isinf(s) for s in scores):
            return all(s == first for s in scores)
        # Numerical: tight tolerance, since chain scores are usually [0, 1].
        return max(scores) - min(scores) < 1e-9

    def _warn_no_fitness_signal(self, generation: int, scores: list[float]) -> None:
        score_repr = (
            f"{scores[0]:.4f}" if not math.isinf(scores[0]) and not math.isnan(scores[0])
            else repr(scores[0])
        )
        warnings.warn(
            (
                f"ChainEvolver: generation {generation} — all {len(scores)} individuals "
                f"scored identically ({score_repr}). Selection has no signal, so "
                f"subsequent generations will explore randomly with no convergence. "
                f"Verify that (a) your metric returns varying scores for varying "
                f"chain outputs, and (b) the dataset isn't trivially solved/failed "
                f"by every variant. Use a case-aware metric "
                f"(see mmar_carl.metrics.call_metric_async) if scoring needs per-case "
                f"ground truth."
            ),
            UserWarning,
            stacklevel=3,
        )

    async def _run_smoke_check(self, context_factory: ContextFactory) -> None:
        """Pre-flight: evaluate the base chain on the first dataset case and
        confirm the metric returns a meaningful score before kicking off the
        full evolutionary loop.

        Raises a ``RuntimeError`` with a diagnostic message when:
        - The dataset is empty (nothing to evaluate against — guaranteed
          flat-signal run).
        - The base chain fails to execute on the first case.
        - The metric raises an exception.
        - The metric returns ``-inf`` / ``NaN`` (commonly a sign that the
          metric or `case` plumbing is broken — see case-aware fix).

        Doesn't raise on a 0.0 score by itself because constant-zero metrics
        are valid for smoke testing — the runtime no-fitness-signal warning
        catches those after the first generation.

        The smoke-check evaluation does count as a real LLM call, but only
        ONE — vs. ``population_size * generations * len(dataset)`` for the
        full run. Configurable via ``ChainEvolver(smoke_check=False)``.
        """
        from .metrics import call_metric_async

        cases = list(self.dataset)
        if not cases:
            raise RuntimeError(
                "ChainEvolver smoke check failed: dataset is empty. "
                "Provide at least one DataCase before calling evolve()."
            )

        first_case = cases[0]
        case_label = first_case.label or f"#{first_case.input[:40]}"
        try:
            ctx = context_factory(first_case)
            result = await self.base_chain.execute_async(ctx)
        except Exception as exc:
            raise RuntimeError(
                f"ChainEvolver smoke check failed: base chain raised "
                f"{type(exc).__name__} on case '{case_label}': {exc}. "
                f"Fix the chain or context_factory before running evolution. "
                f"Pass smoke_check=False to ChainEvolver to skip this check."
            ) from exc

        if not result.success:
            failed = result.get_failed_steps()
            err_summary = (
                "; ".join(
                    f"step {s.step_number} '{s.step_title}': {s.error_message}"
                    for s in failed[:3]
                )
                or "(no per-step error recorded)"
            )
            raise RuntimeError(
                f"ChainEvolver smoke check failed: base chain returned "
                f"success=False on case '{case_label}'. Failed steps: {err_summary}. "
                f"Fix the chain before running evolution. "
                f"Pass smoke_check=False to ChainEvolver to skip this check."
            )

        try:
            score = await call_metric_async(self.metric, result, case=first_case)
        except Exception as exc:
            raise RuntimeError(
                f"ChainEvolver smoke check failed: metric '{self.metric.name}' raised "
                f"{type(exc).__name__} on case '{case_label}': {exc}. "
                f"Fix the metric before running evolution. "
                f"Pass smoke_check=False to ChainEvolver to skip this check."
            ) from exc

        if math.isnan(score) or math.isinf(score):
            raise RuntimeError(
                f"ChainEvolver smoke check failed: metric '{self.metric.name}' returned "
                f"{score!r} on case '{case_label}'. Numeric scores are required for "
                f"evolution. If the metric needs per-case ground truth, declare a "
                f"'case' parameter in compute_async (see mmar_carl.metrics.call_metric_async). "
                f"Pass smoke_check=False to ChainEvolver to skip this check."
            )

        log_info(
            f"ChainEvolver: smoke check OK — base chain on case '{case_label}' "
            f"scored {score:.4f}"
        )

    def _seed_population(self) -> list[Any]:
        from .chain import ReasoningChain

        base_spec = self.base_chain.to_dict()
        population: list[Any] = [
            ReasoningChain.from_dict(base_spec, use_typed_steps=True)
        ]
        # Lineage: (mutation_kind_str or None, parent_score or None) paired
        # by index with population. Generation-0 individuals have no
        # measured parent score, so parent_score=None even for mutants.
        lineage: list[tuple[Optional[str], Optional[float]]] = [(None, None)]
        while len(population) < self.population_size:
            if self.mutator is None:
                population.append(ReasoningChain.from_dict(base_spec, use_typed_steps=True))
                lineage.append((None, None))
            else:
                child, kind = self.mutator.mutate_with_kind(self.base_chain, self.rng)
                population.append(child)
                # parent_score=None because base chain hasn't been evaluated yet.
                lineage.append((kind.value if kind else None, None))
        # Attach lineage as a sibling attribute so the evolve loop can read it
        # without changing the population list shape (preserves backward compat
        # for the `_seed_population()` return type used by external callers
        # / tests that wrap it).
        self._pending_lineage = lineage
        return population

    # ------------------------------------------------------------------
    # Checkpoint / resume
    # ------------------------------------------------------------------

    _CHECKPOINT_VERSION = 1

    def _save_checkpoint(
        self,
        *,
        history: list["GenerationStats"],
        overall_best_spec: dict[str, Any],
        overall_best_score: float,
        overall_best_gen: int,
        next_population: list[Any],
        next_lineage: list[tuple[Optional[str], Optional[float]]],
        completed_gen: int,
    ) -> None:
        """Write evolution state to ``self.checkpoint_path`` atomically.

        Writes via a temp file + rename so a Ctrl-C mid-write doesn't
        produce a corrupt checkpoint. Includes a version stamp so future
        format changes can detect-and-migrate or fail-fast.
        """
        import json  # noqa: PLC0415
        import os  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        rng_state = self.rng.getstate()
        # rng.getstate() returns a tuple; JSON-roundtrip via list. Inner
        # tuple of ints survives list-conversion losslessly.
        payload = {
            "version": self._CHECKPOINT_VERSION,
            "next_generation": completed_gen + 1,
            "completed_gen": completed_gen,
            "history": [stats.model_dump() for stats in history],
            "overall_best_spec": overall_best_spec,
            "overall_best_score": overall_best_score,
            "overall_best_gen": overall_best_gen,
            "rng_state": self._encode_rng_state(rng_state),
            "next_population": [
                chain.to_dict() for chain in next_population
            ],
            "next_lineage": [list(t) for t in next_lineage],
            "base_chain_spec": self.base_chain.to_dict(),  # sanity check on resume
            "population_size": self.population_size,
            "generations": self.generations,
        }
        # Atomic write: tmp + rename in same dir as target.
        target = str(self.checkpoint_path)
        target_dir = os.path.dirname(os.path.abspath(target)) or "."
        os.makedirs(target_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".carl_evo_ckpt_", suffix=".json", dir=target_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp_path, target)
        except Exception:
            # Clean up the temp file if rename failed.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _load_checkpoint(self) -> Optional[dict[str, Any]]:
        """Load a previously-written checkpoint. Returns None if file
        doesn't exist or fails validation."""
        import json  # noqa: PLC0415
        import os  # noqa: PLC0415

        target = str(self.checkpoint_path)
        if not os.path.exists(target):
            return None
        try:
            with open(target, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            log_warning(
                f"ChainEvolver: failed to read checkpoint at {target} "
                f"({exc}); starting fresh."
            )
            return None

        if payload.get("version") != self._CHECKPOINT_VERSION:
            log_warning(
                f"ChainEvolver: checkpoint at {target} has unknown version "
                f"{payload.get('version')!r}; starting fresh."
            )
            return None

        # Sanity check: base chain must match what we'd resume against.
        if payload.get("base_chain_spec") != self.base_chain.to_dict():
            log_warning(
                f"ChainEvolver: checkpoint at {target} was written for a "
                f"different base chain; starting fresh."
            )
            return None

        if (
            payload.get("population_size") != self.population_size
            or payload.get("generations") != self.generations
        ):
            log_warning(
                f"ChainEvolver: checkpoint at {target} has different "
                f"population_size / generations than current evolver; "
                f"starting fresh."
            )
            return None

        # Rehydrate.
        from .chain import ReasoningChain  # noqa: PLC0415

        try:
            history = [
                GenerationStats.model_validate(s) for s in payload["history"]
            ]
            population = [
                ReasoningChain.from_dict(spec, use_typed_steps=True)
                for spec in payload["next_population"]
            ]
            lineage = [
                (kind, ps) for kind, ps in payload["next_lineage"]
            ]
            rng_state = self._decode_rng_state(payload["rng_state"])
        except Exception as exc:
            log_warning(
                f"ChainEvolver: failed to rehydrate checkpoint at {target} "
                f"({exc}); starting fresh."
            )
            return None

        return {
            "history": history,
            "overall_best_spec": payload["overall_best_spec"],
            "overall_best_score": payload["overall_best_score"],
            "overall_best_gen": payload["overall_best_gen"],
            "rng_state": rng_state,
            "population": population,
            "pending_lineage": lineage,
            "next_generation": payload["next_generation"],
        }

    @staticmethod
    def _encode_rng_state(state: Any) -> Any:
        """JSON-friendly encoding for ``random.Random.getstate()`` which
        returns a tuple of (int, tuple_of_ints, Optional[float]). Convert
        inner tuples to lists since JSON can't represent tuples."""
        # state is (version_int, internal_state_tuple, gauss_next_or_None)
        return [state[0], list(state[1]), state[2]]

    @staticmethod
    def _decode_rng_state(encoded: Any) -> Any:
        """Inverse of _encode_rng_state — restores the tuple-of-tuples
        shape that random.Random.setstate() requires."""
        return (encoded[0], tuple(encoded[1]), encoded[2])

    @staticmethod
    def _attach_lineage(
        metrics: "IndividualMetrics",
        lineage: list[tuple[Optional[str], Optional[float]]],
        idx: int,
    ) -> None:
        """Populate ``metrics.mutation_kind`` / ``metrics.parent_score`` from
        the lineage list maintained by ``_seed_population`` /
        ``_next_generation``. Pydantic BaseModel allows attribute mutation
        post-construction, so we assign directly rather than rebuilding the
        whole model. Defensive: silently no-op if the lineage list is too
        short (shouldn't happen but keeps things robust)."""
        if idx >= len(lineage):
            return
        kind, parent_score = lineage[idx]
        metrics.mutation_kind = kind
        metrics.parent_score = parent_score

    async def _evaluate(
        self, chain: Any, context_factory: ContextFactory
    ) -> "tuple[float, IndividualMetrics]":
        """Evaluate one individual; return ``(fitness, metrics)``.

        Single-metric path delegates to :class:`DatasetEvaluator` to
        keep the legacy code-path stable. Multi-metric path drives the
        chain itself once per case and applies every metric to the
        same :class:`ReasoningResult` — avoiding N× the chain
        executions when the user composes multiple metrics.
        """
        try:
            if len(self.metrics) == 1:
                evaluator = DatasetEvaluator(
                    chain=chain,
                    dataset=self.dataset,
                    metric=self.metric,
                    strategy=self.selection_strategy,
                )
                report: DatasetEvaluationReport = await evaluator.evaluate_async(context_factory)
                wall = sum(r.execution_time or 0.0 for r in report.all_results)
                tokens = sum(int(r.token_usage.get("total", 0)) for r in report.all_results)
                calls = sum(r.llm_calls for r in report.all_results)
                metrics = IndividualMetrics(
                    score=report.mean_score,
                    wall_time_s=wall,
                    total_tokens=tokens,
                    llm_calls=calls,
                )
                return report.mean_score, metrics

            # Multi-metric path: run the chain once per case, score
            # with every metric, then compose via fitness_fn.
            from .metrics import call_metric_async  # noqa: PLC0415

            per_metric_scores: dict[str, list[float]] = {
                m.name: [] for m in self.metrics
            }
            wall = 0.0
            tokens = 0
            calls = 0
            for case in list(self.dataset):
                ctx = context_factory(case)
                result = await chain.execute_async(ctx)
                wall += result.total_execution_time or 0.0
                tokens += int(result.token_usage.get("total", 0))
                calls += len(result.token_usage_by_step)
                for m in self.metrics:
                    if result.success:
                        try:
                            s = await call_metric_async(m, result, case=case)
                        except Exception as me:
                            log_warning(
                                f"ChainEvolver: metric '{m.name}' raised on case "
                                f"'{getattr(case, 'label', '?')}': {me}"
                            )
                            s = 0.0
                    else:
                        s = 0.0
                    per_metric_scores[m.name].append(float(s))

            means: dict[str, float] = {
                name: (sum(vals) / len(vals) if vals else 0.0)
                for name, vals in per_metric_scores.items()
            }
            fitness = float(self.fitness_fn(means))
            metrics_obj = IndividualMetrics(
                score=fitness,
                wall_time_s=wall,
                total_tokens=tokens,
                llm_calls=calls,
                scores_by_metric=means,
            )
            return fitness, metrics_obj
        except Exception as exc:
            log_warning(f"ChainEvolver: evaluation failed for an individual — {exc}")
            return float("-inf"), IndividualMetrics(score=float("-inf"))

    async def _evaluate_population_concurrent(
        self, population: list[Any], context_factory: ContextFactory
    ) -> list["tuple[float, IndividualMetrics]"]:
        """Evaluate every individual in ``population`` concurrently with a
        semaphore-bounded concurrency limit.

        Returns results in the SAME order as ``population`` so the caller can
        zip them back with the individuals. This is critical for downstream
        sorting/selection determinism (with a seeded ``rng``).
        """
        sem = asyncio.Semaphore(self.max_concurrent_individuals)

        async def _evaluate_one(ind: Any) -> "tuple[float, IndividualMetrics]":
            async with sem:
                return await self._evaluate(ind, context_factory)

        tasks = [asyncio.create_task(_evaluate_one(ind)) for ind in population]
        return await asyncio.gather(*tasks)

    def _next_generation(
        self, scored: list[tuple[float, Any, "IndividualMetrics"]]
    ) -> list[Any]:
        """Build the next generation: elites carried over, rest are mutants.

        Also populates ``self._pending_lineage`` with one
        ``(mutation_kind_str | None, parent_score | None)`` tuple per child,
        in the same order — read by :meth:`evolve` to attach lineage to
        the next gen's :class:`IndividualMetrics`.
        """
        next_gen: list[Any] = []
        lineage: list[tuple[Optional[str], Optional[float]]] = []

        # Elitism — carry top performers unchanged (via clone). Elites are
        # clones, not mutations, so kind=None and parent_score=None
        # (parent_score=None signals "no mutation was applied"; using None
        # here means the mutation-effectiveness chart correctly skips
        # elite carry-overs when computing per-mutation deltas).
        from .chain import ReasoningChain  # noqa: PLC0415

        for _, chain, _ in scored[: self.elitism]:
            next_gen.append(ReasoningChain.from_dict(chain.to_dict(), use_typed_steps=True))
            lineage.append((None, None))

        # Breed the rest by mutating sampled parents from the top half.
        # Build a parents-and-their-scores list so we can record parent_score
        # when we pick a parent for mutation.
        top_half = scored[: max(1, len(scored) // 2)]
        parents_scored = [(chain, score) for score, chain, _ in top_half]
        while len(next_gen) < self.population_size:
            parent, parent_score = self.rng.choice(parents_scored)
            if self.mutator is None:
                child = ReasoningChain.from_dict(parent.to_dict(), use_typed_steps=True)
                # Cloning isn't a mutation — leave both fields None.
                lineage.append((None, None))
            else:
                child, kind = self.mutator.mutate_with_kind(parent, self.rng)
                lineage.append(
                    (kind.value if kind else None, parent_score)
                )
            next_gen.append(child)
        self._pending_lineage = lineage
        return next_gen


# ----------------------------------------------------------------------------
# Cross-run Pareto helper
# ----------------------------------------------------------------------------


def format_runs_pareto(
    results: list[EvolutionResult],
    *,
    labels: Optional[list[str]] = None,
    format: str = "text",
    height: int = 10,
    width: int = 50,
    cost_per_1k: Optional[float] = None,
    png_path: Optional[str] = None,
) -> str:
    """Cross-run Pareto front: total spend vs final best score per run.

    Companion to :py:meth:`EvolutionResult.format_pareto` (per-individual,
    one run) and :py:meth:`EvolutionResult.format_spend_vs_quality` (per
    generation, one run). This view operates on a SET of runs — typical
    use case: the user launched a deliberate budget sweep (e.g. 6 evolutions
    at varying `population_size` × `generations` combinations) and wants to
    see which budget settings delivered the best score per dollar.

    Each run contributes one point: x = sum of `total_tokens` across all
    individuals in `result.history`; y = `result.best_score`. Pareto-dominant
    runs are marked distinctly so the user can tell which budget level is
    "worth scaling up to" and which is past the diminishing-returns elbow.

    Args:
        results: List of `EvolutionResult` from independent evolution runs.
        labels: Optional list of human-readable labels for each run
            (e.g. `["pop=4 gen=2", "pop=8 gen=4"]`); same length as
            `results`. When omitted, runs are indexed as `r0`, `r1`, ...
        format: ``"text"`` (Unicode scatter) or ``"png"`` (matplotlib;
            requires ``mmar-carl[viz]``).
        height, width: Text canvas dimensions.
        cost_per_1k: Uniform price-per-1k-tokens. When provided, x-axis
            becomes USD; otherwise raw tokens.
        png_path: Required for ``format="png"``.

    Returns:
        Rendered chart string, or absolute PNG path. Empty input returns
        a one-line placeholder.
    """
    if not results:
        return "(no runs supplied — nothing to chart)"
    if labels is not None and len(labels) != len(results):
        raise ValueError(
            f"labels length {len(labels)} must match results length {len(results)}."
        )

    # Compute one (cost, score, label) per run.
    points: list[tuple[float, float, str]] = []
    for idx, run in enumerate(results):
        total_tokens = sum(
            m.total_tokens
            for stats in run.history
            for m in stats.population_metrics
        )
        if cost_per_1k is not None:
            cost = (total_tokens / 1000.0) * cost_per_1k
        else:
            cost = float(total_tokens)
        if not math.isfinite(run.best_score):
            continue
        lbl = labels[idx] if labels else f"r{idx}"
        points.append((cost, run.best_score, lbl))

    if not points:
        return "(no runs had a finite best_score — nothing to chart)"

    # Pareto-dominance: same logic as EvolutionResult.format_pareto.
    pareto_indices: set[int] = set()
    for i, (c_i, s_i, _) in enumerate(points):
        dominated = False
        for j, (c_j, s_j, _) in enumerate(points):
            if i == j:
                continue
            if (c_j <= c_i and s_j > s_i) or (c_j < c_i and s_j >= s_i):
                dominated = True
                break
        if not dominated:
            pareto_indices.add(i)

    if format == "text":
        return _format_runs_pareto_text(points, pareto_indices, height, width, cost_per_1k)
    if format == "png":
        return _format_runs_pareto_png(points, pareto_indices, png_path, cost_per_1k)
    raise ValueError(
        f"Unknown format {format!r}. Use 'text' or 'png'."
    )


def _format_runs_pareto_text(
    points: list[tuple[float, float, str]],
    pareto_indices: set[int],
    height: int,
    width: int,
    cost_per_1k: Optional[float],
) -> str:
    x_values = [c for c, _, _ in points]
    y_values = [s for _, s, _ in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    canvas: list[list[str]] = [[" "] * width for _ in range(height)]
    # Paint dominated first, Pareto-front last so they win overlaps.
    for i, (c, s, _) in enumerate(points):
        if i in pareto_indices:
            continue
        col = int(((c - x_min) / (x_max - x_min)) * (width - 1))
        row = int((1.0 - (s - y_min) / (y_max - y_min)) * (height - 1))
        canvas[row][col] = "·"
    for i, (c, s, _) in enumerate(points):
        if i not in pareto_indices:
            continue
        col = int(((c - x_min) / (x_max - x_min)) * (width - 1))
        row = int((1.0 - (s - y_min) / (y_max - y_min)) * (height - 1))
        canvas[row][col] = "★"

    if cost_per_1k is not None:
        x_label = "total cost (USD)"
        x_format = "${:.4f}"
    else:
        x_label = "total tokens"
        x_format = "{:,.0f}"

    lines: list[str] = []
    for row_idx, row in enumerate(canvas):
        if row_idx == 0:
            y_label = f"{y_max:>6.3f} "
        elif row_idx == height - 1:
            y_label = f"{y_min:>6.3f} "
        else:
            y_label = " " * 7
        lines.append(y_label + "│" + "".join(row))
    lines.append(" " * 7 + "└" + "─" * width)
    min_str = x_format.format(x_min)
    max_str = x_format.format(x_max)
    lines.append(
        " " * 8 + min_str + " " * max(0, width - len(min_str) - len(max_str)) + max_str
    )
    lines.append(" " * 8 + x_label)
    lines.append("")
    # List the Pareto-front runs by label so the user can pick a budget.
    front_labels = sorted(
        [(c, lbl) for i, (c, _, lbl) in enumerate(points) if i in pareto_indices]
    )
    if front_labels:
        labels_str = ", ".join(lbl for _, lbl in front_labels)
        lines.append(
            f"legend:  ★ Pareto-front runs   · dominated   "
            f"({len(pareto_indices)} / {len(points)} on front: {labels_str})"
        )
    else:
        lines.append(
            f"legend:  ★ Pareto-front runs   · dominated   "
            f"({len(pareto_indices)} / {len(points)} on front)"
        )
    return "\n".join(lines)


def _format_runs_pareto_png(
    points: list[tuple[float, float, str]],
    pareto_indices: set[int],
    png_path: Optional[str],
    cost_per_1k: Optional[float],
) -> str:
    if png_path is None:
        raise ValueError("format='png' requires png_path=<destination>.")
    from ._optional_deps import require_matplotlib  # noqa: PLC0415

    plt = require_matplotlib()
    import os  # noqa: PLC0415

    dom_x = [c for i, (c, _, _) in enumerate(points) if i not in pareto_indices]
    dom_y = [s for i, (_, s, _) in enumerate(points) if i not in pareto_indices]
    pf_x = [c for i, (c, _, _) in enumerate(points) if i in pareto_indices]
    pf_y = [s for i, (_, s, _) in enumerate(points) if i in pareto_indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        dom_x, dom_y,
        color="#9CA3AF", alpha=0.6, s=80,
        label=f"dominated ({len(dom_x)})",
    )
    ax.scatter(
        pf_x, pf_y,
        facecolors="none", edgecolors="#EF4444", linewidth=2, s=200,
        marker="o", label=f"Pareto front ({len(pf_x)})",
    )
    # Label each point with its run name so the user can pick a budget.
    for c, s, lbl in points:
        ax.annotate(
            lbl, xy=(c, s), xytext=(5, 5),
            textcoords="offset points", fontsize=8, alpha=0.8,
        )
    if pf_x:
        pf_sorted = sorted(zip(pf_x, pf_y), key=lambda p: p[0])
        ax.step(
            [p[0] for p in pf_sorted],
            [p[1] for p in pf_sorted],
            where="post", color="#EF4444", alpha=0.4, linestyle="--",
        )
    ax.set_xlabel("total cost (USD)" if cost_per_1k is not None else "total tokens")
    ax.set_ylabel("best score")
    ax.set_title("Cross-run Pareto front")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    abs_path = os.path.abspath(png_path)
    fig.savefig(abs_path, bbox_inches="tight")
    plt.close(fig)
    return abs_path
