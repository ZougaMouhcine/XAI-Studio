"""
XAI Studio — Report Service
===========================
Generate HTML and PDF evaluation reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ui.widgets import C
from ui.components.plot_canvas import apply_plot_style
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ReportSection:
    title: str
    figure: plt.Figure | None


class ReportService:
    """Create HTML or PDF reports from evaluation artifacts."""

    def generate_html(self, title: str, summary: dict, sections: list[ReportSection], output_path: str) -> str:
        html = self._build_html(title, summary, sections)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("HTML report generated: %s", output_path)
        return output_path

    def generate_pdf(self, title: str, summary: dict, sections: list[ReportSection], output_path: str) -> str:
        with PdfPages(output_path) as pdf:
            summary_fig = self._summary_figure(title, summary)
            pdf.savefig(summary_fig, bbox_inches="tight")
            plt.close(summary_fig)

            for section in sections:
                if section.figure is None:
                    continue
                pdf.savefig(section.figure, bbox_inches="tight")
                plt.close(section.figure)

        logger.info("PDF report generated: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------
    def _build_html(self, title: str, summary: dict, sections: list[ReportSection]) -> str:
        cards = "\n".join([self._summary_card(k, v) for k, v in summary.items()])
        figures = "\n".join([self._figure_block(s.title, s.figure) for s in sections if s.figure is not None])

        return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<title>{title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; padding: 24px; background: #0f1724; color: #e9f0fb; }}
.header {{ margin-bottom: 20px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
.card {{ background: #1b263a; border: 1px solid #2c3b55; border-radius: 12px; padding: 12px; }}
.card-title {{ font-size: 12px; color: #9fb1ca; text-transform: uppercase; letter-spacing: 1px; }}
.card-value {{ font-size: 20px; margin-top: 6px; color: #e9f0fb; }}
.figure {{ margin-top: 18px; background: #1b263a; border: 1px solid #2c3b55; border-radius: 12px; padding: 12px; }}
.figure img {{ width: 100%; height: auto; border-radius: 8px; }}
.figure h3 {{ margin: 0 0 10px 0; font-size: 16px; color: #e9f0fb; }}
</style>
</head>
<body>
  <div class=\"header\">
    <h1>{title}</h1>
  </div>
  <div class=\"cards\">{cards}</div>
  <div>{figures}</div>
</body>
</html>"""

    def _summary_card(self, key: str, value: Any) -> str:
        return f"""<div class=\"card\">
  <div class=\"card-title\">{key}</div>
  <div class=\"card-value\">{value}</div>
</div>"""

    def _figure_block(self, title: str, fig: plt.Figure) -> str:
        img = self._fig_to_base64(fig)
        return f"""<div class=\"figure\">
  <h3>{title}</h3>
  <img src=\"data:image/png;base64,{img}\" />
</div>"""

    @staticmethod
    def _fig_to_base64(fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    # ------------------------------------------------------------------
    # PDF helpers
    # ------------------------------------------------------------------
    def _summary_figure(self, title: str, summary: dict) -> plt.Figure:
        apply_plot_style()
        fig = plt.figure(figsize=(8.2, 10.2), facecolor=C.BG_CARD)
        ax = fig.add_subplot(111)
        ax.axis("off")

        ax.text(0.02, 0.96, title, fontsize=18, fontweight="bold", color=C.TEXT)
        y = 0.9
        for key, value in summary.items():
            ax.text(0.04, y, f"{key}:", fontsize=11, color=C.TEXT_SEC)
            ax.text(0.35, y, str(value), fontsize=11, color=C.TEXT)
            y -= 0.04
        return fig
