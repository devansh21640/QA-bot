from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle


OUTPUT_PATH = Path("Assignment1_Approach_Insights_Short.pdf")
TITLE = "Assignment 1: Store Sales Analysis & Prediction"
SUBTITLE = "Executive Summary: Approach, Forecasting Logic, and Business Insights"


PAGE_BG = "#FFFFFF"
HEADER_BG = "#0F172A"
HEADER_ACCENT = "#22C55E"
CARD_BG = "#F8FAFC"
CARD_BORDER = "#E2E8F0"
TEXT_DARK = "#111827"
TEXT_MUTED = "#475569"


def draw_header(fig, page_title: str, page_number: int) -> None:
    fig.patches.append(
        Rectangle((0.0, 0.905), 1.0, 0.095, transform=fig.transFigure, color=HEADER_BG, zorder=0)
    )
    fig.patches.append(
        Rectangle((0.0, 0.9), 1.0, 0.006, transform=fig.transFigure, color=HEADER_ACCENT, zorder=1)
    )

    fig.text(0.06, 0.965, TITLE, fontsize=15, fontweight="bold", color="white", va="top")
    fig.text(0.06, 0.936, SUBTITLE, fontsize=10.2, color="#D1D5DB", va="top")
    fig.text(0.94, 0.936, f"Page {page_number}", fontsize=9.8, color="#D1D5DB", va="top", ha="right")
    fig.text(0.06, 0.885, page_title, fontsize=13, fontweight="bold", color=TEXT_DARK, va="top")


def draw_footer(fig) -> None:
    fig.text(
        0.06,
        0.03,
        "Prepared for Jubilant Data Science & GenAI Assessment | Tools: Python, pandas, scikit-learn, matplotlib",
        fontsize=8.8,
        color=TEXT_MUTED,
    )


def add_page(pdf: PdfPages, heading: str, sections: list[tuple[str, list[str]]], page_number: int) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.patch.set_facecolor(PAGE_BG)

    draw_header(fig, heading, page_number)

    y = 0.84
    body_font = 10
    line_h = 0.021
    wrap_width = 84

    for section_title, bullets in sections:
        estimated_lines = 1
        for bullet in bullets:
            estimated_lines += max(1, (len(bullet) // (wrap_width - 4)) + 1)
        card_height = 0.04 + estimated_lines * 0.021

        fig.patches.append(
            Rectangle(
                (0.06, y - card_height + 0.006),
                0.88,
                card_height,
                transform=fig.transFigure,
                facecolor=CARD_BG,
                edgecolor=CARD_BORDER,
                linewidth=1.0,
            )
        )

        fig.text(0.08, y - 0.008, section_title, fontsize=11.2, fontweight="bold", color=TEXT_DARK, va="top")
        y -= 0.034

        for bullet in bullets:
            wrapped = textwrap.wrap(f"• {bullet}", width=wrap_width) or ["•"]
            for line in wrapped:
                fig.text(0.09, y, line, fontsize=body_font, color=TEXT_DARK, va="top")
                y -= line_h
            y -= 0.0015

        y -= 0.014

    draw_footer(fig)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    page1_sections = [
        (
            "1) Objective",
            [
                "Develop an end-to-end time-series workflow for daily store sales forecasting using free, CPU-friendly tools.",
                "Deliver practical outputs for business teams: clean analysis, model performance, and a 7-day action plan.",
            ],
        ),
        (
            "2) Data & Preprocessing",
            [
                "Dataset includes Date, Sales, Promotion flag, and DayOfWeek.",
                "Date is parsed as datetime and data is sorted chronologically before analysis.",
                "Data quality checks confirm structure, data types, and missing-value status.",
                "A strict time-based split is applied: train on history, test on the most recent 30 days.",
            ],
        ),
        (
            "3) Feature Engineering (No Leakage)",
            [
                "Lag features: lag_1 and lag_7 to capture short-term memory and weekly seasonality.",
                "Rolling statistics: 7-day and 14-day rolling means to capture local trend patterns.",
                "Calendar features: month, ISO week, day_of_month, and DayOfWeek.",
                "All rolling calculations use shift(1) and train/test-aware handling to prevent future-data leakage.",
            ],
        ),
        (
            "4) Modeling Strategy",
            [
                "Primary model: GradientBoostingRegressor wrapped in a scikit-learn Pipeline with StandardScaler.",
                "Evaluation metrics: MAE, RMSE, and MAPE (with 100 - MAPE shown as an accuracy proxy).",
                "Forecasting method: recursive 7-day rollout where each predicted day feeds the next step.",
                "Future Promotion value is auto-estimated using LogisticRegression to keep forecasting fully automatic.",
            ],
        ),
    ]

    page2_sections = [
        (
            "5) Key Insights",
            [
                "Promotions create measurable sales lift versus non-promotion days.",
                "Weekday seasonality is stable; best and worst days are predictable and actionable.",
                "Recent sales history (lag-based features) is among the strongest drivers of prediction quality.",
                "Model quality is suitable for short-horizon planning when MAPE remains within an acceptable range.",
            ],
        ),
        (
            "6) Business Actions",
            [
                "Use the 7-day forecast to align inventory: increase stock on predicted high-demand days.",
                "Align staffing with expected peaks and troughs to reduce service bottlenecks.",
                "Schedule promotions on weaker-demand days to smooth weekly revenue.",
                "Re-train periodically (e.g., monthly) to maintain performance as demand behavior changes.",
            ],
        ),
        (
            "7) Practical Limitations",
            [
                "No forecasting model can guarantee 100% accuracy on future unseen days.",
                "Unexpected events (holiday shifts, local disruptions, campaign changes) can cause forecast drift.",
                "The best confidence check is walk-forward backtesting on multiple 7-day windows.",
            ],
        ),
        (
            "8) Conclusion",
            [
                "The solution provides a complete, interpretable, and deployment-friendly forecasting pipeline.",
                "It balances modeling rigor (time-aware validation, leakage control) with business usability (clear insights + actionable 7-day plan).",
            ],
        ),
    ]

    with PdfPages(OUTPUT_PATH) as pdf:
        add_page(pdf, "Approach Summary", page1_sections, page_number=1)
        add_page(pdf, "Key Insights & Recommendations", page2_sections, page_number=2)

    print(f"Created: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
