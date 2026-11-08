# Presentation plan: 10 minutes, 5 presenters

Deck: https://claude.ai/artifact/HZty8HWonFJ7bDuw8JxxMJ. Every slide has speaker notes, and the deck can be downloaded as PDF or PPTX.

| Presenter | Slides | Time | Owns these questions |
|---|---|---|---|
| 1 | 1 Cover · 2 The question · 3 Data | 2:00 | Data provenance, cleaning, why Delhi, the change of window, train/test split |
| 2 | 4 Equation · 5 Assumptions · 6 Scaling | 2:00 | Each term of the PDE, units, assumptions, non-dimensionalisation, Pe and Da |
| 3 | 7 Analysis · 8 Verification · 9 Sensitivity | 2:00 | Stability, equilibrium, mass balance, the solver, tests, identifiability, sensitivity |
| 4 | 10 PINN · 11 Loss balancing · 12 Synthetic twin | 2:00 | Architecture, autograd, loss terms, choice of w_p, the twin design and results |
| 5 | 13 Real results · 14 Interpretation · 15 Conclusion | 2:00 | Real-data numbers, the negative result, limitations, next steps |

## The one message

When its physics is right (the synthetic twin), the PINN generalises clearly better than the same network without physics. On real Delhi data it is still more stable and more physically consistent, but the simplified physics does not capture enough to beat simple interpolation. That tells us which physics to add next.

## Numbers every member must know

| Quantity | Value |
|---|---|
| Monitors | 45 (winter, 10–20 Nov 2025) and 42 (spring, 1–11 Mar 2025) |
| Mean PM2.5 | 258 µg/m³ winter, 67 µg/m³ spring |
| Domain | ≈ 50 × 53 km; half-width L = 26.7 km |
| Pe | 75–750 |
| Da | ≈ 0.17 |
| Crossing time | 7–11 h |
| Twin, station RMSE | Data-only NN 29.9 → PINN 22.1 (−26 %) |
| Twin, field RMSE | Data-only NN 54.5 → PINN 31.3 (−43 %) |
| Twin, D | True 300 m²/s → learned 58–60 m²/s |
| Real winter RMSE | PINN 83.3 · IDW 84.4 · station mean 84.8 · RF 94.2 · data-only NN 111.4 |
| Real spring RMSE | Station mean 29.7 · PINN 33.1 · IDW 33.2 · RF 36.8 · data-only NN 41.4 |
| PDE residual | PINN 3.7–8× smaller than the data-only NN on real data |
| Physics weight | w_p = 0.01, fixed on the twin before any real-data run |

## Rehearsal checklist

- [ ] Replace `[Member 1] … [Member 5]` on the cover slide and in the report.
- [ ] Each presenter times their block at under 2:00.
- [ ] Each presenter can answer at least the questions in their row of `viva_questions.md`.
- [ ] Everyone can state the one message and the twin vs real contrast.
