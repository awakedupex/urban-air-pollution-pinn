"""Write the deck's slide files and index (project/deck.json, project/slides/*.html)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SL = ROOT / "project" / "slides"
SL.mkdir(parents=True, exist_ok=True)

IMG = {
    "map": "/_blob/8365b1df99a57db6cc67296fd9d40f31",
    "fields": "/_blob/1cf335dcd008766fa9983d5ccde41e3d",
    "twincomp": "/_blob/c1054085f376027f3024ce8d52dbb599",
    "realcomp": "/_blob/140815a641471968eadca9d662c2a310",
    "realres": "/_blob/29dffd7af6b9f2cd93bf55d4373d2cd0",
    "ident": "/_blob/ecf3ffe305dcf6457d0f68fd154db9ec",
    "wind": "/_blob/de93b3a2ed559993d8206685f8370ed2",
    "wp": "/_blob/749b6275693351a82f0043bdac9d6962",
    "sens": "/_blob/a32d277570effd2c2bf8fd0c70c2b479",
    "series": "/_blob/483cc972b97d59c1d07b5896e05894b4",
}

DARK, LIGHT, ALT = "#14213D", "#FBFBF8", "#F1F0EA"
BODY, MUTED, BLUE, ORANGE, CARD, RULE = "#4A5568", "#5F6672", "#1F5FAE", "#B4471B", "#FDFDFB", "#E1E0D9"
SERIF = "'Source Serif 4', Georgia, serif"
SANS = "'IBM Plex Sans', Arial, sans-serif"
N = 15


def section(sid, inner, notes, presenter, bg=LIGHT, num=None, extra=""):
    footer = ""
    if num is not None:
        footer = (f'<div style="position:absolute;left:128px;bottom:64px;width:1664px;display:flex;'
                  f'flex-direction:row;justify-content:space-between">'
                  f'<p style="font-size:24px;color:{MUTED}">{presenter}</p>'
                  f'<p style="font-size:24px;color:{MUTED}">{num} / {N}</p></div>')
    return (f'<section id="{sid}" data-transition="fade" style="background:{bg};color:{DARK};font-family:{SANS};'
            f'padding:128px 128px 160px;display:flex;flex-direction:column;gap:40px{extra}">\n'
            f'{inner}\n{footer}\n<aside>{notes}</aside>\n</section>\n')


def title(text, eyebrow):
    return (f'<div style="display:flex;flex-direction:column;gap:12px">'
            f'<p style="font-size:24px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:{BLUE}">{eyebrow}</p>'
            f'<h2 style="font-family:{SERIF};font-size:64px;font-weight:600;line-height:1.1;color:{DARK}">{text}</h2></div>')


def card(head, body, flex="flex:1", tag=None, size=28):
    t = f'<p style="font-size:24px;font-weight:600;color:{BLUE};text-transform:uppercase;letter-spacing:1px">{tag}</p>' if tag else ""
    return (f'<div style="{flex};display:flex;flex-direction:column;gap:12px;background:{CARD};padding:36px;'
            f'border:1px solid {RULE};border-radius:16px">{t}'
            f'<h3 style="font-family:{SERIF};font-size:40px;font-weight:600;line-height:1.15;color:{DARK}">{head}</h3>'
            f'<p style="font-size:{size}px;line-height:1.4;color:{BODY}">{body}</p></div>')


def table(header, rows, widths, size=28):
    th = "".join(f'<th style="width:{w}%;text-align:left">{h}</th>' for h, w in zip(header, widths))
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<table style="font-family:{SANS};font-size:{size}px;color:{DARK}"><tr style="background:{ALT}">{th}</tr>{trs}</table>')


slides = {}

# 1 ------------------------------------------------------------------ cover
slides["cover"] = (
    f'<section id="cover" data-transition="fade" style="background:{DARK};color:{ALT};font-family:{SANS};'
    f'padding:128px;display:flex;flex-direction:column;justify-content:end;gap:32px">\n'
    f'<p style="font-size:24px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#86B6EF">Mathematical Modelling · Group project</p>'
    f'<h1 style="font-family:{SERIF};font-size:96px;font-weight:600;line-height:1.1;color:{ALT}">Physics-informed neural networks for urban air pollution</h1>'
    f'<p style="font-size:40px;line-height:1.3;color:#BFD0E6">Analysis, simulation and prediction of PM2.5 over Delhi</p>'
    f'<p style="font-size:28px;color:#BFD0E6">[Member 1] · [Member 2] · [Member 3] · [Member 4] · [Member 5] · November 2026</p>\n'
    f'<aside>Presenter 1 (about 15 s). Introduce the group and the question in one sentence: can a neural network that is forced to obey the physics of pollutant transport estimate PM2.5 better at places where there is no monitor? Mention that everything uses real public data and is reproducible from the repository.</aside>\n</section>\n')

# 2 ------------------------------------------------------------------ question
slides["question"] = section("question",
    title("Can physics fill the gaps between monitors?", "The problem") +
    '<div style="display:flex;flex-direction:row;gap:32px">' +
    card("258 µg/m³", "Mean PM2.5 over 45 Delhi monitors, 10–20 Nov 2025: about 17× the WHO 24-hour guideline.", tag="Exposure") +
    card("45 points", "Monitors for about 2,700 km²: the field is unobserved almost everywhere.", tag="Sparsity") +
    card("≈ 9 hours", "For air to cross the city at the typical wind speed: hourly data can see transport between stations.", tag="Opportunity") +
    '</div>' +
    f'<p style="font-size:32px;line-height:1.4;color:{BODY}">We compare a network that only fits the monitors with <b>the same network constrained by the advection–diffusion–reaction equation</b>, always scored on monitors it never saw.</p>',
    "Presenter 1 (about 45 s). Health exposure varies across the city, but we only measure at a few dozen points. Purely statistical interpolation has no physics: nothing stops it producing a field the wind could not have made. Transport models need emission inventories we do not have. A PINN sits in between. The key experimental control: the data-only network is identical to the PINN except that the physics weight is zero, so any difference is caused by the physics. Numbers: winter mean 258 µg/m³, WHO 2021 24-hour guideline 15 µg/m³; domain 49.8 by 53.4 km; mean wind about 1.4 m/s over about 50 km gives roughly 9 to 10 hours.",
    "Presenter 1 · The question", num=2)

# 3 ------------------------------------------------------------------ data
chip = lambda label, text: (f'<div style="display:flex;flex-direction:column;gap:6px;background:{CARD};padding:24px 28px;'
                            f'border:1px solid {RULE};border-radius:12px">'
                            f'<p style="font-size:24px;font-weight:600;color:{BLUE};text-transform:uppercase;letter-spacing:1px">{label}</p>'
                            f'<p style="font-size:26px;line-height:1.35;color:{DARK}">{text}</p></div>')
slides["data"] = section("data",
    title("Real data: 45 regulatory monitors, hourly", "Data") +
    '<div style="display:flex;flex-direction:row;gap:40px;align-items:start">' +
    f'<img src="{IMG["map"]}" alt="Maps of the 45 winter and 42 spring monitors, coloured by mean PM2.5 and marked by train, validation or test role" style="width:1060px;height:505px;object-fit:contain;background:#FFFFFE;border-radius:12px">' +
    '<div style="flex:1;display:flex;flex-direction:column;gap:16px">' +
    chip("Observed", "PM2.5 from CPCB, DPCC, IMD, IITM monitors via OpenAQ") +
    chip("Estimated", "Wind and mixing height from ERA5 via Open-Meteo") +
    chip("Calibrated", "Diffusivity and emissions, learned by the PINN") +
    chip("Synthetic", "Twin truth from our own solver") +
    '</div></div>' +
    f'<p style="font-size:24px;line-height:1.4;color:{BODY}">Cleaning: sentinels, units, IST→UTC, 15-min→hourly, stuck sensors, ≥75 % coverage. Winter (10–20 Nov 2025) 9,828 and spring (1–11 Mar 2025) 9,693 hourly values.</p>',
    "Presenter 1 (about 60 s). Data come from the OpenAQ archive, which republishes the official CPCB and DPCC regulatory monitors; low-cost sensors were excluded. Point out the honest change of plan: we planned November 2024, but the archive has no regulatory files for that month, so we moved to the same dates in 2025. Stress that every quantity is labelled observed, estimated, calibrated or synthetic, and that we never show interpolated values as measurements. Stations are split, not hours: about 30 train, 7 validation, 7–8 test, with co-located stations kept together so a test station never has a twin in training.",
    "Presenter 1 · Data", num=3)

# 4 ------------------------------------------------------------------ equation
slides["equation"] = section("equation",
    title("One conservation law, four processes", "Model") +
    f'<div style="background:{ALT};border-radius:16px;padding:32px 40px"><p style="font-family:{SERIF};font-size:44px;text-align:center;color:{DARK}">∂C/∂t + u ∂C/∂x + v ∂C/∂y = D ∇²C − λC + q(x,y) g(τ) / H(t)</p></div>' +
    '<div style="display:flex;flex-direction:row;gap:24px">' +
    card("Advection", "u(t), v(t): ERA5 10 m wind, uniform over the city.", tag="Estimated", size=26) +
    card("Diffusion", "D: eddy diffusivity, learned within 10–2000 m²/s.", tag="Calibrated", size=26) +
    card("Removal", "λ = 5×10⁻⁶ s⁻¹: an e-folding time of 2.3 days.", tag="Fixed", size=26) +
    card("Emission", "q(x,y) ≥ 0 and a 24-hour profile g, mixed into height H(t).", tag="Learned + ERA5", size=26) +
    '</div>',
    "Presenter 2 (about 40 s). This is mass conservation in a well-mixed layer: change in time plus transport by the wind equals turbulent spreading minus removal plus emission. Explain each term and where its value comes from. Units: C in µg/m³, q in µg/m²/s, so dividing by the mixing height H gives µg/m³/s like every other term. H is floored at 50 m because ERA5 reports 10–25 m on calm nights. Likely question: why not a Gaussian plume? It is the steady-state point-source solution of this same equation and needs known stacks, which we do not have.",
    "Presenter 2 · Model", num=4)

# 5 ------------------------------------------------------------------ assumptions
slides["assumptions"] = section("assumptions",
    title("Seven assumptions, each with a cost", "Assumptions") +
    table(["", "Assumption", "What it costs"],
          [["A1", "Well-mixed layer of depth H(t): 3D becomes 2D", "No vertical structure"],
           ["A2", "Spatially uniform ERA5 wind", "No street-scale flow"],
           ["A3", "Constant isotropic diffusivity D", "D cannot be identified"],
           ["A4", "Linear removal, no chemistry", "Secondary aerosol missing"],
           ["A5", "Separable source q(x,y)·g(τ)/H(t)", "Keeps the physics loss meaningful"],
           ["A6", "Unknown inflow, weak outflow condition", "Regional transport unmodelled"],
           ["A7", "No entrainment term", "Morning dilution unmodelled"]],
          [8, 54, 38], size=28),
    "Presenter 2 (about 40 s). Do not read the table; pick A2, A5 and A6. A5 is the one examiners like: if the source were an arbitrary function of space and time, any field would satisfy the equation exactly by defining the source as the residual, and the physics loss would say nothing. Restricting it to a spatial map times a daily cycle divided by mixing height forces the space–time structure to be explained by transport. A6 and A7 come back at the end as the main reasons the real-data result is limited.",
    "Presenter 2 · Assumptions", num=5)

# 6 ------------------------------------------------------------------ scaling
slides["scaling"] = section("scaling",
    title("Scaling says what the data can resolve", "Dimensional analysis") +
    '<div style="display:flex;flex-direction:row;gap:32px">' +
    card("Pe ≈ 75–750", "Péclet number ūℓ/D: advection dominates diffusion, so expect D to be poorly identified.", tag="Transport") +
    card("Da ≈ 0.17", "Damköhler number λℓ/ū: removal is slow over one crossing of the city.", tag="Removal") +
    card("≈ 7–11 h", "Advection time ℓ/ū: hourly data can see pollution move between stations.", tag="Time scale") +
    '</div>' +
    f'<p style="font-size:30px;line-height:1.4;color:{BODY}">Dimensionless form: U = uT/2L, K = DT/2L², Λ = λT/2. On 80-hour windows all are O(10⁻²)–O(10), so the residual is well scaled.</p>',
    "Presenter 2 (about 40 s). We rescale space by the half-width L = 26.7 km, time by the window length and concentration by the training mean, so every network input is in [−1, 1] and every coefficient is of order one; without this the residual terms differ by many orders of magnitude and training fails. With domain size about 50 km and mean wind 1.4–2.2 m/s: Pe between 75 and 750 for D between 1000 and 100 m²/s, Da 0.12–0.19, crossing time 7–11 hours. The prediction from Pe — that D will be hard to learn — is tested on slide 9.",
    "Presenter 2 · Scaling", num=6)

# 7 ------------------------------------------------------------------ analysis
slides["analysis"] = section("analysis",
    title("The continuous model is stable and well posed", "Mathematical analysis") +
    '<div style="display:flex;flex-direction:row;gap:28px">' +
    card("Fourier stability", "C ∝ exp(ik·x + σt) gives Re σ = −(D|k|² + λ) &lt; 0: every mode decays; advection only shifts the phase.", tag="Stability", size=26) +
    card("Equilibrium", "Uniform source: C(t) → C* = S₀/λ, globally stable. With g(τ)/H(t) the solution settles into a daily cycle.", tag="Steady state", size=26) +
    card("Positivity, mass", "Maximum principle: C ≥ 0 if inputs are. Mass changes only through sources, removal and boundary flux.", tag="Structure", size=26) +
    '</div>' +
    f'<div style="background:{ALT};border-radius:16px;padding:28px 36px"><p style="font-size:30px;line-height:1.4;color:{DARK}"><b>Identifiability.</b> An unrestricted source could absorb any residual, so S is restricted to q(x,y)·g(τ)/H(t), and D is weakly identifiable because Pe ≫ 1.</p></div>',
    "Presenter 3 (about 45 s). Stability: substitute a Fourier mode; the growth rate has negative real part for all wavenumbers, so short waves are damped by diffusion and all waves by removal; this is why we can spin the synthetic truth up for 24 hours and do not need an initial field for the PINN. Equilibrium: the uniform ODE dC/dt = S₀ − λC has the stable fixed point S₀/λ; the periodic forcing through the mixing height gives the night-time peaks we see in the data. Integrating over the domain gives the mass balance used as a solver test.",
    "Presenter 3 · Analysis", num=7)

# 8 ------------------------------------------------------------------ verification
slides["verification"] = section("verification",
    title("Every component was tested against an exact answer", "Verification") +
    table(["Check", "Reference", "Result"],
          [["Autograd Laplacian", "Analytic derivatives", "Error 4 × 10⁻¹⁵"],
           ["PDE residual", "Exact Gaussian puff", "3 × 10⁻¹⁶"],
           ["Reference solver", "Puff, Δx = 750 → 188 m", "1.4 × 10⁻² → 1.9 × 10⁻⁴"],
           ["Decay and equilibrium", "Exact ODE solution", "≤ 10⁻⁵ relative"],
           ["Mass conservation", "No sources or removal", "Within 10⁻³"]],
          [32, 36, 32], size=28) +
    f'<p style="font-size:30px;line-height:1.4;color:{BODY}">50 pytest tests cover cleaning, leakage, physics, the PINN and metrics. Solver: 3rd-order upwind advection, central diffusion, SSP-RK3; von Neumann limits Courant ≤ 0.5, diffusion number ≤ 0.25; observed order 2.6–3.7.</p>',
    "Presenter 3 (about 40 s). The Gaussian puff is the exact solution of our equation for an instantaneous release in uniform wind; plugging it into the residual must give zero, and it does to machine precision, which proves the derivatives and coefficients in the loss are exactly the PDE we wrote. The finite-difference solver converges at roughly third order under grid refinement, so it is accurate enough to be the truth of the synthetic twin. Development was test-first: each test was written and seen to fail before the code.",
    "Presenter 3 · Verification", num=8)

# 9 ------------------------------------------------------------------ sensitivity
slides["sensitivity"] = section("sensitivity",
    title("Concentrations barely depend on D, so D cannot be learned", "Sensitivity") +
    '<div style="display:flex;flex-direction:row;gap:24px;align-items:center">' +
    f'<img src="{IMG["sens"]}" alt="Forward sensitivity of station concentrations to removal rate, diffusivity and mixing-height floor" style="width:820px;height:300px;object-fit:contain;background:#FFFFFE;border-radius:12px">' +
    f'<img src="{IMG["ident"]}" alt="Learned diffusivity far below the true value in all splits, and learned versus true diurnal emission profile" style="width:820px;height:330px;object-fit:contain;background:#FFFFFE;border-radius:12px">' +
    '</div>' +
    '<div style="display:flex;flex-direction:row;gap:28px">' +
    card("D −0.03 · λ −0.15 · H<sub></sub>min −0.35".replace("<sub></sub>", "_"), "Elasticity of station-mean PM2.5: diffusion hardly matters; the mixing-height floor matters most.", tag="Forward model", size=26) +
    card("300 → 58–60 m²/s", "D started at its true value and drifted away in every split. The daily emission profile is recovered well.", tag="Synthetic twin", size=26) +
    '</div>',
    "Presenter 3 (about 40 s). Left: we re-ran the solver changing one parameter at a time; the elasticity d ln C / d ln p is −0.03 for D, so quadrupling D barely changes what the monitors would see. Right: on the synthetic twin we know the true D is 300 and we even started the network there; it still drifted to about 60 in every split. That is the identifiability prediction from the Péclet number confirmed. The mixing-height floor, a modelling choice, has the largest effect, which we list as a limitation.",
    "Presenter 3 · Sensitivity", num=9)

# 10 ------------------------------------------------------------------ pinn
box = lambda t: (f'<div style="flex:1;display:flex;flex-direction:column;justify-content:center;background:{CARD};padding:24px;'
                 f'border:1px solid {RULE};border-radius:12px"><p style="font-size:28px;line-height:1.3;text-align:center;color:{DARK}">{t}</p></div>')
arrow = f'<x-shape kind="arrow-right" style="width:48px;height:24px;background:{BLUE}"></x-shape>'
slides["pinn"] = section("pinn",
    title("A network trained to fit monitors and obey the PDE", "The PINN") +
    '<div style="display:flex;flex-direction:row;gap:16px;align-items:center">' +
    box("(x′, y′, t′)<br>in [−1, 1]³") + arrow + box("MLP 5 × 64<br>tanh") + arrow + box("softplus<br>Ĉ ≥ 0") + arrow +
    box("autograd<br>∂t ∂x ∂y ∇²") + arrow + box("residual r<br>with u, v, H, q, g, D") + '</div>' +
    f'<div style="background:{ALT};border-radius:16px;padding:28px 36px"><p style="font-family:{SERIF};font-size:40px;text-align:center;color:{DARK}">L = L<sub></sub>data + w<sub></sub>p · L<sub></sub>PDE + 0.1 · L<sub></sub>outflow + 10⁻³ · L<sub></sub>source</p></div>'.replace("<sub></sub>", "_") +
    '<div style="display:flex;flex-direction:row;gap:24px">' +
    card("tanh, not ReLU", "ReLU has zero second derivative, so ∇²Ĉ would vanish.", size=26) +
    card("Weather not an input", "So ∂Ĉ/∂t is a true partial derivative; wind enters the PDE.", size=26) +
    card("Fair control", "Data-only NN: the same model and optimiser with w_p = 0.", size=26) +
    '</div>',
    "Presenter 4 (about 45 s). The network maps scaled space and time to a concentration; softplus guarantees it is non-negative. Automatic differentiation gives exact derivatives, which form the residual using the ERA5 wind and mixing height, the learned emission map and daily profile, and the bounded diffusivity. The loss adds the misfit at training monitors, the mean squared residual at 4096 random collocation points resampled every 250 steps, a weak outflow condition and a smoothness penalty on the source. Adam for 6000 iterations; the validation monitors only choose the best iterate. Real 10-day windows are trained as three 80-hour chunks.",
    "Presenter 4 · PINN", num=10)

# 11 ------------------------------------------------------------------ weight
slides["weight"] = section("weight",
    title("The physics weight decides whether training works", "Loss balancing") +
    '<div style="display:flex;flex-direction:row;gap:40px;align-items:center">' +
    f'<img src="{IMG["wp"]}" alt="Held-out and whole-field RMSE against the physics weight: best at 0.003 to 0.01, far worse at 1" style="width:1000px;height:420px;object-fit:contain;background:#FFFFFE;border-radius:12px">' +
    '<div style="flex:1;display:flex;flex-direction:column;gap:24px">' +
    f'<p style="font-size:30px;line-height:1.4;color:{BODY}">At w_p = 1 the PDE term starts about 20× larger than the data term (12 vs 0.55) and swamps the gradient: the network under-fits.</p>' +
    f'<p style="font-size:30px;line-height:1.4;color:{BODY}">w_p = 0.003–0.01 is best. We <b>fixed w_p = 0.01 before any real-data run</b>.</p>' +
    f'<p style="font-size:30px;line-height:1.4;color:{BODY}">Adaptive gradient balancing (Wang et al., 2021) adapted too slowly here.</p>' +
    '</div></div>',
    "Presenter 4 (about 35 s). This is the known PINN failure mode of gradient imbalance between loss terms. Our first run used w_p = 1 and the PINN was worse than everything; the diagnosis was the loss magnitudes, not a bug. We swept the weight on the synthetic twin only, chose 0.01, and froze it before touching real data, so the real test monitors never influenced tuning. On real data a sensitivity sweep over 0.003–0.03 changes the RMSE by only a few µg/m³.",
    "Presenter 4 · Loss balancing", num=11)

# 12 ------------------------------------------------------------------ twin results
slides["twin"] = section("twin",
    title("Synthetic twin: physics helps most away from monitors", "Results 1 · synthetic") +
    '<div style="display:flex;flex-direction:row;gap:40px;align-items:start">' +
    '<div style="width:600px;display:flex;flex-direction:column;gap:24px">' +
    table(["Model", "Stations", "Field"],
          [["IDW", "43.8", "—"], ["Random forest", "43.8", "—"], ["Data-only NN", "29.9", "54.5"], ["<b>PINN</b>", "<b>22.1</b>", "<b>31.3</b>"]],
          [50, 25, 25], size=28) +
    f'<p style="font-size:26px;line-height:1.4;color:{BODY}">RMSE in µg/m³, mean of 4 station splits. Truth: our solver with real ERA5 forcing at the 45 real monitor sites. PINN lower field error in 72 of 73 hours.</p>' +
    '</div>' +
    f'<img src="{IMG["fields"]}" alt="Truth and reconstructed concentration maps, error maps, and field error at every hour for the data-only network and the PINN" style="width:1024px;height:656px;object-fit:contain;background:#FFFFFE;border-radius:12px">' +
    '</div>',
    "Presenter 4 (about 45 s). The twin is the controlled experiment: we know the true field everywhere, so we can score the whole map, not just stations. Against the identical network without physics, the PINN cuts held-out station error by 26 percent and whole-field error by 43 percent, and its spread across splits is a third. It beats the data-only network on the field in all four splits, and at the stations in three of four. Its fields satisfy the PDE about four times better even after we fit an emission model to the data-only field. Be explicit that the twin shares the PINN's structure, so it is a best case.",
    "Presenter 4 · Synthetic twin", num=12)

# 13 ------------------------------------------------------------------ real results
slides["real"] = section("real",
    title("Real Delhi data: physics stabilises, but does not beat interpolation", "Results 2 · observed") +
    f'<img src="{IMG["realcomp"]}" alt="RMSE at held-out monitors for five models in winter and spring over three splits" style="width:1250px;height:450px;object-fit:contain;background:#FFFFFE;border-radius:12px;align-self:center">' +
    '<div style="display:flex;flex-direction:row;gap:24px">' +
    card("6 of 6", "Runs where the PINN beats the same network without physics (winter 83 vs 111 µg/m³).", size=26) +
    card("3.7–8×", "Smaller PDE residual: the PINN’s fields are consistent with transport.", size=26) +
    card("1 of 6", "Runs where the PINN is the best model: it ties IDW in winter; the station mean wins in spring.", size=26) +
    '</div>',
    "Presenter 5 (about 45 s). Three splits per window, w_p fixed in advance. Winter RMSE: PINN 83.3, IDW 84.4, station mean 84.8, random forest 94.2, data-only network 111.4 with a huge spread (65 to 182). Spring: station mean 29.7, PINN 33.1, IDW 33.2. So the physics clearly regularises the network, and produces physically consistent fields, but on real data it does not beat simple interpolation; the differences between PINN, IDW and station mean are far smaller than the split-to-split spread. Say it plainly: this is a negative result, and we report it as one.",
    "Presenter 5 · Real data", num=13)

# 14 ------------------------------------------------------------------ why
slides["why"] = section("why",
    title("Why the physics cannot do more here", "Interpretation") +
    '<div style="display:flex;flex-direction:row;gap:40px;align-items:start">' +
    '<div style="width:760px;display:grid;grid-template-columns:1fr 1fr;gap:20px">' +
    card("r = 0.7", "Median inter-station correlation: a strong city-wide common signal.", size=26) +
    card("r = 0.50", "Monitor anemometers vs ERA5 wind (0.55 vs 1.38 m/s).", size=26) +
    card("Missing", "Regional inflow, entrainment and secondary aerosol.", size=26) +
    card("37–266", "Learned D (m²/s) across real chunks: unidentified.", size=26) +
    '</div>' +
    f'<img src="{IMG["wind"]}" alt="Hourly wind speed at monitors against the ERA5 wind used in the model, winter window" style="width:864px;height:350px;object-fit:contain;background:#FFFFFE;border-radius:12px">' +
    '</div>' +
    f'<p style="font-size:32px;line-height:1.4;color:{DARK}">The twin shows the method works when the model is right, so the gap is <b>the model, not the PINN</b>.</p>',
    "Presenter 5 (about 40 s). Four pieces of evidence. One: stations move together, so a uniform estimate already explains most of an unseen station. Two: the monitors' own wind sensors agree with ERA5 only moderately; they read low because they sit in the urban canopy, but the uncertainty in the transporting wind is real (the March files have no wind data, so this is winter only). Three: the biggest missing processes act through the boundaries and the vertical, exactly assumptions A4, A6 and A7. Four: D is not identified, as predicted. The contrast with the twin isolates the cause.",
    "Presenter 5 · Interpretation", num=14)

# 15 ------------------------------------------------------------------ conclusion
slides["conclusion"] = (
    f'<section id="conclusion" data-transition="fade" style="background:{DARK};color:{ALT};font-family:{SANS};'
    f'padding:128px;display:flex;flex-direction:column;justify-content:space-between;gap:40px">\n'
    f'<div style="display:flex;flex-direction:column;gap:16px"><p style="font-size:24px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#86B6EF">Conclusion</p>'
    f'<h2 style="font-family:{SERIF};font-size:72px;font-weight:600;line-height:1.1;color:{ALT}">A physics-informed model is only as useful as the physics it encodes.</h2></div>'
    f'<div style="display:flex;flex-direction:column;gap:20px">'
    f'<p style="font-size:32px;line-height:1.4;color:#D5DEEA"><b>Synthetic twin:</b> 26 % lower station error and 43 % lower field error than the same network without physics.</p>'
    f'<p style="font-size:32px;line-height:1.4;color:#D5DEEA"><b>Real Delhi data:</b> more stable and physically consistent, but not more accurate than interpolation.</p>'
    f'<p style="font-size:32px;line-height:1.4;color:#D5DEEA"><b>Next:</b> a regional boundary condition and a mixing-height entrainment term.</p></div>'
    f'<p style="font-size:32px;color:#86B6EF">Thank you. Questions welcome.</p>\n'
    f'<aside>Presenter 5 (about 30 s). Close on the single message: the PINN does exactly what it is designed to do when its physics is right, and the honest real-data result tells us which physics to add next. Hand over to questions; the group has prepared answers in presentation/viva_questions.md.</aside>\n</section>\n')

for sid, html in slides.items():
    (SL / f"{sid}.html").write_text(html)

order = ["cover", "question", "data", "equation", "assumptions", "scaling", "analysis", "verification",
         "sensitivity", "pinn", "weight", "twin", "real", "why", "conclusion"]
assert order == list(slides), "order mismatch"
deck = {
    "v": 4, "createdOnFiles": {"v": 1, "at": "2026-11-06T13:40:00Z"}, "lists": "css",
    "title": "PINNs for Delhi PM2.5", "order": order, "cover": "cover",
    "sections": {
        "s1": {"description": "Presenter 1: the question and the real data.", "start": "cover"},
        "s2": {"description": "Presenter 2: the model, its assumptions and its scaling.", "start": "equation"},
        "s3": {"description": "Presenter 3: analysis, verification and sensitivity.", "start": "analysis"},
        "s4": {"description": "Presenter 4: the PINN, loss balancing and the synthetic twin.", "start": "pinn"},
        "s5": {"description": "Presenter 5: real-data results, interpretation and conclusion.", "start": "real"},
    },
    "faces": {
        "source-serif-4": {"family": "Source Serif 4", "href": "https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400..700&display=swap"},
        "ibm-plex-sans": {"family": "IBM Plex Sans", "href": "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap"},
    },
    "designSystems": [],
}
(ROOT / "project" / "deck.json").write_text(json.dumps(deck, indent=1, ensure_ascii=False))
print("wrote", len(slides), "slides")
