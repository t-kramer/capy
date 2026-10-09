"""capy | 02 | Comfort Model
Paste this script into a GHPython component.

Computes comfort classification for all sensor points over all hours.

Inputs:
    env         : capy ThermalEnvironment (from component 00 or 01)
    Model       : str — 'empirical' (default), 'pmv', or 'adaptive'
    LowerBound  : float — lower comfort bound in °C (empirical only, default 21)
    UpperBound  : float — upper comfort bound in °C (empirical only, default 28)
    Standard    : str — 'ashrae55' or 'en15251' (adaptive only)
    Met         : float — metabolic rate in met (PMV only, default 1.1)
    Clo         : float — clothing insulation in clo (PMV only, default 0.8)

Outputs:
    comfort     : capy ComfortResult — pass to metric components
    info        : str summary string
"""

if env is None:
    raise ValueError("Connect a capy ThermalEnvironment to env.")

_model = str(Model).lower().strip() if Model is not None else "empirical"

if _model == "empirical":
    from capy.comfort.empirical import calculate_empirical
    _lo = float(LowerBound) if LowerBound is not None else 21.0
    _hi = float(UpperBound) if UpperBound is not None else 28.0
    comfort = calculate_empirical(env, lower=_lo, upper=_hi)

elif _model == "pmv":
    from capy.comfort.pmv import calculate_pmv, PMVParameters
    _met = float(Met) if Met is not None else 1.1
    _clo = float(Clo) if Clo is not None else 0.8
    params = PMVParameters(met=_met, clo=_clo)
    comfort = calculate_pmv(env, params=params)

elif _model in ("adaptive", "ashrae55", "en15251", "en16798"):
    from capy.comfort.adaptive import calculate_adaptive
    _std = str(Standard).lower() if Standard is not None else "ashrae55"
    comfort = calculate_adaptive(env, standard=_std)

else:
    raise ValueError(f"Unknown model '{_model}'. Choose: 'empirical', 'pmv', 'adaptive'")

pct = comfort.is_comfortable().mean() * 100
info = f"{comfort.model} | {pct:.1f}% comfortable"
print(f"capy | {info}")
