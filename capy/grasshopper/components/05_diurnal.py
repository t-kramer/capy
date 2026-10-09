"""capy | 05 | Diurnal Temperature Flux
Paste this script into a GHPython component.

Computes three diurnal flux metrics:
  Amplitude      — daily peak-to-trough temperature swing per point [°C]
  DriftRate      — hour-to-hour |ΔT| per point [°C/hr]
  PhaseAlignment — Pearson r correlation of indoor vs outdoor daily cycle

Inputs:
    env            : capy ThermalEnvironment (from component 00 or 01)
    Schedule       : capy OccupancySchedule  (optional)
    TargetMin      : float — lower bound of target amplitude band [°C] (default 2)
    TargetMax      : float — upper bound of target amplitude band [°C] (default 4)
    AbruptThreshold: float — drift rate above which change is abrupt [°C/hr] (default 2)
    EPW            : Ladybug EPW or path string  (required for phase alignment)

Outputs:
    mean_amplitude     : float — mean daily ΔT across all points and days [°C]
    in_target_fraction : float — fraction of point-days in [TargetMin, TargetMax] [0–1]
    median_drift_rate  : float — median hourly drift rate across all points [°C/hr]
    abrupt_fraction    : float — fraction of hours with abrupt temperature change [0–1]
    phase_alignment    : float — mean Pearson r across all points [-1, +1]
                         (+1 = aligned with outdoor, -1 = inverted)
    amplitude_per_point: list  — median annual daily amplitude per point [°C]
"""

from capy.metrics.diurnal import diurnal_amplitude, temperature_drift_rate

if env is None:
    raise ValueError("Connect a capy ThermalEnvironment to env.")

_t_min = float(TargetMin) if TargetMin is not None else 2.0
_t_max = float(TargetMax) if TargetMax is not None else 4.0
_abrupt = float(AbruptThreshold) if AbruptThreshold is not None else 2.0

amp = diurnal_amplitude(env, schedule=Schedule, target_min=_t_min, target_max=_t_max)
drift = temperature_drift_rate(env, abrupt_threshold=_abrupt)

mean_amplitude = amp.mean_amplitude
in_target_fraction = amp.fraction_in_target
median_drift_rate = float(drift.median_rate.mean())
abrupt_fraction = drift.abrupt_fraction
amplitude_per_point = amp.median_amplitude.tolist()

# Phase alignment — requires EPW
phase_alignment = None
if EPW is not None:
    from capy.metrics.diurnal import diurnal_phase_alignment
    _epw_path = str(EPW._file_path) if hasattr(EPW, "_file_path") else str(EPW)
    pa = diurnal_phase_alignment(env, epw_path=_epw_path)
    phase_alignment = pa.mean_score

print(
    f"capy | Amplitude={mean_amplitude:.2f}°C  "
    f"InTarget={in_target_fraction*100:.0f}%  "
    f"DriftMedian={median_drift_rate:.3f}°C/hr  "
    f"Abrupt={abrupt_fraction*100:.1f}%"
    + (f"  PhaseR={phase_alignment:.3f}" if phase_alignment is not None else "")
)
