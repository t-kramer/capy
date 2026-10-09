"""Tests for capy.metrics.diurnal — amplitude, drift rate, phase alignment."""
from pathlib import Path

import numpy as np
import pytest

from capy.io.epw import load_dry_bulb as _parse_epw_dry_bulb
from capy.metrics.diurnal import (
    diurnal_amplitude,
    diurnal_phase_alignment,
    temperature_drift_rate,
)

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _write_minimal_epw(path: Path, dry_bulb_values: np.ndarray) -> None:
    """Write a bare-bones EPW file with 8 header lines then data rows."""
    with open(path, "w") as f:
        for _ in range(8):
            f.write("HEADER,header,header\n")
        for i, v in enumerate(dry_bulb_values):
            # EPW row: year,month,day,hour,minute,*,dry_bulb,...
            # We need at least 7 comma-separated fields; col 6 = dry-bulb
            f.write(f"2001,1,1,{i % 24 + 1},0,x,{v:.2f},0,0\n")


# ------------------------------------------------------------------ #
# diurnal_amplitude
# ------------------------------------------------------------------ #

class TestDiurnalAmplitude:
    def test_constant_temp_gives_zero_amplitude(self, sample_grid):
        import pandas as pd

        from capy.core.environment import ThermalEnvironment
        data = np.full((60, 8760), 22.0)
        env = ThermalEnvironment(
            grid=sample_grid,
            time_index=pd.date_range("2001-01-01", periods=8760, freq="h"),
            variables={"air_temperature": data, "mean_radiant_temperature": data,
                       "relative_humidity": np.full((60, 8760), 50.0)},
        )
        result = diurnal_amplitude(env)
        assert result.mean_amplitude == pytest.approx(0.0, abs=1e-6)
        assert result.amplitude_matrix.shape == (60, 365)

    def test_known_amplitude(self, synthetic_env):
        result = diurnal_amplitude(synthetic_env)
        assert result.amplitude_matrix.shape == (60, 365)
        # synthetic_env has ±2°C diurnal → daily range = 4°C
        assert result.mean_amplitude == pytest.approx(4.0, abs=0.5)

    def test_target_band_fractions(self, synthetic_env):
        # 4°C amplitude hits exactly the boundary of [2, 4]
        result_wide = diurnal_amplitude(synthetic_env, target_min=0.0, target_max=10.0)
        result_narrow = diurnal_amplitude(synthetic_env, target_min=10.0, target_max=20.0)
        assert result_wide.fraction_in_target == pytest.approx(1.0)
        assert result_narrow.fraction_in_target == pytest.approx(0.0)

    def test_with_schedule(self, synthetic_env, work_schedule):
        result = diurnal_amplitude(synthetic_env, schedule=work_schedule)
        # Should only include weekday-with-occupied-hours days
        assert 0.0 <= result.fraction_in_target <= 1.0

    def test_median_amplitude_shape(self, synthetic_env):
        result = diurnal_amplitude(synthetic_env)
        assert result.median_amplitude.shape == (60,)

    def test_median_amplitude_grid_is_scalar(self, synthetic_env):
        result = diurnal_amplitude(synthetic_env)
        assert isinstance(result.median_amplitude_grid, float)
        assert result.median_amplitude_grid == pytest.approx(
            float(np.median(result.median_amplitude))
        )

    def test_repr_contains_amplitude(self, synthetic_env):
        result = diurnal_amplitude(synthetic_env)
        assert "°C" in repr(result)


# ------------------------------------------------------------------ #
# temperature_drift_rate
# ------------------------------------------------------------------ #

class TestTemperatureDriftRate:
    def test_constant_gives_zero_drift(self, sample_grid):
        import pandas as pd

        from capy.core.environment import ThermalEnvironment
        data = np.full((60, 8760), 22.0)
        env = ThermalEnvironment(
            grid=sample_grid,
            time_index=pd.date_range("2001-01-01", periods=8760, freq="h"),
            variables={"air_temperature": data, "mean_radiant_temperature": data,
                       "relative_humidity": np.full((60, 8760), 50.0)},
        )
        result = temperature_drift_rate(env)
        assert result.drift_array.shape == (60, 8759)
        assert float(result.drift_array.max()) == pytest.approx(0.0, abs=1e-6)
        assert result.abrupt_fraction == pytest.approx(0.0)

    def test_drift_array_shape(self, synthetic_env):
        result = temperature_drift_rate(synthetic_env)
        assert result.drift_array.shape == (60, 8759)

    def test_abrupt_threshold_effect(self, synthetic_env):
        strict = temperature_drift_rate(synthetic_env, abrupt_threshold=0.0)
        lenient = temperature_drift_rate(synthetic_env, abrupt_threshold=100.0)
        assert strict.abrupt_fraction >= lenient.abrupt_fraction

    def test_non_negative_drift(self, synthetic_env):
        result = temperature_drift_rate(synthetic_env)
        assert float(result.drift_array.min()) >= 0.0

    def test_median_rate_shape(self, synthetic_env):
        result = temperature_drift_rate(synthetic_env)
        assert result.median_rate.shape == (60,)

    def test_repr(self, synthetic_env):
        result = temperature_drift_rate(synthetic_env)
        assert "°C/hr" in repr(result)


# ------------------------------------------------------------------ #
# EPW parsing helpers
# ------------------------------------------------------------------ #

class TestEPWParsing:
    def test_parse_epw_dry_bulb_returns_8760(self, tmp_path):
        tdb = np.full(8760, 10.0)
        epw_path = tmp_path / "test.epw"
        _write_minimal_epw(epw_path, tdb)
        arr = _parse_epw_dry_bulb(str(epw_path))
        assert arr.shape == (8760,)
        assert arr[0] == pytest.approx(10.0)


# ------------------------------------------------------------------ #
# diurnal_phase_alignment
# ------------------------------------------------------------------ #

class TestDiurnalPhaseAlignment:
    def test_no_epw_raises(self, synthetic_env):
        with pytest.raises(ValueError, match="EPW"):
            diurnal_phase_alignment(synthetic_env)

    def test_aligned_returns_positive_score(self, synthetic_env, tmp_path):
        # Outdoor cycle identical to indoor base → r ≈ 1
        hours = np.arange(8760)
        diurnal = 2 * np.sin(2 * np.pi * (hours % 24) / 24 - np.pi / 2) + 22
        epw_path = tmp_path / "aligned.epw"
        _write_minimal_epw(epw_path, diurnal)

        result = diurnal_phase_alignment(synthetic_env, epw_path=str(epw_path))
        assert result.scores.shape == (60,)
        assert result.mean_score > 0.5

    def test_inverted_returns_negative_score(self, synthetic_env, tmp_path):
        # Outdoor cycle phase-inverted
        hours = np.arange(8760)
        diurnal = -2 * np.sin(2 * np.pi * (hours % 24) / 24 - np.pi / 2) + 22
        epw_path = tmp_path / "inverted.epw"
        _write_minimal_epw(epw_path, diurnal)

        result = diurnal_phase_alignment(synthetic_env, epw_path=str(epw_path))
        assert result.mean_score < -0.5

    def test_epw_path_from_env(self, synthetic_env, tmp_path):
        hours = np.arange(8760)
        diurnal = 2 * np.sin(2 * np.pi * (hours % 24) / 24) + 22
        epw_path = tmp_path / "env_epw.epw"
        _write_minimal_epw(epw_path, diurnal)

        synthetic_env.epw_path = str(epw_path)
        result = diurnal_phase_alignment(synthetic_env)
        assert result.scores.shape == (60,)

    def test_n_days_included(self, synthetic_env, tmp_path):
        # All days at 22°C mean → within default [15, 28] range
        hours = np.arange(8760)
        tdb = 2 * np.sin(2 * np.pi * (hours % 24) / 24) + 22
        epw_path = tmp_path / "days.epw"
        _write_minimal_epw(epw_path, tdb)
        result = diurnal_phase_alignment(synthetic_env, epw_path=str(epw_path))
        assert result.n_days_included == 365

    def test_no_qualifying_days_raises(self, synthetic_env, tmp_path):
        # All days at 0°C mean → outside default [15, 28] range
        tdb = np.full(8760, 0.0)
        epw_path = tmp_path / "cold.epw"
        _write_minimal_epw(epw_path, tdb)
        with pytest.raises(ValueError, match="No days found"):
            diurnal_phase_alignment(synthetic_env, epw_path=str(epw_path))

    def test_outdoor_temp_range_filters_days(self, synthetic_env, tmp_path):
        # 182 full cold days (10°C) then 183 warm days (22°C) — clean day boundary
        tdb = np.concatenate([np.full(182 * 24, 10.0), np.full(183 * 24, 22.0)])
        epw_path = tmp_path / "mixed.epw"
        _write_minimal_epw(epw_path, tdb)
        result = diurnal_phase_alignment(
            synthetic_env, epw_path=str(epw_path), outdoor_temp_range=(15.0, 28.0)
        )
        assert result.n_days_included == 183

    def test_repr(self, synthetic_env, tmp_path):
        hours = np.arange(8760)
        diurnal = 2 * np.sin(2 * np.pi * (hours % 24) / 24) + 22
        epw_path = tmp_path / "repr.epw"
        _write_minimal_epw(epw_path, diurnal)
        result = diurnal_phase_alignment(synthetic_env, epw_path=str(epw_path))
        assert "mean_r=" in repr(result)
        assert "n_days=" in repr(result)
