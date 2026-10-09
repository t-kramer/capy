"""Tests for capy.metrics.alliesthesia — thermal_niches, context_score."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capy.core.environment import ThermalEnvironment
from capy.io.epw import load_dry_bulb as _load_outdoor_drybulb
from capy.io.epw import running_mean as _running_mean
from capy.metrics.alliesthesia import context_score, thermal_niches

# ------------------------------------------------------------------ #
# EPW fixture helper
# ------------------------------------------------------------------ #

def _write_epw(path: Path, dry_bulb: np.ndarray) -> None:
    with open(path, "w") as f:
        for _ in range(8):
            f.write("HEADER,x\n")
        for v in dry_bulb:
            f.write(f"2001,1,1,1,0,x,{v:.2f},0,0\n")


# ------------------------------------------------------------------ #
# thermal_niches
# ------------------------------------------------------------------ #

class TestThermalNiches:
    def test_uniform_field_is_uniform(self, sample_grid):
        data = np.full((60, 8760), 22.0)
        env = ThermalEnvironment(
            grid=sample_grid,
            time_index=pd.date_range("2001-01-01", periods=8760, freq="h"),
            variables={"air_temperature": data, "mean_radiant_temperature": data,
                       "relative_humidity": np.full((60, 8760), 50.0)},
        )
        result = thermal_niches(env)
        assert result.niche_fraction == pytest.approx(0.0)
        assert result.uniform_fraction == pytest.approx(1.0)

    def test_gradient_env_has_niches(self, synthetic_env):
        # synthetic_env has 0–4°C spatial gradient → niches exist
        result = thermal_niches(synthetic_env)
        assert result.niche_fraction > 0.0

    def test_fractions_bounded(self, synthetic_env):
        result = thermal_niches(synthetic_env)
        assert 0.0 <= result.niche_fraction <= 1.0
        assert 0.0 <= result.discomfort_fraction <= 1.0
        assert 0.0 <= result.uniform_fraction <= 1.0

    def test_narrow_deviation_gives_more_niches(self, synthetic_env):
        narrow = thermal_niches(synthetic_env, niche_deviation=0.1)
        wide = thermal_niches(synthetic_env, niche_deviation=5.0)
        assert narrow.niche_fraction >= wide.niche_fraction

    def test_with_schedule(self, synthetic_env, work_schedule):
        result = thermal_niches(synthetic_env, schedule=work_schedule)
        assert 0.0 <= result.niche_fraction <= 1.0
        # Occupied hours only
        n_occ = work_schedule.n_occupied
        assert len(result.hourly_n_niches) == n_occ

    def test_hourly_arrays_shape_no_schedule(self, synthetic_env):
        result = thermal_niches(synthetic_env)
        assert result.hourly_n_niches.shape == (8760,)
        assert result.hourly_spatial_range.shape == (8760,)

    def test_extreme_gradient_increases_discomfort(self, sample_grid):
        # 0–10°C gradient should push most hours into discomfort (> delta_t_max=4)
        hours = 8760
        gradient = np.linspace(0, 10, 60)
        ta = np.ones((60, hours)) * 22 + gradient[:, None]
        env = ThermalEnvironment(
            grid=sample_grid,
            time_index=pd.date_range("2001-01-01", periods=hours, freq="h"),
            variables={"air_temperature": ta, "mean_radiant_temperature": ta,
                       "relative_humidity": np.full((60, hours), 50.0)},
        )
        result = thermal_niches(env, delta_t_max=4.0)
        assert result.discomfort_fraction == pytest.approx(1.0)

    def test_repr(self, synthetic_env):
        r = repr(thermal_niches(synthetic_env))
        assert "%" in r


# ------------------------------------------------------------------ #
# context_score
# ------------------------------------------------------------------ #

class TestContextScore:
    def _warm_season_epw(self, path: Path):
        # 35°C outdoor all year → warm season
        _write_epw(path, np.full(8760, 35.0))

    def _cold_season_epw(self, path: Path):
        # 5°C outdoor all year → cold season
        _write_epw(path, np.full(8760, 5.0))

    def test_no_epw_raises(self, synthetic_env):
        from capy.comfort.empirical import calculate_empirical
        comfort = calculate_empirical(synthetic_env)
        with pytest.raises(ValueError, match="EPW"):
            context_score(synthetic_env, comfort)

    def test_returns_correct_shape(self, synthetic_env, tmp_path):
        from capy.comfort.empirical import calculate_empirical
        self._warm_season_epw(tmp_path / "warm.epw")
        comfort = calculate_empirical(synthetic_env)
        result = context_score(synthetic_env, comfort, epw_path=str(tmp_path / "warm.epw"))
        assert result.scores.shape == (60, 8760)
        assert result.scores.dtype == np.int8

    def test_fractions_sum_to_one(self, synthetic_env, tmp_path):
        from capy.comfort.empirical import calculate_empirical
        self._warm_season_epw(tmp_path / "warm.epw")
        comfort = calculate_empirical(synthetic_env)
        result = context_score(synthetic_env, comfort, epw_path=str(tmp_path / "warm.epw"))
        total = result.beneficial_fraction + result.neutral_fraction + result.problematic_fraction
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_scores_in_valid_range(self, synthetic_env, tmp_path):
        from capy.comfort.empirical import calculate_empirical
        self._warm_season_epw(tmp_path / "w.epw")
        comfort = calculate_empirical(synthetic_env)
        result = context_score(synthetic_env, comfort, epw_path=str(tmp_path / "w.epw"))
        assert set(np.unique(result.scores)).issubset({-1, 0, 1})

    def test_all_comfortable_gives_no_beneficial_or_problematic(self, synthetic_env, tmp_path):
        from capy.comfort.empirical import calculate_empirical
        # Very wide band → everything comfortable
        self._warm_season_epw(tmp_path / "e.epw")
        comfort = calculate_empirical(synthetic_env, lower=0.0, upper=100.0)
        result = context_score(synthetic_env, comfort, epw_path=str(tmp_path / "e.epw"))
        assert result.beneficial_fraction == pytest.approx(0.0)
        assert result.problematic_fraction == pytest.approx(0.0)

    def test_schedule_zeros_unoccupied(self, synthetic_env, work_schedule, tmp_path):
        from capy.comfort.empirical import calculate_empirical
        self._warm_season_epw(tmp_path / "e.epw")
        comfort = calculate_empirical(synthetic_env)
        result = context_score(
            synthetic_env, comfort, epw_path=str(tmp_path / "e.epw"), schedule=work_schedule
        )
        unocc = ~work_schedule.mask
        assert (result.scores[:, unocc] == 0).all()

    def test_repr(self, synthetic_env, tmp_path):
        from capy.comfort.empirical import calculate_empirical
        self._warm_season_epw(tmp_path / "r.epw")
        comfort = calculate_empirical(synthetic_env)
        result = context_score(synthetic_env, comfort, epw_path=str(tmp_path / "r.epw"))
        assert "%" in repr(result)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

class TestRunningMean:
    def test_constant_input_returns_constant(self):
        t = np.full(8760, 20.0)
        rm = _running_mean(t, alpha=0.8, n_days=7)
        assert rm.shape == (8760,)
        np.testing.assert_allclose(rm[24:], 20.0, atol=0.5)

    def test_step_change_converges(self):
        t = np.concatenate([np.full(4380, 10.0), np.full(4380, 30.0)])
        rm = _running_mean(t, alpha=0.8, n_days=7)
        # Should approach 30 after several days
        assert rm[-1] > 25.0


class TestLoadOutdoorDrybulb:
    def test_loads_from_minimal_epw(self, tmp_path):
        vals = np.full(8760, 15.0)
        _write_epw(tmp_path / "t.epw", vals)
        arr = _load_outdoor_drybulb(str(tmp_path / "t.epw"))
        assert arr.shape == (8760,)
        np.testing.assert_allclose(arr, 15.0)
