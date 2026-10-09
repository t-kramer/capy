"""Tests for capy.metrics — sTA, THI, BHI, air movement."""
import numpy as np
import pytest

from capy.metrics import spatial_thermal_autonomy, spatial_variability
from capy.metrics.air_movement import Window, assess_air_movement

# ------------------------------------------------------------------ #
# sTA                                                                  #
# ------------------------------------------------------------------ #

class TestSpatialThermalAutonomy:
    def test_all_comfortable(self, synthetic_env):
        """When all points are comfortable, sTA should be 100%."""
        from capy.comfort.empirical import calculate_empirical
        # Wide band ensures everything is comfortable
        comfort = calculate_empirical(synthetic_env, lower=0.0, upper=50.0)
        result = spatial_thermal_autonomy(comfort)
        assert result.annual_pct == pytest.approx(100.0)

    def test_none_comfortable(self, synthetic_env):
        """When nothing is comfortable, sTA should be 0%."""
        from capy.comfort.empirical import calculate_empirical
        # Impossible band
        comfort = calculate_empirical(synthetic_env, lower=100.0, upper=101.0)
        result = spatial_thermal_autonomy(comfort)
        assert result.annual_pct == pytest.approx(0.0)

    def test_returns_correct_shape(self, synthetic_env):
        from capy.comfort.empirical import calculate_empirical
        comfort = calculate_empirical(synthetic_env)
        result = spatial_thermal_autonomy(comfort)
        assert result.hourly_array.shape == (8760,)
        assert result.comfort_hours_matrix.shape == (60, 8760)

    def test_with_schedule(self, synthetic_env, work_schedule):
        from capy.comfort.empirical import calculate_empirical
        comfort = calculate_empirical(synthetic_env)
        result = spatial_thermal_autonomy(comfort, schedule=work_schedule)
        assert result.schedule_applied is True

    def test_area_weighted_vs_uniform(self, synthetic_env):
        from capy.comfort.empirical import calculate_empirical
        comfort = calculate_empirical(synthetic_env)
        weighted = spatial_thermal_autonomy(comfort, area_weighted=True)
        uniform = spatial_thermal_autonomy(comfort, area_weighted=False)
        # With uniform areas, results should be identical
        assert weighted.annual_pct == pytest.approx(uniform.annual_pct)

    def test_tau_effect(self, synthetic_env):
        from capy.comfort.empirical import calculate_empirical
        comfort = calculate_empirical(synthetic_env)
        high_tau = spatial_thermal_autonomy(comfort, tau=0.9)
        low_tau = spatial_thermal_autonomy(comfort, tau=0.5)
        assert high_tau.annual_pct <= low_tau.annual_pct


# ------------------------------------------------------------------ #
# Spatial Variability                                                  #
# ------------------------------------------------------------------ #

class TestSpatialVariability:
    def test_uniform_env_range_is_zero(self, sample_grid):
        import pandas as pd

        from capy.core.environment import ThermalEnvironment
        time_index = pd.date_range("2001-01-01", periods=8760, freq="h")
        ta = np.full((60, 8760), 22.0)
        env = ThermalEnvironment(
            grid=sample_grid, time_index=time_index,
            variables={"air_temperature": ta, "mean_radiant_temperature": ta,
                       "relative_humidity": np.full((60, 8760), 50.0)}
        )
        result = spatial_variability(env)
        assert result.median_spatial_range == pytest.approx(0.0)
        assert result.fraction_uniform == pytest.approx(100.0)

    def test_fractions_sum_to_100(self, synthetic_env):
        result = spatial_variability(synthetic_env)
        total = result.fraction_variable + result.fraction_uniform + result.fraction_excessive
        assert total == pytest.approx(100.0, abs=0.1)

    def test_gradient_env_has_nonzero_range(self, synthetic_env):
        result = spatial_variability(synthetic_env)
        assert result.median_spatial_range > 0
        assert result.hourly_range.shape == (8760,)

    def test_custom_thresholds(self, synthetic_env):
        narrow = spatial_variability(synthetic_env, delta_t_min=1.0, delta_t_max=2.0)
        wide = spatial_variability(synthetic_env, delta_t_min=0.0, delta_t_max=100.0)
        assert wide.fraction_variable == pytest.approx(100.0)
        assert narrow.fraction_variable <= wide.fraction_variable

    def test_repr(self, synthetic_env):
        result = spatial_variability(synthetic_env)
        assert "variable=" in repr(result)
        assert "median_range=" in repr(result)


# ------------------------------------------------------------------ #
# Air Movement                                                         #
# ------------------------------------------------------------------ #

class TestAirMovement:
    def test_uniform_velocity_mode(self, synthetic_env):
        result = assess_air_movement(synthetic_env, velocity=0.5)
        assert result.data_mode == "uniform"
        assert result.mean_velocity is not None
        assert result.mean_velocity.shape == (60, 8760)
        assert 0.0 <= result.pleasure_zone_fraction <= 1.0

    def test_point_data_mode(self, synthetic_env):
        v = np.full((60, 8760), 0.3)
        result = assess_air_movement(synthetic_env, velocity=v)
        assert result.data_mode == "point_data"

    def test_geometry_proxy_mode(self, synthetic_env):
        windows = [
            Window(area=2.0, orientation=0.0),   # North
            Window(area=2.0, orientation=180.0),  # South — cross-ventilation
        ]
        result = assess_air_movement(synthetic_env, windows=windows)
        assert result.data_mode == "geometry_proxy"
        assert result.nv_potential is not None
        assert 0.0 <= result.nv_potential <= 1.0

    def test_unavailable_mode(self, synthetic_env):
        result = assess_air_movement(synthetic_env)
        assert result.data_mode == "unavailable"
        assert np.isnan(result.pleasure_zone_fraction)

    def test_cross_vent_higher_than_single_sided(self, synthetic_env):
        windows_cross = [
            Window(area=2.0, orientation=0.0),
            Window(area=2.0, orientation=180.0),
        ]
        windows_single = [
            Window(area=2.0, orientation=0.0),
            Window(area=2.0, orientation=10.0),  # almost same direction
        ]
        cross = assess_air_movement(synthetic_env, windows=windows_cross)
        single = assess_air_movement(synthetic_env, windows=windows_single)
        assert cross.nv_potential > single.nv_potential
