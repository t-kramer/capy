"""Unit tests for capy.comfort — base, empirical, and PMV models."""
import numpy as np
import pytest

from capy.comfort.base import ComfortResult
from capy.comfort.empirical import DEFAULT_LOWER, DEFAULT_UPPER, calculate_empirical

# ------------------------------------------------------------------ #
# ComfortResult — base class
# ------------------------------------------------------------------ #

class TestComfortResult:
    def test_construction(self, synthetic_env):
        n_pts = synthetic_env.grid.n_points
        n_h = len(synthetic_env.time_index)
        values = np.zeros((n_pts, n_h))
        discrete = np.zeros((n_pts, n_h), dtype=np.int8)
        result = ComfortResult(
            values=values, discrete=discrete,
            model="Test", comfort_band=(21.0, 28.0), env=synthetic_env,
        )
        assert result.model == "Test"
        assert result.comfort_band == (21.0, 28.0)

    def test_values_shape_mismatch_raises(self, synthetic_env):
        n_pts = synthetic_env.grid.n_points
        n_h = len(synthetic_env.time_index)
        with pytest.raises(ValueError, match="values shape"):
            ComfortResult(
                values=np.zeros((n_pts + 1, n_h)),
                discrete=np.zeros((n_pts, n_h), dtype=np.int8),
                model="Test", comfort_band=(21.0, 28.0), env=synthetic_env,
            )

    def test_discrete_shape_mismatch_raises(self, synthetic_env):
        n_pts = synthetic_env.grid.n_points
        n_h = len(synthetic_env.time_index)
        with pytest.raises(ValueError, match="discrete shape"):
            ComfortResult(
                values=np.zeros((n_pts, n_h)),
                discrete=np.zeros((n_pts, n_h + 1), dtype=np.int8),
                model="Test", comfort_band=(21.0, 28.0), env=synthetic_env,
            )

    def test_is_comfortable_returns_boolean(self, synthetic_env):
        result = calculate_empirical(synthetic_env, lower=0.0, upper=100.0)
        mask = result.is_comfortable()
        assert mask.dtype == bool
        assert mask.shape == (synthetic_env.grid.n_points, len(synthetic_env.time_index))

    def test_comfort_fraction_all_comfortable(self, synthetic_env):
        result = calculate_empirical(synthetic_env, lower=0.0, upper=100.0)
        frac = result.comfort_fraction()
        np.testing.assert_allclose(frac, 1.0)

    def test_comfort_fraction_none_comfortable(self, synthetic_env):
        result = calculate_empirical(synthetic_env, lower=100.0, upper=101.0)
        frac = result.comfort_fraction()
        np.testing.assert_allclose(frac, 0.0)

    def test_comfort_fraction_with_schedule(self, synthetic_env, work_schedule):
        result = calculate_empirical(synthetic_env, lower=0.0, upper=100.0)
        frac = result.comfort_fraction(schedule=work_schedule)
        assert frac.shape == (synthetic_env.grid.n_points,)
        np.testing.assert_allclose(frac, 1.0)

    def test_repr(self, synthetic_env):
        result = calculate_empirical(synthetic_env)
        r = repr(result)
        assert "Empirical" in r
        assert "band" in r


# ------------------------------------------------------------------ #
# calculate_empirical
# ------------------------------------------------------------------ #

class TestCalculateEmpirical:
    def test_output_shape(self, synthetic_env):
        result = calculate_empirical(synthetic_env)
        n_pts = synthetic_env.grid.n_points
        n_h = len(synthetic_env.time_index)
        assert result.values.shape == (n_pts, n_h)
        assert result.discrete.shape == (n_pts, n_h)

    def test_discrete_values_are_valid(self, synthetic_env):
        result = calculate_empirical(synthetic_env)
        assert set(np.unique(result.discrete)).issubset({-1, 0, 1})

    def test_all_comfortable_when_band_covers_everything(self, synthetic_env):
        result = calculate_empirical(synthetic_env, lower=0.0, upper=100.0)
        assert (result.discrete == 0).all()

    def test_none_comfortable_when_band_is_out_of_range(self, synthetic_env):
        result = calculate_empirical(synthetic_env, lower=100.0, upper=101.0)
        assert (result.discrete != 0).all()

    def test_cold_classification(self, synthetic_env):
        result = calculate_empirical(synthetic_env, lower=100.0, upper=101.0)
        # All temps well below 100 → all classified as too cold (-1)
        assert (result.discrete == -1).all()

    def test_hot_classification(self, synthetic_env):
        result = calculate_empirical(synthetic_env, lower=-50.0, upper=-1.0)
        # All temps above -1 → all classified as too warm (+1)
        assert (result.discrete == 1).all()

    def test_boundary_lower_is_comfortable(self, synthetic_env):
        # Points exactly at lower bound should be comfortable
        t_op = synthetic_env.operative_temperature()
        lower = float(t_op.min())
        result = calculate_empirical(synthetic_env, lower=lower, upper=100.0)
        assert (result.discrete >= 0).all()

    def test_boundary_upper_is_comfortable(self, synthetic_env):
        t_op = synthetic_env.operative_temperature()
        upper = float(t_op.max())
        result = calculate_empirical(synthetic_env, lower=0.0, upper=upper)
        assert (result.discrete <= 0).all()

    def test_model_name(self, synthetic_env):
        result = calculate_empirical(synthetic_env)
        assert result.model == "Empirical"

    def test_comfort_band_stored(self, synthetic_env):
        result = calculate_empirical(synthetic_env, lower=20.0, upper=26.0)
        assert result.comfort_band == (20.0, 26.0)

    def test_values_are_operative_temperature(self, synthetic_env):
        result = calculate_empirical(synthetic_env)
        t_op = synthetic_env.operative_temperature()
        np.testing.assert_array_equal(result.values, t_op)

    def test_default_bounds(self, synthetic_env):
        result = calculate_empirical(synthetic_env)
        assert result.comfort_band == (DEFAULT_LOWER, DEFAULT_UPPER)

    def test_returns_comfort_result(self, synthetic_env):
        result = calculate_empirical(synthetic_env)
        assert isinstance(result, ComfortResult)


# ------------------------------------------------------------------ #
# calculate_pmv
# ------------------------------------------------------------------ #


class TestCalculatePMV:
    def test_output_shape(self, synthetic_env):
        from capy.comfort.pmv import calculate_pmv
        result = calculate_pmv(synthetic_env)
        n_pts = synthetic_env.grid.n_points
        n_h = len(synthetic_env.time_index)
        assert result.values.shape == (n_pts, n_h)
        assert result.discrete.shape == (n_pts, n_h)

    def test_discrete_values_are_valid(self, synthetic_env):
        from capy.comfort.pmv import calculate_pmv
        result = calculate_pmv(synthetic_env)
        assert set(np.unique(result.discrete)).issubset({-1, 0, 1})

    def test_velocity_fallback(self, synthetic_env):
        from capy.comfort.pmv import calculate_pmv
        # synthetic_env has no air_velocity — should use default without raising
        result = calculate_pmv(synthetic_env, default_velocity=0.1)
        assert result.values.shape == (synthetic_env.grid.n_points, len(synthetic_env.time_index))

    def test_model_name(self, synthetic_env):
        from capy.comfort.pmv import calculate_pmv
        result = calculate_pmv(synthetic_env)
        assert result.model == "PMV"

    def test_returns_comfort_result(self, synthetic_env):
        from capy.comfort.pmv import calculate_pmv
        result = calculate_pmv(synthetic_env)
        assert isinstance(result, ComfortResult)

    def test_custom_params(self, synthetic_env):
        from capy.comfort.pmv import PMVParameters, calculate_pmv
        params = PMVParameters(met=1.0, clo=0.5)
        result = calculate_pmv(synthetic_env, params=params)
        assert result.values.shape == (synthetic_env.grid.n_points, len(synthetic_env.time_index))

    def test_with_air_velocity_in_env(self, synthetic_env):
        from capy.comfort.pmv import calculate_pmv
        synthetic_env.add("air_velocity", np.full(
            (synthetic_env.grid.n_points, len(synthetic_env.time_index)), 0.2
        ))
        result = calculate_pmv(synthetic_env)
        assert result.values.shape == (synthetic_env.grid.n_points, len(synthetic_env.time_index))
