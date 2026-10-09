"""Tests for capy.metrics.journey — thermal_journey."""
import numpy as np
import pandas as pd
import pytest

from capy.core.environment import ThermalEnvironment
from capy.metrics.journey import thermal_journey

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def linear_env(sample_grid):
    """Environment where T_op increases linearly from left to right across the grid.

    sample_grid is a 6×10 array of points at integer (x, y) coords.
    T_op at point i = 20 + x[i] so there is a clear west→east gradient.
    Constant over time.
    """
    time_index = pd.date_range("2001-01-01", periods=8760, freq="h")
    # x-coord gives the temperature gradient
    xs = sample_grid.points[:, 0]          # (60,)
    ta = (20.0 + xs[:, None]) * np.ones((60, 8760))
    return ThermalEnvironment(
        grid=sample_grid,
        time_index=time_index,
        variables={
            "air_temperature": ta,
            "mean_radiant_temperature": ta,
            "relative_humidity": np.full((60, 8760), 50.0),
        },
    )


# ------------------------------------------------------------------ #
# Basic shapes and values
# ------------------------------------------------------------------ #

class TestThermalJourney:
    def test_output_shapes_nearest(self, linear_env):
        path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        result = thermal_journey(linear_env, path_points=path)
        assert result.profile.shape == (3, 8760)
        assert result.gradient.shape == (2, 8760)
        assert result.transitions.shape == (2,)
        assert result.sensor_indices.shape == (3,)

    def test_output_shapes_idw(self, linear_env):
        path = np.array([[0.5, 0.5, 0.0], [2.5, 0.5, 0.0], [4.5, 0.5, 0.0]])
        result = thermal_journey(linear_env, path_points=path, method="idw")
        assert result.profile.shape == (3, 8760)
        assert result.gradient.shape == (2, 8760)

    def test_nearest_returns_exact_grid_value(self, linear_env):
        # Path exactly on grid points x=0,1,2
        path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        result = thermal_journey(linear_env, path_points=path, method="nearest")
        # T_op at x=0 → 20, x=1 → 21, x=2 → 22 (ta+mrt)/2 = ta since ta == mrt
        np.testing.assert_allclose(result.profile[:, 0], [20.0, 21.0, 22.0])

    def test_gradient_signs_warming_journey(self, linear_env):
        # Travelling in the +x direction = increasing temperature
        path = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        result = thermal_journey(linear_env, path_points=path)
        # Median gradient at each step should be positive
        assert np.median(result.gradient[0]) > 0
        assert np.median(result.gradient[1]) > 0

    def test_transitions_warming(self, linear_env):
        path = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        result = thermal_journey(linear_env, path_points=path)
        assert (result.transitions == 1).all()

    def test_transitions_cooling(self, linear_env):
        # Travelling in -x direction
        path = np.array([[5.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        result = thermal_journey(linear_env, path_points=path)
        assert (result.transitions == -1).all()

    def test_neutral_threshold_effect(self, linear_env):
        # Path with very small step → gradient < neutral_threshold
        path = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
        result = thermal_journey(linear_env, path_points=path, neutral_threshold=5.0)
        assert result.transitions[0] == 0   # small step classified as neutral

    def test_idw_smoothness(self, linear_env):
        # IDW on a mid-point between sensors should interpolate smoothly
        path = np.array([[0.5, 0.0, 0.0], [2.5, 0.0, 0.0]])
        result = thermal_journey(linear_env, path_points=path, method="idw")
        # Profile values should be between the grid extremes
        assert 20 < result.profile[0, 0] < 22
        assert 22 < result.profile[1, 0] < 24

    def test_single_point_path(self, linear_env):
        path = np.array([[0.0, 0.0, 0.0]])
        result = thermal_journey(linear_env, path_points=path)
        assert result.profile.shape == (1, 8760)
        assert result.gradient.shape == (0, 8760)
        assert result.transitions.shape == (0,)

    def test_invalid_method_raises(self, linear_env):
        path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="method"):
            thermal_journey(linear_env, path_points=path, method="kriging")

    def test_invalid_path_shape_raises(self, linear_env):
        with pytest.raises(ValueError, match="\\(m, 3\\)"):
            thermal_journey(linear_env, path_points=np.array([[0.0, 0.0]]))

    def test_with_schedule(self, linear_env, work_schedule):
        path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        result = thermal_journey(linear_env, path_points=path, schedule=work_schedule)
        # Without comfort_result, score remains nan
        assert np.isnan(result.journey_comfort_score)

    def test_comfort_score_with_comfort_result(self, linear_env):
        from capy.comfort.empirical import calculate_empirical
        comfort = calculate_empirical(linear_env, lower=19.0, upper=25.0)
        path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        result = thermal_journey(linear_env, path_points=path, comfort_result=comfort)
        assert 0.0 <= result.journey_comfort_score <= 1.0

    def test_comfort_score_all_comfortable(self, linear_env):
        from capy.comfort.empirical import calculate_empirical
        comfort = calculate_empirical(linear_env, lower=0.0, upper=100.0)
        path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        result = thermal_journey(linear_env, path_points=path, comfort_result=comfort)
        assert result.journey_comfort_score == pytest.approx(1.0)

    def test_repr(self, linear_env):
        path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        result = thermal_journey(linear_env, path_points=path)
        assert "JourneyResult" in repr(result)
        assert "n_steps=2" in repr(result)

    def test_path_points_preserved(self, linear_env):
        path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        result = thermal_journey(linear_env, path_points=path)
        np.testing.assert_array_equal(result.path_points, path)

    def test_custom_variable(self, linear_env):
        path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        result = thermal_journey(linear_env, path_points=path, variable="air_temperature")
        assert result.variable == "air_temperature"
        assert result.profile.shape == (2, 8760)
