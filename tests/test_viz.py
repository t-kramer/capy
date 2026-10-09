"""Tests for capy.viz — heatmap and spatial plot functions.

Uses the non-interactive Agg backend so no display is required.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ------------------------------------------------------------------ #
# Hourly heatmap
# ------------------------------------------------------------------ #

class TestPlotHourlyHeatmap:
    def test_returns_figure_from_env(self, synthetic_env):
        from capy.viz.heatmap import plot_hourly_heatmap
        fig = plot_hourly_heatmap(synthetic_env)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_from_comfort(self, synthetic_env):
        from capy.comfort.empirical import calculate_empirical
        from capy.viz.heatmap import plot_hourly_heatmap
        comfort = calculate_empirical(synthetic_env)
        fig = plot_hourly_heatmap(comfort)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_variable(self, synthetic_env):
        from capy.viz.heatmap import plot_hourly_heatmap
        fig = plot_hourly_heatmap(synthetic_env, variable="air_temperature")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_schedule(self, synthetic_env, work_schedule):
        from capy.viz.heatmap import plot_hourly_heatmap
        fig = plot_hourly_heatmap(synthetic_env, schedule=work_schedule)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_from_occupancy_schedule(self, work_schedule):
        from capy.viz.heatmap import plot_hourly_heatmap
        fig = plot_hourly_heatmap(work_schedule)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_axis_origin_lower(self, synthetic_env):
        from capy.viz.heatmap import plot_hourly_heatmap
        fig = plot_hourly_heatmap(synthetic_env)
        ax = fig.axes[0]
        im = ax.get_images()[0]
        # extent bottom should be 0 (hour 0 at bottom)
        assert im.get_extent()[2] == pytest.approx(0.0)

    def test_save_path(self, synthetic_env, tmp_path):
        from capy.viz.heatmap import plot_hourly_heatmap
        out = tmp_path / "hourly.png"
        plot_hourly_heatmap(synthetic_env, save_path=str(out))
        assert out.exists()

    def test_no_spines(self, synthetic_env):
        from capy.viz.heatmap import plot_hourly_heatmap
        fig = plot_hourly_heatmap(synthetic_env)
        ax = fig.axes[0]
        for spine in ax.spines.values():
            assert not spine.get_visible()

    def test_unoccupied_transparent_not_grey(self, synthetic_env, work_schedule):
        # Unoccupied NaN pixels should use an RGBA bad colour with alpha=0, not #d0d0d0
        from capy.viz.heatmap import plot_hourly_heatmap
        fig = plot_hourly_heatmap(synthetic_env, schedule=work_schedule)
        ax = fig.axes[0]
        im = ax.get_images()[0]
        bad_color = im.cmap(np.ma.masked)
        # alpha channel (index 3) must be 0 (transparent)
        assert bad_color[3] == pytest.approx(0.0)


# ------------------------------------------------------------------ #
# Spatial floor map
# ------------------------------------------------------------------ #

class TestPlotFloorMap:
    def test_returns_figure_from_env(self, synthetic_env):
        from capy.viz.spatial import plot_floor_map
        fig = plot_floor_map(synthetic_env)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_from_comfort(self, synthetic_env):
        from capy.comfort.empirical import calculate_empirical
        from capy.viz.spatial import plot_floor_map
        comfort = calculate_empirical(synthetic_env)
        fig = plot_floor_map(comfort, variable="comfort_fraction")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_specific_hour(self, synthetic_env):
        from capy.viz.spatial import plot_floor_map
        fig = plot_floor_map(synthetic_env, hour=1000)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_invalid_hour_raises(self, synthetic_env):
        from capy.viz.spatial import plot_floor_map
        with pytest.raises(ValueError, match="hour"):
            plot_floor_map(synthetic_env, hour=99999)

    def test_invalid_stat_raises(self, synthetic_env):
        from capy.viz.spatial import plot_floor_map
        with pytest.raises(ValueError, match="annual_stat"):
            plot_floor_map(synthetic_env, annual_stat="mode")

    def test_annual_stats(self, synthetic_env):
        from capy.viz.spatial import plot_floor_map
        for stat in ("mean", "median", "max", "min"):
            fig = plot_floor_map(synthetic_env, annual_stat=stat)
            assert isinstance(fig, matplotlib.figure.Figure)

    def test_save_path(self, synthetic_env, tmp_path):
        from capy.viz.spatial import plot_floor_map
        out = tmp_path / "floor.png"
        plot_floor_map(synthetic_env, save_path=str(out))
        assert out.exists()

    def test_custom_title(self, synthetic_env):
        from capy.viz.spatial import plot_floor_map
        fig = plot_floor_map(synthetic_env, title="My map")
        ax = fig.axes[0]
        assert "My map" in ax.get_title(loc="left")

    def test_no_spines(self, synthetic_env):
        from capy.viz.spatial import plot_floor_map
        fig = plot_floor_map(synthetic_env)
        ax = fig.axes[0]
        for spine in ax.spines.values():
            assert not spine.get_visible()

    def test_no_arrow_when_north_angle_zero(self, synthetic_env):
        # default grid.north_angle=0 → no North arrow
        from capy.viz.spatial import plot_floor_map
        fig = plot_floor_map(synthetic_env)
        ax = fig.axes[0]
        assert len(ax.texts) == 0

    def test_arrow_when_north_angle_set(self, synthetic_env):
        # grid with non-zero north_angle → North arrow appears
        from capy.viz.spatial import plot_floor_map
        synthetic_env.grid.north_angle = 30.0
        fig = plot_floor_map(synthetic_env)
        ax = fig.axes[0]
        assert any(t.get_text() == "N" for t in ax.texts)

    def test_rotate_false_suppresses_rotation(self, sample_grid):
        # rotate=False ignores north_angle even when set
        from capy.viz.spatial import _rotated_xy
        sample_grid.north_angle = 90.0
        xs_no_rot, _ = _rotated_xy(sample_grid, angle=0.0)
        np.testing.assert_array_equal(xs_no_rot, sample_grid.points[:, 0])

    def test_north_angle_rotates_coordinates(self, sample_grid):
        from capy.viz.spatial import _rotated_xy
        xs_rot, _ = _rotated_xy(sample_grid, angle=90.0)
        assert not np.allclose(xs_rot, sample_grid.points[:, 0])


# ------------------------------------------------------------------ #
# Journey profile
# ------------------------------------------------------------------ #

class TestPlotJourneyProfile:
    def _make_journey(self, synthetic_env):
        from capy.metrics.journey import thermal_journey
        path = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ])
        return thermal_journey(synthetic_env, path_points=path)

    def test_returns_figure(self, synthetic_env):
        from capy.viz.journey import plot_journey_profile
        journey = self._make_journey(synthetic_env)
        fig = plot_journey_profile(journey)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_specific_hours(self, synthetic_env):
        from capy.viz.journey import plot_journey_profile
        journey = self._make_journey(synthetic_env)
        fig = plot_journey_profile(journey, hours=[0, 1000, 4380])
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_no_transitions(self, synthetic_env):
        from capy.viz.journey import plot_journey_profile
        journey = self._make_journey(synthetic_env)
        fig = plot_journey_profile(journey, show_transitions=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_save_path(self, synthetic_env, tmp_path):
        from capy.viz.journey import plot_journey_profile
        journey = self._make_journey(synthetic_env)
        out = tmp_path / "journey.png"
        plot_journey_profile(journey, save_path=str(out))
        assert out.exists()

    def test_invalid_hour_raises(self, synthetic_env):
        from capy.viz.journey import plot_journey_profile
        journey = self._make_journey(synthetic_env)
        with pytest.raises(ValueError, match="hour"):
            plot_journey_profile(journey, hours=[99999])

    def test_single_step_no_crash(self, synthetic_env):
        from capy.metrics.journey import thermal_journey
        from capy.viz.journey import plot_journey_profile
        path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        journey = thermal_journey(synthetic_env, path_points=path)
        fig = plot_journey_profile(journey)
        assert isinstance(fig, matplotlib.figure.Figure)
