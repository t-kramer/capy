"""Tests for capy.core — SpatialGrid, ThermalEnvironment, OccupancySchedule."""
import numpy as np
import pandas as pd
import pytest

from capy.core.environment import ThermalEnvironment
from capy.core.grid import SpatialGrid
from capy.core.schedule import OccupancySchedule

# ------------------------------------------------------------------ #
# SpatialGrid                                                          #
# ------------------------------------------------------------------ #

class TestSpatialGrid:
    def test_basic_construction(self):
        pts = np.zeros((10, 3))
        areas = np.ones(10)
        grid = SpatialGrid(points=pts, areas=areas)
        assert grid.n_points == 10
        assert grid.total_area == pytest.approx(10.0)

    def test_zone_id_and_name(self):
        pts = np.zeros((4, 3))
        grid = SpatialGrid(
            points=pts, areas=np.ones(4),
            zone_id="98juht34jk", zone_name="Office, South"
        )
        assert grid.zone_id == "98juht34jk"
        assert grid.zone_name == "Office, South"
        assert "Office, South" in repr(grid)

    def test_area_weights_sum_to_one(self, sample_grid):
        assert sample_grid.area_weights.sum() == pytest.approx(1.0)

    def test_wrong_points_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            SpatialGrid(
                points=np.zeros((10, 2)),  # wrong: should be (n, 3)
                areas=np.ones(10),
            )

    def test_mismatched_areas_raises(self):
        with pytest.raises(ValueError):
            SpatialGrid(
                points=np.zeros((10, 3)),
                areas=np.ones(5),   # wrong length
            )

    def test_from_points_factory(self):
        pts = np.random.rand(20, 3)
        grid = SpatialGrid.from_points(pts, unit_area=2.0, zone_id=1, zone_name="Zone A")
        assert grid.n_points == 20
        assert grid.total_area == pytest.approx(40.0)
        assert grid.zone_id == 1
        assert grid.zone_name == "Zone A"

    def test_from_pts_file(self, tmp_path):
        pts_content = "1.0 2.0 0.8 0.0 0.0 1.0\n3.0 4.0 0.8 0.0 0.0 1.0\n"
        pts_file = tmp_path / "grid.pts"
        pts_file.write_text(pts_content)
        grid = SpatialGrid.from_pts_file(str(pts_file), unit_area=0.5)
        assert grid.n_points == 2
        assert grid.points[0, 0] == pytest.approx(1.0)
        assert grid.total_area == pytest.approx(1.0)

    def test_area_weights_zero_raises(self):
        grid = SpatialGrid(points=np.zeros((3, 3)), areas=np.zeros(3))
        with pytest.raises(ValueError, match="zero"):
            _ = grid.area_weights


# ------------------------------------------------------------------ #
# OccupancySchedule                                                    #
# ------------------------------------------------------------------ #

class TestOccupancySchedule:
    def test_work_hours_default(self):
        sch = OccupancySchedule.work_hours()
        assert sch.mask.shape == (8760,)
        # Weekdays 8–18 = 10h/day × 5 days/week × ~52 weeks ≈ 2600h
        assert 2500 < sch.n_occupied < 2700

    def test_always(self):
        sch = OccupancySchedule.always()
        assert sch.n_occupied == 8760
        assert sch.occupied_fraction == pytest.approx(1.0)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            OccupancySchedule(mask=np.ones(100, dtype=bool))

    def test_work_hours_includes_weekends(self):
        without = OccupancySchedule.work_hours(include_weekends=False)
        with_ = OccupancySchedule.work_hours(include_weekends=True)
        assert with_.n_occupied > without.n_occupied

    def test_from_array(self):
        arr = np.zeros(8760, dtype=bool)
        arr[:100] = True
        sch = OccupancySchedule.from_array(arr)
        assert sch.n_occupied == 100

    def test_from_hours(self):
        sch = OccupancySchedule.from_hours([9, 10, 11])
        assert sch.n_occupied == 3 * 365

    def test_as_indices(self):
        sch = OccupancySchedule.from_array(np.zeros(8760, dtype=bool))
        sch.mask[0] = True
        sch.mask[100] = True
        idx = sch.as_indices()
        assert list(idx) == [0, 100]


# ------------------------------------------------------------------ #
# ThermalEnvironment                                                   #
# ------------------------------------------------------------------ #

class TestThermalEnvironment:
    def test_basic_properties(self, sample_env):
        env = sample_env
        assert env.grid.n_points == 60
        assert len(env.time_index) == 8760
        assert "air_temperature" in env.variables

    def test_accessors(self, sample_env):
        ta = sample_env.air_temperature()
        assert ta.shape == (60, 8760)
        mrt = sample_env.mrt()
        assert mrt.shape == (60, 8760)

    def test_operative_temperature_computed(self, sample_env):
        t_op = sample_env.operative_temperature()
        assert t_op.shape == (60, 8760)
        # Should equal (ta + mrt) / 2 when not stored directly
        expected = (sample_env.air_temperature() + sample_env.mrt()) / 2
        np.testing.assert_allclose(t_op, expected)

    def test_operative_temperature_stored(self, sample_env):
        t_op_manual = sample_env.operative_temperature()
        sample_env.add("operative_temperature", t_op_manual)
        assert np.array_equal(sample_env.operative_temperature(), t_op_manual)

    def test_alias_resolution(self, sample_grid):
        time_index = pd.date_range("2001-01-01", periods=8760, freq="h")
        ta = np.ones((60, 8760)) * 22.0
        # Use alias 'ta' instead of canonical 'air_temperature'
        env = ThermalEnvironment(
            grid=sample_grid, time_index=time_index, variables={"ta": ta}
        )
        # Should be stored under canonical name
        assert "air_temperature" in env.variables
        assert "ta" not in env.variables

    def test_add_variable(self, sample_env):
        new_var = np.zeros((60, 8760))
        sample_env.add("co2_concentration", new_var)
        assert "co2_concentration" in sample_env.variables

    def test_add_wrong_shape_raises(self, sample_env):
        with pytest.raises(ValueError):
            sample_env.add("bad_var", np.zeros((10, 100)))

    def test_missing_variable_raises(self, sample_env):
        with pytest.raises(KeyError):
            sample_env.get("nonexistent_variable")

    def test_repr(self, sample_env):
        r = repr(sample_env)
        assert "60" in r
        assert "8760" in r

    def test_has(self, sample_env):
        assert sample_env.has("air_temperature") is True
        assert sample_env.has("ta") is True        # alias
        assert sample_env.has("co2") is False

    def test_air_velocity_raises_when_missing(self, sample_env):
        with pytest.raises(KeyError):
            sample_env.air_velocity()

    def test_with_schedule_masks_unoccupied(self, sample_env, work_schedule):
        filtered = sample_env.with_schedule(work_schedule)
        ta = filtered.air_temperature()
        unoccupied = ~work_schedule.as_boolean()
        assert np.all(np.isnan(ta[:, unoccupied]))
        assert not np.any(np.isnan(ta[:, work_schedule.as_boolean()]))

    def test_with_schedule_wrong_length_raises(self, sample_env):
        bad_schedule = OccupancySchedule(mask=np.ones(8760, dtype=bool))
        bad_schedule.mask = np.ones(100, dtype=bool)  # bypass validation
        bad_schedule.mask.resize(100)
        # Simulate mismatched length via a manual mask
        class _ShortSchedule:
            mask = np.ones(100, dtype=bool)
            def as_boolean(self): return self.mask
        with pytest.raises(ValueError, match="length"):
            sample_env.with_schedule(_ShortSchedule())

    def test_from_arrays(self, sample_grid):
        ta = np.full((60, 8760), 22.0)
        mrt = np.full((60, 8760), 24.0)
        env = ThermalEnvironment.from_arrays(
            grid=sample_grid,
            variables={"ta": ta, "mrt": mrt},
        )
        assert env.grid.n_points == 60
        assert len(env.time_index) == 8760
        assert env.has("air_temperature")
        assert env.has("mean_radiant_temperature")


# ------------------------------------------------------------------ #
# Integration: CSV load                                                #
# ------------------------------------------------------------------ #

class TestCSVLoader:
    def test_load_from_csv_folder(self, tmp_path):
        from capy.io.csv import from_csv_folder

        n = 5
        ta = np.full((n, 8760), 22.0)
        mrt = np.full((n, 8760), 24.0)
        rh = np.full((n, 8760), 50.0)

        np.savetxt(tmp_path / "air_temperature.csv", ta, delimiter=",")
        np.savetxt(tmp_path / "mean_radiant_temperature.csv", mrt, delimiter=",")
        np.savetxt(tmp_path / "relative_humidity.csv", rh, delimiter=",")

        env = from_csv_folder(str(tmp_path))
        assert env.grid.n_points == n
        assert len(env.time_index) == 8760
        assert "air_temperature" in env.variables

    def test_load_real_sample_data(self):
        from pathlib import Path

        from capy.io.csv import from_csv_folder

        csv_dir = Path(__file__).parent / "data" / "csv"
        if not csv_dir.exists():
            pytest.skip("Sample data not available")

        env = from_csv_folder(str(csv_dir))
        assert env.grid.n_points > 0
        assert len(env.time_index) == 8760
