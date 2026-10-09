"""Tests for capy.io — CSV loader and Honeybee adapter."""

import numpy as np
import pytest

from capy.io.csv import from_csv_folder
from capy.io.honeybee import (
    _collection_to_array,
    _extract_sensor_points,
    from_honeybee_grid,
    from_lbt_data_collection,
)

N_POINTS = 10
N_HOURS = 8760


# ------------------------------------------------------------------ #
# CSV loader                                                           #
# ------------------------------------------------------------------ #

class TestCSVLoader:
    def test_basic_load(self, tmp_path):
        ta = np.full((N_POINTS, N_HOURS), 22.0)
        mrt = np.full((N_POINTS, N_HOURS), 24.0)
        rh = np.full((N_POINTS, N_HOURS), 50.0)

        np.savetxt(tmp_path / "air_temperature.csv", ta, delimiter=",")
        np.savetxt(tmp_path / "mean_radiant_temperature.csv", mrt, delimiter=",")
        np.savetxt(tmp_path / "relative_humidity.csv", rh, delimiter=",")

        env = from_csv_folder(str(tmp_path))
        assert env.grid.n_points == N_POINTS
        assert len(env.time_index) == N_HOURS
        assert "air_temperature" in env.variables

    def test_alias_recognised(self, tmp_path):
        ta = np.full((5, N_HOURS), 21.0)
        np.savetxt(tmp_path / "ta.csv", ta, delimiter=",")
        env = from_csv_folder(str(tmp_path))
        assert "air_temperature" in env.variables

    def test_missing_folder_raises(self):
        with pytest.raises((FileNotFoundError, ValueError)):
            from_csv_folder("/nonexistent/path/xyz")


# ------------------------------------------------------------------ #
# Honeybee adapter — numpy path (no LBT install required)             #
# ------------------------------------------------------------------ #

class TestHoneybeeAdapterNumpy:
    """Test from_honeybee_grid() using plain numpy arrays as collections."""

    def _make_pts(self, n=N_POINTS):
        xs = np.arange(n, dtype=float)
        pts = np.column_stack([xs, np.zeros(n), np.zeros(n)])
        return pts

    def test_numpy_array_sensor_grid(self):
        pts = self._make_pts()
        ta = np.full((N_POINTS, N_HOURS), 22.0)
        mrt = np.full((N_POINTS, N_HOURS), 24.0)
        rh = np.full((N_POINTS, N_HOURS), 50.0)

        env = from_honeybee_grid(
            sensor_grid=pts,
            data_collections={
                "air_temperature": ta,
                "mean_radiant_temperature": mrt,
                "relative_humidity": rh,
            },
        )
        assert env.grid.n_points == N_POINTS
        assert env.air_temperature().shape == (N_POINTS, N_HOURS)
        np.testing.assert_array_equal(env.air_temperature(), ta)

    def test_broadcast_1d_collection(self):
        pts = self._make_pts()
        ta_1d = np.full(N_HOURS, 22.0)
        env = from_honeybee_grid(
            sensor_grid=pts,
            data_collections={"air_temperature": ta_1d},
        )
        assert env.air_temperature().shape == (N_POINTS, N_HOURS)
        # All rows identical (broadcast)
        assert np.all(env.air_temperature()[0] == env.air_temperature()[-1])

    def test_list_of_arrays_per_sensor(self):
        pts = self._make_pts()
        # Each sensor gets a distinct offset
        collections = [np.full(N_HOURS, 20.0 + i) for i in range(N_POINTS)]
        env = from_honeybee_grid(
            sensor_grid=pts,
            data_collections={"air_temperature": collections},
        )
        assert env.air_temperature()[0, 0] == pytest.approx(20.0)
        assert env.air_temperature()[-1, 0] == pytest.approx(20.0 + N_POINTS - 1)

    def test_epw_path_stored(self):
        pts = self._make_pts()
        env = from_honeybee_grid(
            sensor_grid=pts,
            data_collections={"air_temperature": np.full((N_POINTS, N_HOURS), 22.0)},
            epw_path="/path/to/file.epw",
        )
        assert env.epw_path == "/path/to/file.epw"

    def test_unit_area_applied(self):
        pts = self._make_pts()
        env = from_honeybee_grid(
            sensor_grid=pts,
            data_collections={"air_temperature": np.full((N_POINTS, N_HOURS), 22.0)},
            unit_area=2.5,
        )
        assert env.grid.total_area == pytest.approx(N_POINTS * 2.5)

    def test_wrong_point_count_raises(self):
        pts = self._make_pts()
        # 5 collections for 10 sensors — mismatch
        bad = [np.full(N_HOURS, 22.0) for _ in range(5)]
        with pytest.raises(ValueError, match="sensor"):
            from_honeybee_grid(
                sensor_grid=pts,
                data_collections={"air_temperature": bad},
            )

    def test_wrong_array_shape_raises(self):
        pts = self._make_pts()
        with pytest.raises(ValueError):
            from_honeybee_grid(
                sensor_grid=pts,
                data_collections={"air_temperature": np.zeros((5, 100))},
            )

    def test_alias_normalisation(self):
        pts = self._make_pts()
        env = from_honeybee_grid(
            sensor_grid=pts,
            data_collections={"ta": np.full((N_POINTS, N_HOURS), 22.0)},
        )
        assert "air_temperature" in env.variables
        assert "ta" not in env.variables


class TestExtractSensorPoints:
    def test_numpy_passthrough(self):
        pts = np.zeros((5, 3))
        result = _extract_sensor_points(pts)
        assert result.shape == (5, 3)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            _extract_sensor_points(np.zeros((5, 2)))

    def test_mock_sensorgrid(self):
        """Test with a minimal duck-typed SensorGrid mock."""
        class FakePos:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z

        class FakeSensor:
            def __init__(self, i):
                self.pos = FakePos(float(i), 0.0, 0.0)

        class FakeSensorGrid:
            def __init__(self, n):
                self.sensors = [FakeSensor(i) for i in range(n)]
                self.identifier = "test_grid"

        grid = FakeSensorGrid(8)
        pts = _extract_sensor_points(grid)
        assert pts.shape == (8, 3)
        assert pts[3, 0] == pytest.approx(3.0)


class TestCollectionToArray:
    def test_2d_array(self):
        arr = np.zeros((N_POINTS, N_HOURS))
        result = _collection_to_array(arr, N_POINTS, "ta")
        assert result.shape == (N_POINTS, N_HOURS)

    def test_1d_broadcast(self):
        arr = np.ones(N_HOURS) * 5.0
        result = _collection_to_array(arr, N_POINTS, "ta")
        assert result.shape == (N_POINTS, N_HOURS)
        assert result[0, 0] == pytest.approx(5.0)

    def test_mock_lbt_collection_single(self):
        class FakeCollection:
            @property
            def values(self):
                return [22.0] * N_HOURS

        result = _collection_to_array(FakeCollection(), N_POINTS, "ta")
        assert result.shape == (N_POINTS, N_HOURS)

    def test_mock_lbt_collection_list(self):
        class FakeCollection:
            def __init__(self, val):
                self._val = val

            @property
            def values(self):
                return [self._val] * N_HOURS

        collections = [FakeCollection(float(i)) for i in range(N_POINTS)]
        result = _collection_to_array(collections, N_POINTS, "ta")
        assert result.shape == (N_POINTS, N_HOURS)
        assert result[5, 0] == pytest.approx(5.0)


class TestFromLbtDataCollection:
    def test_numpy_1d(self):
        arr = np.ones(N_HOURS) * 3.0
        result = from_lbt_data_collection(arr, "air_temperature", n_points=5)
        assert result.shape == (5, N_HOURS)

    def test_numpy_2d(self):
        arr = np.ones((N_POINTS, N_HOURS))
        result = from_lbt_data_collection(arr, "air_temperature", n_points=N_POINTS)
        assert result.shape == (N_POINTS, N_HOURS)

    def test_infers_n_points_from_list(self):
        collections = [np.ones(N_HOURS) for _ in range(7)]
        result = from_lbt_data_collection(collections, "air_temperature")
        assert result.shape == (7, N_HOURS)
