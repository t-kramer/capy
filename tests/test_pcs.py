"""Tests for capy.pcs — Personal Comfort Systems devices."""
import numpy as np
import pytest

from capy.pcs import (
    CooledChair,
    Fan,
    FootWarmer,
    HeatedChair,
    PersonalHeater,
    PointSelector,
)

N_PTS = 60
N_HRS = 8760


# ---------------------------------------------------------------------------
# PointSelector
# ---------------------------------------------------------------------------

class TestPointSelector:
    def test_from_indices(self, sample_grid):
        sel = PointSelector.from_indices([0, 1, 2], n_points=N_PTS)
        mask = sel.resolve()
        assert mask.shape == (N_PTS,)
        assert mask[:3].all()
        assert not mask[3:].any()

    def test_from_fraction(self, sample_grid):
        sel = PointSelector.from_fraction(sample_grid, fraction=0.5, seed=42)
        mask = sel.resolve()
        assert mask.sum() == round(N_PTS * 0.5)

    def test_from_fraction_invalid(self, sample_grid):
        with pytest.raises(ValueError, match="fraction"):
            PointSelector.from_fraction(sample_grid, fraction=0.0)

    def test_from_bbox(self, sample_grid):
        # Grid is 6×10; x in [0,5], y in [0,9]
        sel = PointSelector.from_bbox(sample_grid, x=(0, 2), y=(0, 9))
        mask = sel.resolve()
        # Points with x <= 2 — 3 columns × 10 rows = 30
        assert mask.sum() == 30

    def test_repr(self, sample_grid):
        sel = PointSelector.from_indices([0, 1], n_points=N_PTS)
        assert "2/60" in repr(sel)


# ---------------------------------------------------------------------------
# Fan
# ---------------------------------------------------------------------------

class TestFan:
    def test_adds_velocity_to_selected_points(self, synthetic_env):
        fan = Fan(delta_v=0.5)
        env2 = fan.apply(synthetic_env, points=[0, 1, 2])
        v = env2.get("air_velocity")
        assert v[0, 0] == pytest.approx(0.1 + 0.5)
        assert v[1, 0] == pytest.approx(0.1 + 0.5)
        assert v[2, 0] == pytest.approx(0.1 + 0.5)

    def test_creates_velocity_if_missing(self, synthetic_env):
        assert not synthetic_env.has("air_velocity")
        env2 = Fan(delta_v=0.5).apply(synthetic_env, points=[0])
        v = env2.get("air_velocity")
        assert v[0, 0] == pytest.approx(0.6)
        # Env that had air_velocity set should also work
        assert v.shape == (N_PTS, N_HRS)

    def test_unselected_points_unchanged(self, synthetic_env):
        env2 = Fan(delta_v=0.5).apply(synthetic_env, points=[0, 1])
        v = env2.get("air_velocity")
        assert v[3, 0] == pytest.approx(0.1)

    def test_when_schedule(self, synthetic_env, work_schedule):
        env2 = Fan(delta_v=0.5).apply(synthetic_env, points=[0], when=work_schedule)
        v = env2.get("air_velocity")
        occ = work_schedule.as_boolean()
        # Occupied hours: velocity raised
        assert v[0, occ][0] == pytest.approx(0.6)
        # Unoccupied hours: velocity at baseline
        assert v[0, ~occ][0] == pytest.approx(0.1)

    def test_when_comfort_mask(self, synthetic_env):
        # Build a (n_pts, n_hrs) mask — first 100 hours for all points
        mask = np.zeros((N_PTS, N_HRS), dtype=bool)
        mask[:, :100] = True
        env2 = Fan(delta_v=0.5).apply(synthetic_env, points=[0, 1], when=mask)
        v = env2.get("air_velocity")
        assert v[0, 50] == pytest.approx(0.6)
        assert v[0, 200] == pytest.approx(0.1)

    def test_immutability(self, synthetic_env):
        Fan(delta_v=0.5).apply(synthetic_env, points=[0])
        assert not synthetic_env.has("air_velocity")


# ---------------------------------------------------------------------------
# FootWarmer / HeatedChair / CooledChair / PersonalHeater
# ---------------------------------------------------------------------------

class TestMRTDevices:
    def test_footwarmer_raises_mrt(self, synthetic_env):
        mrt_before = synthetic_env.get("mean_radiant_temperature")[0, 0]
        env2 = FootWarmer(delta_mrt=3.0).apply(synthetic_env, points=[0])
        assert env2.get("mean_radiant_temperature")[0, 0] == pytest.approx(mrt_before + 3.0)

    def test_heated_chair_raises_mrt(self, synthetic_env):
        mrt_before = synthetic_env.get("mean_radiant_temperature")[0, 0]
        env2 = HeatedChair(delta_mrt=4.0).apply(synthetic_env, points=[0])
        assert env2.get("mean_radiant_temperature")[0, 0] == pytest.approx(mrt_before + 4.0)

    def test_cooled_chair_lowers_mrt(self, synthetic_env):
        mrt_before = synthetic_env.get("mean_radiant_temperature")[0, 0]
        env2 = CooledChair(delta_mrt=-4.0).apply(synthetic_env, points=[0])
        assert env2.get("mean_radiant_temperature")[0, 0] == pytest.approx(mrt_before - 4.0)

    def test_mrt_unselected_unchanged(self, synthetic_env):
        mrt_before = synthetic_env.get("mean_radiant_temperature")[5, 0]
        env2 = FootWarmer(delta_mrt=3.0).apply(synthetic_env, points=[0])
        assert env2.get("mean_radiant_temperature")[5, 0] == pytest.approx(mrt_before)

    def test_personal_heater_raises_ta(self, synthetic_env):
        ta_before = synthetic_env.get("air_temperature")[0, 0]
        env2 = PersonalHeater(delta_ta=2.0).apply(synthetic_env, points=[0])
        assert env2.get("air_temperature")[0, 0] == pytest.approx(ta_before + 2.0)


# ---------------------------------------------------------------------------
# Chained devices
# ---------------------------------------------------------------------------

class TestChaining:
    def test_fan_then_footwarmer(self, synthetic_env):
        env2 = Fan(delta_v=0.5).apply(synthetic_env, points=[0])
        env3 = FootWarmer(delta_mrt=3.0).apply(env2, points=[0])
        assert env3.get("air_velocity")[0, 0] == pytest.approx(0.6)
        mrt_base = synthetic_env.get("mean_radiant_temperature")[0, 0]
        assert env3.get("mean_radiant_temperature")[0, 0] == pytest.approx(mrt_base + 3.0)

    def test_pcs_applied_flag(self, synthetic_env):
        env2 = Fan(delta_v=0.5).apply(synthetic_env, points=[0])
        assert env2.info.get("pcs_applied") is True
        assert "pcs_applied" not in synthetic_env.info
