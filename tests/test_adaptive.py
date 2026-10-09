"""Tests for capy.comfort.adaptive — ASHRAE 55 and EN 15251 models."""

import numpy as np
import pytest

from capy.comfort.adaptive import calculate_adaptive
from capy.io.epw import running_mean as _running_mean


def _write_epw(path, t_outdoor: np.ndarray) -> None:
    with open(path, "w") as f:
        for _ in range(8):
            f.write("HEADER,x\n")
        for v in t_outdoor:
            f.write(f"2001,1,1,1,0,x,{v:.2f},0,0\n")


# ------------------------------------------------------------------ #
# Basic shapes and values
# ------------------------------------------------------------------ #

class TestCalculateAdaptive:
    def test_no_epw_and_no_prevailing_raises(self, synthetic_env):
        with pytest.raises(ValueError, match="EPW"):
            calculate_adaptive(synthetic_env)

    def test_scalar_prevailing_no_epw(self, synthetic_env):
        result = calculate_adaptive(synthetic_env, prevailing_mean_outdoor_temp=20.0)
        assert result.values.shape == (60, 8760)
        assert result.discrete.shape == (60, 8760)
        assert set(np.unique(result.discrete)).issubset({-1, 0, 1})

    def test_t_comf_ashrae55_formula(self, synthetic_env):
        t_prev = 20.0
        result = calculate_adaptive(synthetic_env, prevailing_mean_outdoor_temp=t_prev)
        expected_t_comf = 0.31 * t_prev + 17.8
        np.testing.assert_allclose(result.t_comf, expected_t_comf, atol=1e-6)

    def test_t_comf_en15251_formula(self, synthetic_env):
        t_prev = 20.0
        result = calculate_adaptive(
            synthetic_env, standard="en15251", prevailing_mean_outdoor_temp=t_prev
        )
        expected_t_comf = 0.33 * t_prev + 18.8
        np.testing.assert_allclose(result.t_comf, expected_t_comf, atol=1e-6)

    def test_unknown_standard_raises(self, synthetic_env):
        with pytest.raises(ValueError, match="Unknown standard"):
            calculate_adaptive(synthetic_env, standard="iso7730",
                               prevailing_mean_outdoor_temp=20.0)

    def test_unknown_band_raises(self, synthetic_env):
        with pytest.raises(ValueError, match="Band"):
            calculate_adaptive(synthetic_env, acceptance_band="99%",
                               prevailing_mean_outdoor_temp=20.0)

    def test_applicability_mask_hot_prevailing(self, synthetic_env):
        # T_prevailing = 40 °C → outside ASHRAE 55 valid range (max 33.5)
        result = calculate_adaptive(synthetic_env, prevailing_mean_outdoor_temp=40.0)
        assert not result.applicability_mask.any()   # all hours invalid

    def test_applicability_mask_valid_prevailing(self, synthetic_env):
        result = calculate_adaptive(synthetic_env, prevailing_mean_outdoor_temp=22.0)
        assert result.applicability_mask.all()       # all hours valid

    def test_80pct_band_wider_than_90pct(self, synthetic_env):
        r90 = calculate_adaptive(synthetic_env, acceptance_band="90%",
                                 prevailing_mean_outdoor_temp=22.0)
        r80 = calculate_adaptive(synthetic_env, acceptance_band="80%",
                                 prevailing_mean_outdoor_temp=22.0)
        assert r80.is_comfortable().mean() >= r90.is_comfortable().mean()

    def test_en15251_class_i_narrower_than_class_iii(self, synthetic_env):
        r1 = calculate_adaptive(synthetic_env, standard="en15251",
                                acceptance_band="I", prevailing_mean_outdoor_temp=22.0)
        r3 = calculate_adaptive(synthetic_env, standard="en15251",
                                acceptance_band="III", prevailing_mean_outdoor_temp=22.0)
        assert r1.is_comfortable().mean() <= r3.is_comfortable().mean()

    def test_result_is_comfort_result(self, synthetic_env):
        from capy.comfort.base import ComfortResult
        result = calculate_adaptive(synthetic_env, prevailing_mean_outdoor_temp=20.0)
        assert isinstance(result, ComfortResult)

    def test_repr(self, synthetic_env):
        result = calculate_adaptive(synthetic_env, prevailing_mean_outdoor_temp=20.0)
        assert "adaptive" in repr(result)
        assert "±" in repr(result)

    def test_with_epw_file(self, synthetic_env, tmp_path):
        t_out = np.full(8760, 22.0)
        _write_epw(tmp_path / "t.epw", t_out)
        result = calculate_adaptive(synthetic_env, epw_path=str(tmp_path / "t.epw"))
        assert result.t_prevailing.shape == (8760,)
        np.testing.assert_allclose(result.t_prevailing[24:], 22.0, atol=0.5)

    def test_epw_from_env_attr(self, synthetic_env, tmp_path):
        t_out = np.full(8760, 20.0)
        epw_path = tmp_path / "e.epw"
        _write_epw(epw_path, t_out)
        synthetic_env.epw_path = str(epw_path)
        result = calculate_adaptive(synthetic_env)
        assert result.t_prevailing.shape == (8760,)

    def test_en16798_alias(self, synthetic_env):
        r1 = calculate_adaptive(synthetic_env, standard="en15251",
                                prevailing_mean_outdoor_temp=20.0)
        r2 = calculate_adaptive(synthetic_env, standard="en16798",
                                prevailing_mean_outdoor_temp=20.0)
        np.testing.assert_array_equal(r1.discrete, r2.discrete)


class TestRunningMean:
    def test_constant_converges(self):
        t = np.full(8760, 20.0)
        rm = _running_mean(t, alpha=0.8, n_days=7)
        np.testing.assert_allclose(rm[24:], 20.0, atol=0.5)

    def test_shape(self):
        t = np.random.randn(8760) + 20
        rm = _running_mean(t)
        assert rm.shape == (8760,)
