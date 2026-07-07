import pytest

from quantlab.costs import BpsCommission, CostModel, FixedSlippage, PerShareCommission


def test_slippage_moves_against_trader():
    slip = FixedSlippage(0.001)
    assert slip.fill_price(100.0, +1) == pytest.approx(100.10)
    assert slip.fill_price(100.0, -1) == pytest.approx(99.90)


def test_slippage_rejects_bad_side():
    with pytest.raises(ValueError):
        FixedSlippage(0.001).fill_price(100.0, 0)


def test_bps_commission():
    comm = BpsCommission(0.0001)  # 1 bp
    assert comm.charge(100, 50.0) == pytest.approx(0.50)
    assert comm.charge(-100, 50.0) == pytest.approx(0.50)


def test_per_share_commission_with_minimum():
    comm = PerShareCommission(per_share=0.005, minimum=1.0)
    assert comm.charge(10, 100.0) == pytest.approx(1.0)  # floor kicks in
    assert comm.charge(1000, 100.0) == pytest.approx(5.0)
    assert comm.charge(0, 100.0) == 0.0


def test_zero_cost_model_is_free():
    cm = CostModel.zero()
    assert cm.fill_price(100.0, 1) == 100.0
    assert cm.commission_for(1000, 100.0) == 0.0
