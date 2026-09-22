from copy import deepcopy
from datetime import timedelta
from unittest.mock import MagicMock, PropertyMock

import ccxt
import pytest

from freqtrade.enums import CandleType, MarginMode, RunMode, TradingMode
from freqtrade.exceptions import InvalidOrderException, RetryableOrderError, TemporaryError
from freqtrade.exchange.common import API_RETRY_COUNT
from freqtrade.util import dt_now, dt_ts, dt_utc
from tests.conftest import EXMS, get_patched_exchange, log_has_re
from tests.exchange.test_exchange import ccxt_exceptionhandlers


@pytest.mark.usefixtures("init_persistence")
def test_fetch_stoploss_order_bitget(default_conf, mocker):
    default_conf["dry_run"] = False
    mocker.patch("freqtrade.exchange.common.time.sleep")
    api_mock = MagicMock()

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="bitget")

    api_mock.fetch_open_orders = MagicMock(return_value=[])
    api_mock.fetch_canceled_and_closed_orders = MagicMock(return_value=[])

    with pytest.raises(RetryableOrderError):
        exchange.fetch_stoploss_order("1234", "ETH/BTC")
    assert api_mock.fetch_open_orders.call_count == API_RETRY_COUNT + 1
    assert api_mock.fetch_canceled_and_closed_orders.call_count == API_RETRY_COUNT + 1

    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_canceled_and_closed_orders.reset_mock()

    api_mock.fetch_canceled_and_closed_orders = MagicMock(
        return_value=[{"id": "1234", "status": "closed", "clientOrderId": "123455"}]
    )
    api_mock.fetch_open_orders = MagicMock(return_value=[{"id": "50110", "clientOrderId": "1234"}])

    resp = exchange.fetch_stoploss_order("1234", "ETH/BTC")
    assert api_mock.fetch_open_orders.call_count == 2
    assert api_mock.fetch_canceled_and_closed_orders.call_count == 2

    assert resp["id"] == "1234"
    assert resp["id_stop"] == "50110"
    assert resp["type"] == "stoploss"

    default_conf["dry_run"] = True
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="bitget")
    dro_mock = mocker.patch(f"{EXMS}.fetch_dry_run_order", MagicMock(return_value={"id": "123455"}))

    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_canceled_and_closed_orders.reset_mock()
    resp = exchange.fetch_stoploss_order("1234", "ETH/BTC")

    assert api_mock.fetch_open_orders.call_count == 0
    assert api_mock.fetch_canceled_and_closed_orders.call_count == 0
    assert dro_mock.call_count == 1


def test_fetch_stoploss_order_bitget_exceptions(default_conf_usdt, mocker):
    default_conf_usdt["dry_run"] = False
    api_mock = MagicMock()

    # Test emulation of the stoploss getters
    api_mock.fetch_canceled_and_closed_orders = MagicMock(return_value=[])

    ccxt_exceptionhandlers(
        mocker,
        default_conf_usdt,
        api_mock,
        "bitget",
        "fetch_stoploss_order",
        "fetch_open_orders",
        retries=API_RETRY_COUNT + 1,
        order_id="12345",
        pair="ETH/USDT",
    )


@pytest.mark.usefixtures("init_persistence")
def test_cancel_stoploss_order_bitget(default_conf_usdt, mocker):
    default_conf_usdt["dry_run"] = False
    api_mock = MagicMock()

    exchange = get_patched_exchange(mocker, default_conf_usdt, api_mock, exchange="bitget")

    # Spot scenario
    exchange.cancel_order = MagicMock(return_value={"id": "1234"})
    assert exchange.cancel_stoploss_order("1234", "ETH/USDT", {}) == {"id": "1234"}
    assert exchange.cancel_order.call_count == 1
    exchange.cancel_order.assert_called_once_with("1234", "ETH/USDT", {"stop": True})

    # Futures scenario
    default_conf_usdt["trading_mode"] = TradingMode.FUTURES
    default_conf_usdt["margin_mode"] = MarginMode.ISOLATED
    exchange = get_patched_exchange(mocker, default_conf_usdt, api_mock, exchange="bitget")
    exchange.cancel_order = MagicMock(return_value={"id": "1234"})
    assert exchange.cancel_stoploss_order("1234", "ETH/USDT:USDT", {}) == {"id": "1234"}
    assert exchange.cancel_order.call_count == 1
    exchange.cancel_order.assert_called_once_with(
        "1234", "ETH/USDT:USDT", {"stop": True, "planType": "pos_loss"}
    )

    exchange.cancel_order = MagicMock(
        side_effect=[InvalidOrderException("API error"), {"id": "1234"}]
    )
    assert exchange.cancel_stoploss_order("1234", "ETH/USDT:USDT", {}) == {"id": "1234"}
    assert exchange.cancel_order.call_count == 2
    exchange.cancel_order.assert_any_call(
        "1234", "ETH/USDT:USDT", {"stop": True, "planType": "pos_loss"}
    )
    exchange.cancel_order.assert_any_call("1234", "ETH/USDT:USDT", {"stop": True})

    # Position close auto-cancels TPSL; Bitget returns 25575 instead of success.
    exchange.cancel_order = MagicMock(
        side_effect=InvalidOrderException(
            'Bitget plan/stoploss 1234 already gone. Message: bitget {"code":"25575"}'
        )
    )
    gone = exchange.cancel_stoploss_order("1234", "ETH/USDT:USDT", {})
    assert gone["id"] == "1234"
    assert gone["status"] == "canceled"


def test_cancel_order_bitget_stoploss_already_gone(default_conf_usdt, mocker):
    default_conf_usdt["dry_run"] = False
    default_conf_usdt["trading_mode"] = TradingMode.FUTURES
    default_conf_usdt["margin_mode"] = MarginMode.ISOLATED
    api_mock = MagicMock()
    api_mock.cancel_order = MagicMock(
        side_effect=ccxt.ExchangeError('bitget {"code":"25575","msg":"Failed to stop the strategy"}')
    )
    exchange = get_patched_exchange(mocker, default_conf_usdt, api_mock, exchange="bitget")

    with pytest.raises(InvalidOrderException, match="already gone"):
        exchange.cancel_order("1486234472269987906", "WLD/USDT:USDT", {"stop": True})
    # Must not retry 25575.
    assert api_mock.cancel_order.call_count == 1


def test_bitget_ohlcv_candle_limit(mocker, default_conf_usdt):
    # This test is also a live test - so we're sure our limits are correct.
    api_mock = MagicMock()
    api_mock.options = {
        "fetchOHLCV": {
            "maxRecentDaysPerTimeframe": {
                "1m": 30,
                "5m": 30,
                "15m": 30,
                "30m": 30,
                "1h": 60,
                "4h": 60,
                "1d": 60,
            }
        }
    }

    exch = get_patched_exchange(mocker, default_conf_usdt, api_mock, exchange="bitget")
    timeframes = ("1m", "5m", "1h")

    for timeframe in timeframes:
        assert exch.ohlcv_candle_limit(timeframe, CandleType.SPOT) == 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUTURES) == 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.MARK) == 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE) == 200

        start_time = dt_ts(dt_now() - timedelta(days=17))
        assert exch.ohlcv_candle_limit(timeframe, CandleType.SPOT, start_time) == 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUTURES, start_time) == 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.MARK, start_time) == 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE, start_time) == 200
        start_time = dt_ts(dt_now() - timedelta(days=48))
        length = 200 if timeframe in ("1m", "5m") else 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.SPOT, start_time) == length
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUTURES, start_time) == length
        assert exch.ohlcv_candle_limit(timeframe, CandleType.MARK, start_time) == length
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE, start_time) == 200

        start_time = dt_ts(dt_now() - timedelta(days=61))
        length = 200
        assert exch.ohlcv_candle_limit(timeframe, CandleType.SPOT, start_time) == length
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUTURES, start_time) == length
        assert exch.ohlcv_candle_limit(timeframe, CandleType.MARK, start_time) == length
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE, start_time) == 200


def test_additional_exchange_init_bitget(default_conf, mocker):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    api_mock = MagicMock()
    api_mock.set_position_mode = MagicMock(return_value={})

    exchange = get_patched_exchange(mocker, default_conf, exchange="bitget", api_mock=api_mock)
    assert api_mock.set_position_mode.call_count == 1
    assert api_mock.set_position_mode.call_args[0][0] is False
    assert exchange.hedge_mode is False

    ccxt_exceptionhandlers(
        mocker, default_conf, api_mock, "bitget", "additional_exchange_init", "set_position_mode"
    )


def test_additional_exchange_init_bitget_hedge_mode(default_conf, mocker):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    default_conf["exchange"]["hedge_mode"] = True
    api_mock = MagicMock()
    api_mock.set_position_mode = MagicMock(return_value={})

    exchange = get_patched_exchange(mocker, default_conf, exchange="bitget", api_mock=api_mock)
    assert api_mock.set_position_mode.call_count == 0
    assert exchange.hedge_mode is True
    # Isolated is not supported on copytrading accounts; hedge_mode forces cross.
    assert exchange.margin_mode == MarginMode.CROSS
    assert default_conf["margin_mode"] == MarginMode.CROSS


def test__get_params_bitget_hedge_mode(default_conf, mocker):
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    default_conf["exchange"]["hedge_mode"] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange="bitget")

    params = exchange._get_params(
        side="buy",
        ordertype="limit",
        leverage=3.0,
        reduceOnly=False,
        time_in_force="GTC",
    )
    assert "marginMode" not in params
    assert params["hedged"] is True
    assert "reduceOnly" not in params

    params_exit = exchange._get_params(
        side="sell",
        ordertype="market",
        leverage=3.0,
        reduceOnly=True,
        time_in_force="GTC",
    )
    assert params_exit["hedged"] is True
    assert params_exit["reduceOnly"] is True
    assert "marginMode" not in params_exit

    stop_params = exchange._get_stop_params(side="sell", ordertype="market", stop_price=100.0)
    assert stop_params["hedged"] is True


def test__set_leverage_bitget_hedge_mode(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.set_leverage = MagicMock(return_value={})
    type(api_mock).has = PropertyMock(return_value={"setLeverage": True})
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED

    # Default (one-way) mode - no extra params
    exchange = get_patched_exchange(mocker, default_conf, exchange="bitget", api_mock=api_mock)
    exchange._set_leverage(3.0, "ETH/USDT:USDT")
    assert api_mock.set_leverage.call_count == 1
    assert api_mock.set_leverage.call_args[1]["params"] == {}

    # Hedge mode - both sides are set at once, so holdSide is not required.
    api_mock.set_leverage.reset_mock()
    default_conf["exchange"]["hedge_mode"] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange="bitget", api_mock=api_mock)
    exchange._set_leverage(3.0, "ETH/USDT:USDT")
    assert api_mock.set_leverage.call_count == 1
    assert api_mock.set_leverage.call_args[1]["symbol"] == "ETH/USDT:USDT"
    assert api_mock.set_leverage.call_args[1]["leverage"] == 3.0
    assert api_mock.set_leverage.call_args[1]["params"] == {
        "longLeverage": "3",
        "shortLeverage": "3",
    }

    # Fractional leverage is passed through unchanged
    api_mock.set_leverage.reset_mock()
    exchange._set_leverage(2.5, "ETH/USDT:USDT")
    assert api_mock.set_leverage.call_args[1]["params"] == {
        "longLeverage": "2.5",
        "shortLeverage": "2.5",
    }

    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        "bitget",
        "_set_leverage",
        "set_leverage",
        pair="ETH/USDT:USDT",
        leverage=5.0,
    )


def test__lev_prep_bitget_hedge_mode_copytrading(default_conf, mocker, caplog):
    api_mock = MagicMock()
    api_mock.set_margin_mode = MagicMock(return_value={})
    api_mock.set_leverage = MagicMock(return_value={})
    type(api_mock).has = PropertyMock(return_value={"setMarginMode": True, "setLeverage": True})
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    default_conf["exchange"]["hedge_mode"] = True

    # Hedge-mode account that allows both calls
    exchange = get_patched_exchange(mocker, default_conf, exchange="bitget", api_mock=api_mock)
    exchange._lev_prep("ETH/USDT:USDT", 3.0, "buy")
    assert api_mock.set_margin_mode.call_count == 1
    assert api_mock.set_leverage.call_count == 1
    assert exchange._ct_margin_mode_unavailable is False
    assert exchange._ct_leverage_unavailable is False

    # Copytrading account rejects both endpoints with error 40731
    copytrading_error = ccxt.ExchangeError(
        'bitget {"code":"40731","msg":"This product does not support copy trading"}'
    )
    api_mock.set_margin_mode = MagicMock(side_effect=copytrading_error)
    api_mock.set_leverage = MagicMock(side_effect=copytrading_error)
    exchange = get_patched_exchange(mocker, default_conf, exchange="bitget", api_mock=api_mock)

    # Must not raise - trading continues without setting margin-mode / leverage.
    exchange._lev_prep("ETH/USDT:USDT", 3.0, "buy")
    assert exchange._ct_margin_mode_unavailable is True
    assert exchange._ct_leverage_unavailable is True
    assert log_has_re(r"Bitget: This account does not allow setting the margin mode.*", caplog)
    assert log_has_re(r"Bitget: This account does not allow setting leverage.*", caplog)
    # The retrier exhausts its retries before the endpoint is flagged as unavailable.
    assert api_mock.set_margin_mode.call_count == API_RETRY_COUNT + 1
    assert api_mock.set_leverage.call_count == API_RETRY_COUNT + 1

    # Subsequent entries skip the unavailable endpoints entirely.
    exchange._lev_prep("ETH/USDT:USDT", 3.0, "buy")
    assert api_mock.set_margin_mode.call_count == API_RETRY_COUNT + 1
    assert api_mock.set_leverage.call_count == API_RETRY_COUNT + 1

    # Other errors are still raised.
    api_mock.set_margin_mode = MagicMock(side_effect=ccxt.ExchangeError("Some other error"))
    exchange = get_patched_exchange(mocker, default_conf, exchange="bitget", api_mock=api_mock)
    with pytest.raises(TemporaryError):
        exchange._lev_prep("ETH/USDT:USDT", 3.0, "buy")


def test_get_funding_fees_bitget_hedge_mode(default_conf, mocker):
    now = dt_now()
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    api_mock = MagicMock()
    api_mock.fetch_funding_history = MagicMock(return_value=[{"amount": 0.5}])
    type(api_mock).has = PropertyMock(return_value={"fetchFundingHistory": True})

    # Regular one-way account uses the private funding-history endpoint.
    exchange = get_patched_exchange(mocker, default_conf, exchange="bitget", api_mock=api_mock)
    calc = mocker.patch.object(exchange, "_fetch_and_calculate_funding_fees", return_value=0.1)
    assert exchange.get_funding_fees("ETH/USDT:USDT", 1.0, False, now) == 0.5
    assert api_mock.fetch_funding_history.call_count == 1
    assert calc.call_count == 0

    # Copytrading / hedge-mode: private history is rejected (40731), so use public rates.
    default_conf["exchange"]["hedge_mode"] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange="bitget", api_mock=api_mock)
    calc = mocker.patch.object(exchange, "_fetch_and_calculate_funding_fees", return_value=0.25)
    api_mock.fetch_funding_history.reset_mock()
    assert exchange.get_funding_fees("ETH/USDT:USDT", 1.0, False, now) == 0.25
    assert api_mock.fetch_funding_history.call_count == 0
    assert calc.call_count == 1


def test_fill_missing_order_price_bitget(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf, exchange="bitget")

    filled = {
        "id": "1",
        "price": None,
        "average": None,
        "filled": 44.0,
        "cost": None,
        "info": {"priceAvg": "2.15", "quoteVolume": "94.6"},
    }
    filled = exchange._fill_missing_order_price(filled)
    assert filled["average"] == 2.15
    assert filled["price"] == 2.15

    from_cost = {
        "id": "2",
        "price": None,
        "average": None,
        "filled": 10.0,
        "cost": 25.0,
        "info": {},
    }
    from_cost = exchange._fill_missing_order_price(from_cost)
    assert from_cost["price"] == 2.5
    assert from_cost["average"] == 2.5

    untouched = {"id": "3", "price": 1.2, "average": None, "filled": 1, "info": {}}
    assert exchange._fill_missing_order_price(untouched)["price"] == 1.2


def test_normalize_ccxt_order_bitget_fee_and_hedge_side(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf, exchange="bitget")

    zero_fee = {
        "id": "1",
        "side": "buy",
        "price": 1.0,
        "fee": {"cost": 0.0, "currency": "USDT"},
        "info": {"fee": "0", "posMode": "hedge_mode", "tradeSide": "close", "side": "buy"},
    }
    zero_fee = exchange._normalize_ccxt_order(zero_fee)
    assert zero_fee["fee"] is None
    assert zero_fee["side"] == "sell"
    assert zero_fee["reduceOnly"] is True

    real_fee = {
        "id": "2",
        "side": "buy",
        "price": 1.0,
        "fee": {"cost": 0.0, "currency": "USDT"},
        "info": {"fee": "-0.012", "marginCoin": "USDT", "posMode": "one_way_mode"},
    }
    real_fee = exchange._normalize_ccxt_order(real_fee)
    assert real_fee["fee"]["cost"] == 0.012
    assert real_fee["fee"]["currency"] == "USDT"
    assert real_fee["side"] == "buy"

    neg_parsed = {
        "id": "3",
        "side": "sell",
        "price": 1.0,
        "fee": {"cost": -0.01969, "currency": "USDT"},
        "info": {"fee": "0"},
    }
    neg_parsed = exchange._normalize_ccxt_order(neg_parsed)
    assert neg_parsed["fee"]["cost"] == 0.01969


def test_get_trades_for_order_bitget_hedge_mode_skips_private_history(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.fetch_my_trades = MagicMock(return_value=[{"order": "1"}])
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    default_conf["exchange"]["hedge_mode"] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange="bitget", api_mock=api_mock)

    assert exchange.get_trades_for_order("1", "ETH/USDT:USDT", dt_now()) == []
    assert api_mock.fetch_my_trades.call_count == 0


def test_dry_run_liquidation_price_cross_bitget(default_conf, mocker):
    default_conf["dry_run"] = True
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.CROSS
    api_mock = MagicMock()
    mocker.patch(f"{EXMS}.get_maintenance_ratio_and_amt", MagicMock(return_value=(0.005, 0.0)))
    exchange = get_patched_exchange(mocker, default_conf, exchange="bitget", api_mock=api_mock)

    liq = exchange.dry_run_liquidation_price(
        "ETH/USDT:USDT",
        100_000,
        False,
        0.1,
        100,
        10,
        100,
        [],
    )
    assert liq is not None
    assert isinstance(liq, float)


def test__lev_prep_bitget(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.set_margin_mode = MagicMock()
    api_mock.set_leverage = MagicMock()
    type(api_mock).has = PropertyMock(return_value={"setMarginMode": True, "setLeverage": True})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="bitget")
    exchange._lev_prep("BTC/USDC:USDC", 3.2, "buy")

    assert api_mock.set_margin_mode.call_count == 0
    assert api_mock.set_leverage.call_count == 0

    # test in futures mode
    api_mock.set_margin_mode.reset_mock()
    api_mock.set_leverage.reset_mock()
    default_conf["dry_run"] = False

    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="bitget")
    exchange._lev_prep("BTC/USDC:USDC", 3.2, "buy")

    assert api_mock.set_margin_mode.call_count == 1
    assert api_mock.set_leverage.call_count == 1
    api_mock.set_leverage.assert_called_with(symbol="BTC/USDC:USDC", leverage=3.2, params={})

    api_mock.reset_mock()

    exchange._lev_prep("BTC/USDC:USDC", 19.99, "sell")

    assert api_mock.set_margin_mode.call_count == 1
    assert api_mock.set_leverage.call_count == 1
    api_mock.set_leverage.assert_called_with(symbol="BTC/USDC:USDC", leverage=19.99, params={})


def test_check_delisting_time_bitget(default_conf_usdt, mocker):
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange="bitget")
    exchange._config["runmode"] = RunMode.BACKTEST
    delist_fut_mock = MagicMock(return_value=None)
    mocker.patch.object(exchange, "_check_delisting_futures", delist_fut_mock)

    # Invalid run mode
    resp = exchange.check_delisting_time("BTC/USDT")
    assert resp is None
    assert delist_fut_mock.call_count == 0

    # Delist spot called
    exchange._config["runmode"] = RunMode.DRY_RUN
    resp1 = exchange.check_delisting_time("BTC/USDT")
    assert resp1 is None
    assert delist_fut_mock.call_count == 0

    # Delist futures called
    exchange.trading_mode = TradingMode.FUTURES
    resp1 = exchange.check_delisting_time("BTC/USDT:USDT")
    assert resp1 is None
    assert delist_fut_mock.call_count == 1


def test__check_delisting_futures_bitget(default_conf_usdt, mocker, markets):
    markets["BTC/USDT:USDT"] = deepcopy(markets["SOL/BUSD:BUSD"])
    markets["BTC/USDT:USDT"]["info"]["limitOpenTime"] = "-1"
    markets["SOL/BUSD:BUSD"]["info"]["limitOpenTime"] = "-1"
    markets["ADA/USDT:USDT"]["info"]["limitOpenTime"] = "1760745600000"  # 2025-10-18
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange="bitget")
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))

    resp_sol = exchange._check_delisting_futures("SOL/BUSD:BUSD")
    # No delisting date
    assert resp_sol is None
    # Has a delisting date
    resp_ada = exchange._check_delisting_futures("ADA/USDT:USDT")
    assert resp_ada == dt_utc(2025, 10, 18)
