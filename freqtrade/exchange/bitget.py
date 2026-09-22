import logging
from datetime import datetime, timedelta

import ccxt

from freqtrade.constants import BuySell
from freqtrade.enums import OPTIMIZE_MODES, CandleType, MarginMode, PriceType, TradingMode
from freqtrade.exceptions import (
    DDosProtection,
    ExchangeError,
    InvalidOrderException,
    OperationalException,
    RetryableOrderError,
    TemporaryError,
)
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import API_RETRY_COUNT, retrier
from freqtrade.exchange.exchange_types import CcxtOrder, FtHas
from freqtrade.util import dt_from_ts, dt_now, dt_ts


logger = logging.getLogger(__name__)


class Bitget(Exchange):
    """Bitget exchange class.
    Contains adjustments needed for Freqtrade to work with this exchange.
    """

    _ft_has: FtHas = {
        "stoploss_on_exchange": True,
        "stop_price_param": "stopPrice",
        "stop_price_prop": "stopPrice",
        "stoploss_blocks_assets": False,  # Stoploss orders do not block assets
        "stoploss_order_types": {"limit": "limit", "market": "market"},
        "stoploss_query_requires_stop_flag": True,
        "ohlcv_candle_limit": 200,  # 200 for historical candles, 1000 for recent ones.
        "order_time_in_force": ["GTC", "FOK", "IOC", "PO"],
    }
    _ft_has_futures: FtHas = {
        "funding_fee_candle_limit": 100,
        "has_delisting": True,
        "stop_price_param": "stopLossPrice",
        "stop_price_prop": "stopLossPrice",
        "stop_price_type_field": "triggerType",
        "stop_price_type_value_mapping": {
            PriceType.LAST: "fill_price",
            PriceType.MARK: "mark_price",
        },
        # ccxt maps "total" to accountEquity, which includes unrealized PnL
        "balance_includes_unrealized_pnl": True,
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        (TradingMode.SPOT, MarginMode.NONE),
        (TradingMode.FUTURES, MarginMode.ISOLATED),
        # Cross is required for copytrading / hedge-mode accounts (isolated is rejected).
        (TradingMode.FUTURES, MarginMode.CROSS),
    ]

    # When True, place futures orders with CCXT hedged=True (hedge-mode account)
    # while freqtrade still keeps one open trade per pair (one-way bot behavior).
    # Needed for Bitget copytrading accounts, which only support hedge mode.
    hedge_mode: bool = False

    # Copytrading accounts reject regular set-margin-mode / set-leverage calls with
    # error 40731 ("This product does not support copy trading").
    # Once detected, skip further calls to the respective endpoint.
    _ct_margin_mode_unavailable: bool = False
    _ct_leverage_unavailable: bool = False

    def ohlcv_candle_limit(
        self, timeframe: str, candle_type: CandleType, since_ms: int | None = None
    ) -> int:
        """
        Exchange ohlcv candle limit
        bitget has the following behaviour:
        * 1000 candles for up-to-date data
        * 200 candles for historic data (prior to a certain date)
        :param timeframe: Timeframe to check
        :param candle_type: Candle-type
        :param since_ms: Starting timestamp
        :return: Candle limit as integer
        """
        timeframe_map = self._api.options["fetchOHLCV"]["maxRecentDaysPerTimeframe"]
        days = timeframe_map.get(timeframe, 30)

        if candle_type in (CandleType.FUTURES, CandleType.SPOT, CandleType.MARK) and (
            not since_ms or dt_ts(dt_now() - timedelta(days=days)) < since_ms
        ):
            return 1000

        return super().ohlcv_candle_limit(timeframe, candle_type, since_ms)

    def _convert_stop_order(self, pair: str, order_id: str, order: CcxtOrder) -> CcxtOrder:
        if order.get("status", "open") == "closed":
            # Use orderID as cliendOrderId filter to fetch the regular followup order.
            # Could be done with "fetch_order" - but clientOid as filter doesn't seem to work
            # https://www.bitget.com/api-doc/spot/trade/Get-Order-Info

            for method in (
                self._api.fetch_canceled_and_closed_orders,
                self._api.fetch_open_orders,
            ):
                orders = method(pair)
                orders_f = [order for order in orders if order["clientOrderId"] == order_id]
                if orders_f:
                    order_reg = orders_f[0]
                    self._log_exchange_response("fetch_stoploss_order1", order_reg)
                    order_reg["id_stop"] = order_reg["id"]
                    order_reg["id"] = order_id
                    order_reg["type"] = "stoploss"
                    order_reg["status_stop"] = "triggered"
                    return order_reg
        order = self._order_contracts_to_amount(order)
        order["type"] = "stoploss"
        return order

    def _fetch_stop_order_fallback(self, order_id: str, pair: str) -> CcxtOrder:
        # old stoploss orders
        paramsold = {"stop": True}
        # new stoploss orders with stopLossPrice (used in futures starting 2026.4)
        paramsnew = {"planType": "profit_loss"}
        params_to_try = (
            (paramsnew, paramsold) if self.trading_mode == TradingMode.FUTURES else (paramsold,)
        )

        for params2 in params_to_try:
            for method in (
                self._api.fetch_open_orders,
                self._api.fetch_canceled_and_closed_orders,
            ):
                try:
                    orders = method(pair, params=params2)
                    orders_f = [order for order in orders if order["id"] == order_id]
                    if orders_f:
                        order = orders_f[0]
                        self._log_exchange_response("get_stop_order_fallback", order)
                        return self._convert_stop_order(pair, order_id, order)
                except (ccxt.OrderNotFound, ccxt.InvalidOrder):
                    pass
                except ccxt.DDoSProtection as e:
                    raise DDosProtection(e) from e
                except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
                    raise TemporaryError(
                        f"Could not get order due to {e.__class__.__name__}. Message: {e}"
                    ) from e
                except ccxt.BaseError as e:
                    raise OperationalException(e) from e
        raise RetryableOrderError(f"StoplossOrder not found (pair: {pair} id: {order_id}).")

    @retrier(retries=API_RETRY_COUNT)
    def fetch_stoploss_order(
        self, order_id: str, pair: str, params: dict | None = None
    ) -> CcxtOrder:
        if self._config["dry_run"]:
            return self.fetch_dry_run_order(order_id)

        return self._fetch_stop_order_fallback(order_id, pair)

    def cancel_stoploss_order(self, order_id: str, pair: str, params: dict | None = None) -> dict:
        cancel_params = params.copy() if params else {}
        cancel_params["stop"] = True

        if self.trading_mode != TradingMode.FUTURES:
            return self.cancel_order(order_id, pair, cancel_params)

        try:
            return self.cancel_order(order_id, pair, {**cancel_params, "planType": "pos_loss"})
        except (InvalidOrderException, IndexError):
            # Keep compatibility with stoploss orders created by older versions.
            return self.cancel_order(order_id, pair, cancel_params)

    @retrier
    def additional_exchange_init(self) -> None:
        """
        Additional exchange initialization logic.
        .api will be available at this point.
        Must be overridden in child methods if required.
        """
        self.hedge_mode = bool(self._config["exchange"].get("hedge_mode", False))
        if self.hedge_mode and self.trading_mode == TradingMode.FUTURES:
            # Elite / copytrading accounts reject isolated ("fixedMargin") with error 25200
            # and only support cross margin + hedge mode.
            # https://www.bitget.com/api-doc/uta/copy/Elite-Trading-API-Guide
            if self.margin_mode == MarginMode.ISOLATED:
                logger.warning(
                    "Bitget: copytrading / hedge_mode accounts only support cross margin "
                    "(isolated is rejected with 25200). Overriding margin_mode to cross."
                )
                self.margin_mode = MarginMode.CROSS
                self._config["margin_mode"] = MarginMode.CROSS
        try:
            if not self._config["dry_run"]:
                if self.trading_mode == TradingMode.FUTURES:
                    if self.hedge_mode:
                        # Copytrading / hedge-only accounts cannot switch to one-way mode.
                        # Assume the account is already in hedge mode and tag orders accordingly.
                        logger.info(
                            "Bitget: hedge_mode enabled. Using hedge-mode order params "
                            "(cross margin); freqtrade still opens only one position side per pair."
                        )
                    else:
                        position_mode = self._api.set_position_mode(False)
                        self._log_exchange_response("set_position_mode", position_mode)
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"Error in additional_exchange_init due to {e.__class__.__name__}. Message: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _get_params(
        self,
        side: BuySell,
        ordertype: str,
        leverage: float,
        reduceOnly: bool,
        time_in_force: str = "GTC",
    ) -> dict:
        params = super()._get_params(
            side=side,
            ordertype=ordertype,
            leverage=leverage,
            reduceOnly=reduceOnly,
            time_in_force=time_in_force,
        )
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode:
            if self.margin_mode == MarginMode.ISOLATED:
                params["marginMode"] = self.margin_mode.value.lower()
            # CROSS: do not pass marginMode.
            # Bitget's enum is "isolated" | "crossed". Freqtrade's value is "cross".
            # ccxt maps that to "crossed", then merges leftover params back onto the
            # request, so "cross" overwrites it and Bitget returns 40034
            # ("Parameter cross does not exist"). Omitting the param lets ccxt default
            # to crossed.
            if self.hedge_mode:
                # CCXT maps hedged + reduceOnly to tradeSide open/close and posSide.
                params["hedged"] = True
        return params

    def _get_stop_params(self, side: BuySell, ordertype: str, stop_price: float) -> dict:
        params = super()._get_stop_params(side, ordertype, stop_price)
        if self.trading_mode == TradingMode.FUTURES and self.hedge_mode:
            params["hedged"] = True
        return params

    def _order_contracts_to_amount(self, order: CcxtOrder) -> CcxtOrder:
        order = super()._order_contracts_to_amount(order)
        return self._fill_missing_order_price(order)

    @staticmethod
    def _fill_missing_order_price(order: CcxtOrder) -> CcxtOrder:
        """
        Bitget market / hedge-mode close orders often come back with empty price and
        priceAvg. Recover average/price/cost from info or cost/filled so persistence
        can store ft_price.
        """
        if order.get("price") or order.get("average"):
            return order

        info = order.get("info") or {}
        raw_avg = info.get("priceAvg") or info.get("fillPrice") or info.get("priceAvgPx")
        try:
            avg = float(raw_avg) if raw_avg not in (None, "") else None
        except (TypeError, ValueError):
            avg = None

        filled = order.get("filled") or 0.0
        cost = order.get("cost")
        if not cost:
            raw_cost = info.get("quoteVolume") or info.get("quoteSize")
            try:
                cost = float(raw_cost) if raw_cost not in (None, "") else None
            except (TypeError, ValueError):
                cost = None
            if cost:
                order["cost"] = cost

        price = avg or ((cost / filled) if cost and filled else None)
        if price:
            if not order.get("average"):
                order["average"] = price
            if not order.get("price"):
                order["price"] = price
        return order

    def _lev_prep(self, pair: str, leverage: float, side: BuySell, accept_fail: bool = False):
        if self.trading_mode == TradingMode.FUTURES and self.hedge_mode:
            if not self._ct_margin_mode_unavailable:
                try:
                    self.set_margin_mode(pair, self.margin_mode, accept_fail)
                except TemporaryError as e:
                    if "40731" not in str(e):
                        raise
                    # Copytrading accounts don't support changing the margin mode.
                    self._ct_margin_mode_unavailable = True
                    logger.warning(
                        "Bitget: This account does not allow setting the margin mode via API "
                        "(copytrading). The account must stay in cross margin mode."
                    )
            self._set_leverage(leverage, pair, accept_fail)
            return
        super()._lev_prep(pair, leverage, side, accept_fail)

    def _set_leverage(
        self,
        leverage: float,
        pair: str | None = None,
        accept_fail: bool = False,
        params: dict | None = None,
    ):
        if self.trading_mode == TradingMode.FUTURES and self.hedge_mode:
            if self._ct_leverage_unavailable:
                return
            # In isolated hedge mode, bitget requires either "holdSide",
            # or both longLeverage and shortLeverage to be set at once
            # (in which case holdSide is not needed).
            # Set both sides to the same leverage to keep one call per entry.
            # https://www.bitget.com/api-doc/contract/account/Change-Leverage
            leverage_str = f"{leverage:g}"
            params = {
                **(params or {}),
                "longLeverage": leverage_str,
                "shortLeverage": leverage_str,
            }
            try:
                super()._set_leverage(leverage, pair, accept_fail, params)
            except TemporaryError as e:
                if "40731" not in str(e):
                    raise
                # Copytrading accounts don't support changing leverage via this endpoint.
                self._ct_leverage_unavailable = True
                logger.warning(
                    "Bitget: This account does not allow setting leverage via API "
                    "(copytrading). Freqtrade will not set leverage on the exchange - "
                    "make sure the leverage configured on bitget matches the leverage "
                    "used by your strategy."
                )
            return
        super()._set_leverage(leverage, pair, accept_fail, params)

    def get_funding_fees(
        self, pair: str, amount: float, is_short: bool, open_date: datetime
    ) -> float:
        """
        Copytrading accounts reject fetch_funding_history with error 40731.
        Calculate fees from public funding-rate / mark-price history instead.
        """
        if (
            self.trading_mode == TradingMode.FUTURES
            and self.hedge_mode
            and not self._config["dry_run"]
        ):
            try:
                return self._fetch_and_calculate_funding_fees(pair, amount, is_short, open_date)
            except ExchangeError:
                logger.warning(f"Could not update funding fees for {pair}.")
                return 0.0
        return super().get_funding_fees(pair, amount, is_short, open_date)

    def dry_run_liquidation_price(
        self,
        pair: str,
        open_rate: float,
        is_short: bool,
        amount: float,
        stake_amount: float,
        leverage: float,
        wallet_balance: float,
        open_trades: list,
    ) -> float | None:
        """
        Important: Must be fetching data from cached values as this is used by backtesting!


        https://www.bitget.com/support/articles/12560603808759
        MMR: Maintenance margin rate of the trading pair.

        CoinMainIndexPrice: The index price for Coin-M futures. For USDT-M futures,
                            the index price is: 1.

        TakerFeeRatio: The fee rate applied when placing taker orders.

        Position direction: The current position direction of the trading pair.
                        1 indicates a long position, and -1 indicates a short position.

        Formula:

        Estimated liquidation price = [
            position margin - position size x average entry price x position direction
        ] ÷ [position size x (MMR + TakerFeeRatio - position direction)]

        :param pair: Pair to calculate liquidation price for
        :param open_rate: Entry price of position
        :param is_short: True if the trade is a short, false otherwise
        :param amount: Absolute value of position size incl. leverage (in base currency)
        :param stake_amount: Stake amount - Collateral in settle currency.
        :param leverage: Leverage used for this position.
        :param wallet_balance: Amount of margin_mode in the wallet being used to trade
            Cross-Margin Mode: crossWalletBalance
            Isolated-Margin Mode: isolatedWalletBalance
        :param open_trades: List of other open trades in the same wallet
        """
        market = self.markets[pair]
        taker_fee_rate = market["taker"] or self._api.describe().get("fees", {}).get(
            "trading", {}
        ).get("taker", 0.001)
        mm_ratio, _ = self.get_maintenance_ratio_and_amt(pair, stake_amount)

        if self.trading_mode == TradingMode.FUTURES and self.margin_mode in (
            MarginMode.ISOLATED,
            MarginMode.CROSS,
        ):
            # Isolated: wallet_balance is position margin.
            # Cross: wallet_balance is account equity. Same formula; live trading uses
            # the exchange-provided liquidationPrice from fetch_positions instead.
            position_direction = -1 if is_short else 1

            return (wallet_balance - (amount * open_rate * position_direction)) / (
                amount * (mm_ratio + taker_fee_rate - position_direction)
            )
        else:
            raise OperationalException(
                "Freqtrade currently only supports isolated or cross futures for bitget"
            )

    def check_delisting_time(self, pair: str) -> datetime | None:
        """
        Check if the pair gonna be delisted.
        By default, it returns None.
        :param pair: Market symbol
        :return: Datetime if the pair gonna be delisted, None otherwise
        """
        if self._config["runmode"] in OPTIMIZE_MODES:
            return None

        if self.trading_mode == TradingMode.FUTURES:
            return self._check_delisting_futures(pair)
        return None

    def _check_delisting_futures(self, pair: str) -> datetime | None:
        delivery_time = self.markets.get(pair, {}).get("info", {}).get("limitOpenTime", None)
        if delivery_time:
            if isinstance(delivery_time, str) and (delivery_time != ""):
                delivery_time = int(delivery_time)

            if not isinstance(delivery_time, int) or delivery_time <= 0:
                return None

            max_delivery = dt_ts() + (
                14 * 24 * 60 * 60 * 1000
            )  # Assume exchange don't announce delisting more than 14 days in advance

            if delivery_time < max_delivery:
                return dt_from_ts(delivery_time)

        return None
