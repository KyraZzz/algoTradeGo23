# Copyright 2021 Optiver Asia Pacific Pty. Ltd.
#
# This file is part of Ready Trader Go.
#
#     Ready Trader Go is free software: you can redistribute it and/or
#     modify it under the terms of the GNU Affero General Public License
#     as published by the Free Software Foundation, either version 3 of
#     the License, or (at your option) any later version.
#
#     Ready Trader Go is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public
#     License along with Ready Trader Go.  If not, see
#     <https://www.gnu.org/licenses/>.
import asyncio
import itertools

from typing import List

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side


MAX_LOT_SIZE = 100
POSITION_LIMIT = 100
TICK_SIZE_IN_CENTS = 100
MIN_BID_NEAREST_TICK = (
    MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
THRESHOLD = 2e-3


class AutoTrader(BaseAutoTrader):
    """Statistical Arbitrage Naive Trading

    market_data_1: net_profit $5789.84
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.bids = set()
        self.asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0
        self.top_bid_dic = dict()
        self.top_ask_dic = dict()

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s",
                            client_order_id, error_message.decode())
        if client_order_id != 0 and (client_order_id in self.bids or client_order_id in self.asks):
            self.on_order_status_message(client_order_id, 0, 0, 0)

    def on_hedge_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your hedge orders is filled.

        The price is the average price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received hedge filled for order %d with average price %d and volume %d", client_order_id,
                         price, volume)

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """
        '''The order book message provides entry and end signal for the strategy.

        The entry signal is triggered when the spread between the two instruments is larger than the TRESHOLD.
        More specifically, two 'realistic spreads' are monitored, which are (the best bid in the future - the 
        best ask in the etf) and (the best bid in the etf - the best ask in the future).
        If either of the two spreads is larger than the TRESHOLD*the lower of the two prices, then a profitable
        entry singal is triggered. We place two orders simultaneously to effectively short the spread.

        The end signal is triggered when the spread becomes 0. We place two orders to unwind and materialise our
        profits.

        This strategy is naive because it only trades on one signal at a time. It must close one position before 
        trading on new signals.
        '''

        self.logger.info("received order book for instrument %d with sequence number %d", instrument,
                         sequence_number)
        if instrument == Instrument.ETF:
            other = Instrument.FUTURE
            self.top_bid_dic[instrument] = [
                (price, bid_volumes[idx]) for idx, price in enumerate(bid_prices)]
            self.top_ask_dic[instrument] = [
                (price, ask_volumes[idx]) for idx, price in enumerate(ask_prices)]

            f_ask_p0 = self.top_ask_dic[other][0][0]
            f_bid_p0 = self.top_bid_dic[other][0][0]
            e_ask_p0 = self.top_ask_dic[instrument][0][0]
            e_bid_p0 = self.top_bid_dic[instrument][0][0]

            # entry signal
            if self.position == 0 and other in self.top_bid_dic.keys() and other in self.top_ask_dic.keys():
                if f_bid_p0 - e_ask_p0 >= THRESHOLD * e_ask_p0:
                    # hit bid in future, take offer in etf
                    self.bid_id = next(self.order_ids)
                    volume = min(
                        self.top_ask_dic[instrument][0][1], POSITION_LIMIT)
                    self.send_insert_order(
                        self.bid_id, Side.BUY, e_ask_p0, volume, Lifespan.FILL_AND_KILL)
                    self.bids.add(self.bid_id)
                elif e_bid_p0 - f_ask_p0 >= THRESHOLD * f_ask_p0:
                    # hit bid in etf, take offer in future
                    self.ask_id = next(self.order_ids)
                    volume = min(
                        self.top_bid_dic[instrument][0][1], POSITION_LIMIT)
                    self.send_insert_order(
                        self.ask_id, Side.SELL, e_bid_p0, volume, Lifespan.FILL_AND_KILL)
                    self.asks.add(self.ask_id)

            # exit signal
            elif self.position != 0:
                volume = abs(self.position)
                # when we have long etf and we need to sell it
                if self.position > 0 and e_bid_p0 > f_ask_p0:
                    self.ask_id = next(self.order_ids)
                    self.send_insert_order(
                        self.ask_id, Side.SELL, e_bid_p0, volume, Lifespan.F)
                    self.asks.add(self.ask_id)
                # when we have short etf and we need to buy it
                elif self.position < 0 and f_bid_p0 > e_ask_p0:
                    self.bid_id = next(self.order_ids)
                    self.send_insert_order(
                        self.bid_id, Side.BUY, e_ask_p0, volume, Lifespan.F)
                    self.bids.add(self.bid_id)

        elif instrument == Instrument.FUTURE:
            self.top_bid_dic[instrument] = [
                (price, bid_volumes[idx]) for idx, price in enumerate(bid_prices)]
            self.top_ask_dic[instrument] = [
                (price, ask_volumes[idx]) for idx, price in enumerate(ask_prices)]

    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received order filled for order %d with price %d and volume %d", client_order_id,
                         price, volume)
        if client_order_id in self.bids:
            self.position += volume
            self.send_hedge_order(next(self.order_ids),
                                  Side.ASK, MIN_BID_NEAREST_TICK, volume)
        elif client_order_id in self.asks:
            self.position -= volume
            self.send_hedge_order(next(self.order_ids),
                                  Side.BID, MAX_ASK_NEAREST_TICK, volume)

    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        self.logger.info("received order status for order %d with fill volume %d remaining %d and fees %d",
                         client_order_id, fill_volume, remaining_volume, fees)
        if remaining_volume == 0:
            # if client_order_id == self.bid_id:
            #     self.bid_id = 0
            # elif client_order_id == self.ask_id:
            #     self.ask_id = 0

            # It could be either a bid or an ask
            self.bids.discard(client_order_id)
            self.asks.discard(client_order_id)

    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically when there is trading activity on the market.

        The five best ask (i.e. sell) and bid (i.e. buy) prices at which there
        has been trading activity are reported along with the aggregated volume
        traded at each of those price levels.

        If there are less than five prices on a side, then zeros will appear at
        the end of both the prices and volumes arrays.
        """
        self.logger.info("received trade ticks for instrument %d with sequence number %d", instrument,
                         sequence_number)
