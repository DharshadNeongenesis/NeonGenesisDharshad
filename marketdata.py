from __future__ import annotations

from collections import deque
from math import sqrt
from typing import Deque

from orderbook import OrderBook
from trade import Trade


class MarketData:
    """
    Maintains market statistics generated from
    executed trades and the order book.
    """

    def __init__(self, history_size: int = 1000) -> None:
        if history_size <= 0:
            raise ValueError(
                "History size must be greater than zero."
            )

        self._price_history: Deque[float] = deque(
            maxlen=history_size
        )

        self._trade_history: Deque[Trade] = deque(
            maxlen=history_size
        )

        self._total_volume = 0
        self._total_trades = 0

    @property
    def last_price(self) -> float | None:
        """Return the most recently traded price."""

        if not self._price_history:
            return None

        return self._price_history[-1]

    @property
    def total_volume(self) -> int:
        """Return the total traded quantity."""

        return self._total_volume

    @property
    def total_trades(self) -> int:
        """Return the total number of executed trades."""

        return self._total_trades

    @property
    def price_history(self) -> tuple[float, ...]:
        """Return historical traded prices."""

        return tuple(self._price_history)

    @property
    def trade_history(self) -> tuple[Trade, ...]:
        """Return recorded trades."""

        return tuple(self._trade_history)

    def update_from_trade(self, trade: Trade) -> None:
        """
        Record an executed trade and update
        market statistics.
        """

        self._price_history.append(trade.price)
        self._trade_history.append(trade)

        self._total_volume += trade.quantity
        self._total_trades += 1

    def volatility(self) -> float:
        """
        Calculate historical price volatility
        using standard deviation of simple returns.

        Returns 0.0 when insufficient price data exists.
        """

        if len(self._price_history) < 2:
            return 0.0

        prices = list(self._price_history)

        returns: list[float] = []

        for previous, current in zip(
            prices,
            prices[1:],
        ):
            if previous <= 0:
                continue

            returns.append(
                (current - previous) / previous
            )

        if len(returns) < 2:
            return 0.0

        mean = sum(returns) / len(returns)

        variance = sum(
            (value - mean) ** 2
            for value in returns
        ) / len(returns)

        return sqrt(variance)

    def order_book_imbalance(
        self,
        order_book: OrderBook,
    ) -> float:
        """
        Calculate top-of-book imbalance.

        Returns a value between -1.0 and +1.0.

        Positive:
            More bid quantity.

        Negative:
            More ask quantity.
        """

        bid = order_book.best_bid
        ask = order_book.best_ask

        bid_quantity = (
            bid.quantity
            if bid is not None
            else 0
        )

        ask_quantity = (
            ask.quantity
            if ask is not None
            else 0
        )

        total_quantity = (
            bid_quantity + ask_quantity
        )

        if total_quantity == 0:
            return 0.0

        return (
            bid_quantity - ask_quantity
        ) / total_quantity

    def snapshot(
        self,
        order_book: OrderBook,
    ) -> dict[str, float | int | None]:
        """
        Return the current market state.

        This snapshot will eventually become one
        of the primary inputs to the AI layer.
        """

        return {
            "last_price": self.last_price,
            "spread": order_book.spread,
            "volume": self.total_volume,
            "trades": self.total_trades,
            "volatility": self.volatility(),
            "order_book_imbalance": (
                self.order_book_imbalance(order_book)
            ),
        }


if __name__ == "__main__":
    from order import Order, OrderSide

    book = OrderBook()
    market_data = MarketData()

    book.add_order(
        Order(
            1,
            OrderSide.BUY,
            100.00,
            10,
        )
    )

    book.add_order(
        Order(
            2,
            OrderSide.SELL,
            102.00,
            8,
        )
    )

    trade = Trade(
        trade_id=1,
        price=101.00,
        quantity=5,
        buy_order_id=1,
        sell_order_id=2,
    )

    market_data.update_from_trade(trade)

    print("\n========== MARKET DATA ==========")

    print(
        f"Last Price: "
        f"{market_data.last_price:.2f}"
    )

    print(
        f"Total Volume: "
        f"{market_data.total_volume}"
    )

    print(
        f"Total Trades: "
        f"{market_data.total_trades}"
    )

    print(
        f"Spread: "
        f"{book.spread:.2f}"
    )

    print(
        f"Volatility: "
        f"{market_data.volatility():.6f}"
    )

    print(
        f"Order Book Imbalance: "
        f"{market_data.order_book_imbalance(book):.4f}"
    )

    print("\nMarket Snapshot:")

    for key, value in market_data.snapshot(book).items():
        print(f"{key}: {value}")

    print("=================================")