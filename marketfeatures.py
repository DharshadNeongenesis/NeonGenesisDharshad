from __future__ import annotations

from orderbook import OrderBook
from marketdata import MarketData


class MarketFeatures:
    """
    Converts raw market data into structured features
    that can be consumed by the AI prediction layer.
    """

    def __init__(
        self,
        market_data: MarketData,
        order_book: OrderBook,
    ) -> None:

        self._market_data = market_data
        self._order_book = order_book

    @property
    def last_price(self) -> float | None:
        """Return the latest traded price."""

        return self._market_data.last_price

    @property
    def spread(self) -> float | None:
        """Return the current bid-ask spread."""

        return self._order_book.spread

    @property
    def volatility(self) -> float:
        """Return current historical volatility."""

        return self._market_data.volatility()

    @property
    def order_book_imbalance(self) -> float:
        """Return top-of-book order imbalance."""

        return self._market_data.order_book_imbalance(
            self._order_book
        )

    @property
    def volume(self) -> int:
        """Return total traded volume."""

        return self._market_data.total_volume

    @property
    def trade_count(self) -> int:
        """Return total number of trades."""

        return self._market_data.total_trades

    def price_change(self) -> float:
        """
        Calculate the percentage change between
        the first and latest recorded prices.

        Returns 0.0 when insufficient data exists.
        """

        prices = self._market_data.price_history

        if len(prices) < 2:
            return 0.0

        first_price = prices[0]
        last_price = prices[-1]

        if first_price <= 0:
            return 0.0

        return (
            (last_price - first_price)
            / first_price
        )

    def snapshot(self) -> dict[str, float | int | None]:
        """
        Return all features required by the AI layer.
        """

        return {
            "last_price": self.last_price,
            "spread": self.spread,
            "volatility": self.volatility,
            "order_book_imbalance": (
                self.order_book_imbalance
            ),
            "volume": self.volume,
            "trade_count": self.trade_count,
            "price_change": self.price_change(),
        }

    def display(self) -> None:
        """Display the current feature set."""

        print("\n========== MARKET FEATURES ==========")

        features = self.snapshot()

        for name, value in features.items():
            print(f"{name}: {value}")

        print("======================================")


if __name__ == "__main__":
    from order import Order, OrderSide
    from trade import Trade

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

    market_data.update_from_trade(
        Trade(
            trade_id=1,
            price=101.00,
            quantity=5,
            buy_order_id=1,
            sell_order_id=2,
        )
    )

    market_features = MarketFeatures(
        market_data,
        book,
    )

    market_features.display()  