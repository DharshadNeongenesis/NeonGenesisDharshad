from __future__ import annotations

import random

from order import Order, OrderSide
from matchingengine import MatchingEngine

try:
    from marketdata import MarketData
except ImportError:  # pragma: no cover - fallback for naming variations
    from marketdata import MarketData


class MarketSimulator:
    """
    Simulates a continuously changing market by generating
    buy and sell orders around a moving reference price.

    Architecture:

        MarketSimulator
              ↓
        MatchingEngine
              ↓
          OrderBook
              ↓
            Trades
              ↓
         MarketData
    """

    def __init__(
        self,
        initial_price: float = 100.00,
        price_volatility: float = 0.02,
        order_size_range: tuple[int, int] = (1, 10),
        price_levels: int = 3,
        price_step: float = 0.50,
        seed: int | None = None,
    ) -> None:

        if initial_price <= 0:
            raise ValueError(
                "Initial price must be greater than zero."
            )

        if price_volatility < 0:
            raise ValueError(
                "Price volatility cannot be negative."
            )

        if (
            len(order_size_range) != 2
            or order_size_range[0] <= 0
            or order_size_range[1] < order_size_range[0]
        ):
            raise ValueError(
                "Invalid order size range."
            )

        if price_levels <= 0:
            raise ValueError(
                "Price levels must be greater than zero."
            )

        if price_step <= 0:
            raise ValueError(
                "Price step must be greater than zero."
            )

        self._reference_price = initial_price
        self._price_volatility = price_volatility
        self._order_size_range = order_size_range
        self._price_levels = price_levels
        self._price_step = price_step

        self._engine = MatchingEngine()
        self._market_data = MarketData()

        self._next_order_id = 1

        self._rng = random.Random(seed)

    @property
    def reference_price(self) -> float:
        """Return the simulator's current reference price."""

        return self._reference_price

    def set_reference_price(self, price: float) -> None:
        """
        Set the simulator's reference price.

        Intended for controlled simulations, scenario testing,
        and stress testing.
        """

        if price <= 0:
            raise ValueError(
                "Reference price must be greater than zero."
            )

        self._reference_price = float(price)

    @property
    def matching_engine(self) -> MatchingEngine:
        """Return the matching engine."""

        return self._engine

    @property
    def market_data(self) -> MarketData:
        """Return current market data."""

        return self._market_data

    def _generate_price(self) -> float:
        """
        Generate a random price around the current
        reference price.
        """

        movement = self._rng.gauss(
            0,
            self._price_volatility,
        )

        self._reference_price *= 1 + movement

        return max(
            0.01,
            round(self._reference_price, 2),
        )

    def _generate_order(
        self,
        side: OrderSide,
    ) -> Order:
        """
        Generate a limit order around the current
        reference price.
        """

        level = self._rng.randint(
            1,
            self._price_levels,
        )

        if side is OrderSide.BUY:
            price = (
                self._reference_price
                - level * self._price_step
            )
        else:
            price = (
                self._reference_price
                + level * self._price_step
            )

        price = max(
            0.01,
            round(price, 2),
        )

        quantity = self._rng.randint(
            self._order_size_range[0],
            self._order_size_range[1],
        )

        order = Order(
            order_id=self._next_order_id,
            side=side,
            price=price,
            quantity=quantity,
        )

        self._next_order_id += 1

        return order

    def step(self) -> list:
        """
        Advance the simulation by one market step.

        A step:
            1. Moves the reference price.
            2. Generates a buy order.
            3. Generates a sell order.
            4. Submits both to the matching engine.
            5. Records any resulting trades.
        """

        self._generate_price()

        buy_order = self._generate_order(
            OrderSide.BUY
        )

        sell_order = self._generate_order(
            OrderSide.SELL
        )

        buy_report = self._engine.submit_order(
            buy_order
        )

        sell_report = self._engine.submit_order(
            sell_order
        )

        trades = [
            *buy_report.trades,
            *sell_report.trades,
        ]

        for trade in trades:
            self._market_data.update_from_trade(
                trade
            )

        return trades

    def run(
        self,
        steps: int = 100,
    ) -> None:
        """
        Run the market simulation for a fixed number
        of steps.
        """

        if steps <= 0:
            raise ValueError(
                "Steps must be greater than zero."
            )

        for step_number in range(1, steps + 1):

            trades = self.step()

            print(
                f"\n========== STEP {step_number} =========="
            )

            print(
                f"Reference Price: "
                f"{self._reference_price:.2f}"
            )

            print(
                f"Trades Executed: "
                f"{len(trades)}"
            )

            if trades:
                for trade in trades:
                    print(trade)

            snapshot = self._market_data.snapshot(
                self._engine.order_book
            )

            print("\nMARKET SNAPSHOT")

            for key, value in snapshot.items():
                print(
                    f"{key}: {value}"
                )


if __name__ == "__main__":
    simulator = MarketSimulator(
        initial_price=100.00,
        price_volatility=0.01,
        order_size_range=(1, 10),
        price_levels=3,
        price_step=0.50,
        seed=42,
    )

    simulator.run(steps=10)