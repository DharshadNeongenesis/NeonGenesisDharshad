from __future__ import annotations

from dataclasses import dataclass

from order import Order, OrderSide
from orderbook import OrderBook
from trade import Trade


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """Result of processing an order."""

    order_id: int
    trades: tuple[Trade, ...]
    remaining_quantity: int
    fully_filled: bool


class MatchingEngine:
    """
    Coordinates order submission and execution.
    """

    def __init__(self) -> None:
        self._order_book = OrderBook()

    @property
    def order_book(self) -> OrderBook:
        """Return the underlying order book."""

        return self._order_book

    def submit_order(
        self,
        order: Order,
    ) -> ExecutionReport:
        """
        Submit an order and execute compatible orders.
        """

        self._order_book.add_order(order)

        trades = self._order_book.match_orders()

        remaining_quantity = (
            self._order_book.get_remaining_quantity(
                order.order_id
            )
        )

        return ExecutionReport(
            order_id=order.order_id,
            trades=tuple(trades),
            remaining_quantity=remaining_quantity,
            fully_filled=remaining_quantity == 0,
        )

    def cancel_order(self, order_id: int) -> bool:
        """Cancel an outstanding order."""

        return self._order_book.cancel_order(order_id)


if __name__ == "__main__":
    engine = MatchingEngine()

    engine.submit_order(
        Order(
            1,
            OrderSide.BUY,
            100.00,
            10,
        )
    )

    engine.submit_order(
        Order(
            2,
            OrderSide.SELL,
            105.00,
            5,
        )
    )

    print("\nInitial Order Book:")
    engine.order_book.display()

    incoming_order = Order(
        3,
        OrderSide.BUY,
        105.00,
        3,
    )

    report = engine.submit_order(
        incoming_order
    )

    print("\nEXECUTION REPORT")
    print("------------------------------")
    print(f"Order ID: {report.order_id}")
    print(f"Trades: {len(report.trades)}")
    print(
        f"Remaining Quantity: "
        f"{report.remaining_quantity}"
    )
    print(
        f"Fully Filled: "
        f"{report.fully_filled}"
    )

    for trade in report.trades:
        print(f"\n{trade}")

    print("\nFinal Order Book:")
    engine.order_book.display()