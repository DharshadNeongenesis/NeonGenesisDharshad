from __future__ import annotations

from collections import deque
from typing import Deque

from order import Order, OrderSide
from trade import Trade


class OrderBook:
    """
    Price-time-priority limit order book.

    Bids:
        Highest price has priority.

    Asks:
        Lowest price has priority.

    Orders at the same price:
        First in, first out (FIFO).
    """

    def __init__(self) -> None:
        self._bids: dict[float, Deque[Order]] = {}
        self._asks: dict[float, Deque[Order]] = {}

        self._orders_by_id: dict[int, Order] = {}
        self._next_trade_id = 1

    @property
    def best_bid(self) -> Order | None:
        """Return the highest-priority buy order."""

        if not self._bids:
            return None

        price = max(self._bids)
        return self._bids[price][0]

    @property
    def best_ask(self) -> Order | None:
        """Return the highest-priority sell order."""

        if not self._asks:
            return None

        price = min(self._asks)
        return self._asks[price][0]

    @property
    def spread(self) -> float | None:
        """Return the current bid-ask spread."""

        if self.best_bid is None or self.best_ask is None:
            return None

        return self.best_ask.price - self.best_bid.price

    def add_order(self, order: Order) -> None:
        """Add an order while preserving price-time priority."""

        if order.order_id in self._orders_by_id:
            raise ValueError(
                f"Order ID {order.order_id} already exists."
            )

        book = (
            self._bids
            if order.side is OrderSide.BUY
            else self._asks
        )

        if order.price not in book:
            book[order.price] = deque()

        book[order.price].append(order)
        self._orders_by_id[order.order_id] = order

    def get_remaining_quantity(self, order_id: int) -> int:
        """
        Return the remaining quantity of an active order.

        Returns 0 if the order has been completely filled
        or is no longer active.
        """

        order = self._orders_by_id.get(order_id)

        if order is None:
            return 0

        return order.quantity

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an outstanding order.

        Returns True if cancelled successfully.
        """

        order = self._orders_by_id.get(order_id)

        if order is None:
            return False

        book = (
            self._bids
            if order.side is OrderSide.BUY
            else self._asks
        )

        orders = book[order.price]
        orders.remove(order)

        if not orders:
            del book[order.price]

        del self._orders_by_id[order_id]

        return True

    def match_orders(self) -> list[Trade]:
        """Match all compatible buy and sell orders."""

        trades: list[Trade] = []

        while self._bids and self._asks:
            bid_price = max(self._bids)
            ask_price = min(self._asks)

            if bid_price < ask_price:
                break

            bid = self._bids[bid_price][0]
            ask = self._asks[ask_price][0]

            quantity = min(
                bid.quantity,
                ask.quantity,
            )

            trade = Trade(
                trade_id=self._next_trade_id,
                price=ask.price,
                quantity=quantity,
                buy_order_id=bid.order_id,
                sell_order_id=ask.order_id,
            )

            trades.append(trade)
            self._next_trade_id += 1

            bid.fill(quantity)
            ask.fill(quantity)

            if bid.quantity == 0:
                self._bids[bid_price].popleft()
                del self._orders_by_id[bid.order_id]

                if not self._bids[bid_price]:
                    del self._bids[bid_price]

            if ask.quantity == 0:
                self._asks[ask_price].popleft()
                del self._orders_by_id[ask.order_id]

                if not self._asks[ask_price]:
                    del self._asks[ask_price]

        return trades

    def display(self) -> None:
        """Display the current order book."""

        print("\n========== ORDER BOOK ==========")

        print("\nASKS:")

        for price in sorted(self._asks):
            quantity = sum(
                order.quantity
                for order in self._asks[price]
            )

            print(
                f"  {price:>8.2f} | "
                f"{quantity:>5}"
            )

        print("\n-------------------------------")

        print("BIDS:")

        for price in sorted(
            self._bids,
            reverse=True,
        ):
            quantity = sum(
                order.quantity
                for order in self._bids[price]
            )

            print(
                f"  {price:>8.2f} | "
                f"{quantity:>5}"
            )

        print("================================")


if __name__ == "__main__":
    book = OrderBook()

    book.add_order(
        Order(1, OrderSide.BUY, 100.00, 10)
    )

    book.add_order(
        Order(2, OrderSide.BUY, 101.00, 5)
    )

    book.add_order(
        Order(3, OrderSide.SELL, 103.00, 8)
    )

    book.add_order(
        Order(4, OrderSide.SELL, 102.00, 6)
    )

    book.display()

    print(f"\nBest Bid: {book.best_bid.price}")
    print(f"Best Ask: {book.best_ask.price}")
    print(f"Spread: {book.spread}")

    book.add_order(
        Order(5, OrderSide.BUY, 103.00, 3)
    )

    trades = book.match_orders()

    print("\nEXECUTED TRADES:")

    for trade in trades:
        print(trade)

    book.display()