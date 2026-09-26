from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Trade:
    """
    Represents a completed transaction between
    a buyer and a seller.
    """

    trade_id: int
    price: float
    quantity: int
    buy_order_id: int
    sell_order_id: int

    @property
    def value(self) -> float:
        """Return the total notional value of the trade."""

        return self.price * self.quantity

    def __str__(self) -> str:
        """Return a readable trade representation."""

        return (
            f"Trade #{self.trade_id} | "
            f"Price: {self.price:,.2f} | "
            f"Qty: {self.quantity:,} | "
            f"Value: {self.value:,.2f} | "
            f"BUY #{self.buy_order_id} | "
            f"SELL #{self.sell_order_id}"
        )