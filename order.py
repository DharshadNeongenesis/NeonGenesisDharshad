from dataclasses import dataclass
from enum import Enum
import math


class OrderSide(Enum):
    """The two possible sides of an order."""

    BUY = "BUY"
    SELL = "SELL"


@dataclass(slots=True)
class Order:
    """
    Represents a single limit order.

    quantity represents the remaining quantity
    available for execution.
    """

    order_id: int
    side: OrderSide
    price: float
    quantity: int

    def __post_init__(self) -> None:
        """Validate the order after creation."""

        if self.order_id <= 0:
            raise ValueError("Order ID must be positive.")

        if not isinstance(self.side, OrderSide):
            raise TypeError("Side must be an OrderSide.")

        if not math.isfinite(self.price) or self.price <= 0:
            raise ValueError(
                "Order price must be a positive finite number."
            )

        if self.quantity <= 0:
            raise ValueError(
                "Quantity must be greater than zero."
            )

    @property
    def value(self) -> float:
        """Return the current remaining notional value."""

        return self.price * self.quantity

    def fill(self, quantity: int) -> None:
        """
        Reduce the remaining quantity after execution.
        """

        if quantity <= 0:
            raise ValueError(
                "Fill quantity must be greater than zero."
            )

        if quantity > self.quantity:
            raise ValueError(
                "Fill quantity exceeds remaining quantity."
            )

        self.quantity -= quantity

    def __str__(self) -> str:
        """Return a readable representation of the order."""

        return (
            f"Order #{self.order_id} | "
            f"{self.side.value:<4} | "
            f"Price: {self.price:,.2f} | "
            f"Qty: {self.quantity:,} | "
            f"Value: {self.value:,.2f}"
        )


if __name__ == "__main__":
    buy_order = Order(
        order_id=1,
        side=OrderSide.BUY,
        price=100.00,
        quantity=10,
    )

    sell_order = Order(
        order_id=2,
        side=OrderSide.SELL,
        price=102.00,
        quantity=5,
    )

    print(buy_order)
    print(sell_order)