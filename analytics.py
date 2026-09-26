from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from collections import Counter
from typing import Optional


@dataclass(frozen=True, slots=True)
class AnalyticsSnapshot:
    step: int
    price: Optional[float]
    price_return: float

    total_trades: int
    total_volume: int
    average_trade_volume: float

    inventory: int
    average_inventory: float
    maximum_inventory: int

    equity: float
    pnl: float
    drawdown: float
    maximum_drawdown: float

    volatility: float
    average_volatility: float
    maximum_volatility: float

    spread: Optional[float]
    average_spread: float
    minimum_spread: float
    maximum_spread: float

    risk_level: str
    warning_events: int
    critical_events: int
    halted_events: int


class Analytics:
    """
    Analytics engine for NeonGenesis.

    Tracks:
        - Market performance
        - Trading activity
        - Inventory
        - Equity and P&L
        - Drawdown
        - Volatility
        - Spread
        - Risk events

    This module is UI-independent and can later feed the
    NeonGenesis browser dashboard.
    """

    def __init__(self) -> None:
        self._step = 0

        # Market history
        self._prices: list[float] = []
        self._volatilities: list[float] = []
        self._spreads: list[float] = []

        # Trading history
        self._trade_count_history: list[int] = []
        self._volume_history: list[int] = []

        # Portfolio history
        self._inventory_history: list[int] = []
        self._equity_history: list[float] = []
        self._pnl_history: list[float] = []

        # Risk history
        self._risk_history: list[str] = []
        self._risk_counts: Counter[str] = Counter()

        # Performance tracking
        self._peak_equity: Optional[float] = None
        self._maximum_drawdown = 0.0

    # ================================================================
    # RECORD
    # ================================================================

    def record(
        self,
        *,
        price: Optional[float],
        total_trades: int,
        total_volume: int,
        inventory: int,
        equity: float,
        pnl: float,
        volatility: float,
        spread: Optional[float],
        risk_level: str,
    ) -> AnalyticsSnapshot:
        """
        Record the state of NeonGenesis after one simulation step.
        """

        self._validate(
            price=price,
            total_trades=total_trades,
            total_volume=total_volume,
            inventory=inventory,
            equity=equity,
            pnl=pnl,
            volatility=volatility,
            spread=spread,
            risk_level=risk_level,
        )

        self._step += 1

        # ------------------------------------------------------------
        # Market
        # ------------------------------------------------------------

        if price is not None:
            self._prices.append(price)

        self._volatilities.append(volatility)

        if spread is not None:
            self._spreads.append(spread)

        # ------------------------------------------------------------
        # Trading
        # ------------------------------------------------------------

        self._trade_count_history.append(total_trades)
        self._volume_history.append(total_volume)

        # ------------------------------------------------------------
        # Portfolio
        # ------------------------------------------------------------

        self._inventory_history.append(inventory)
        self._equity_history.append(equity)
        self._pnl_history.append(pnl)

        # ------------------------------------------------------------
        # Drawdown
        # ------------------------------------------------------------

        if self._peak_equity is None:
            self._peak_equity = equity
        else:
            self._peak_equity = max(
                self._peak_equity,
                equity,
            )

        drawdown = max(
            self._peak_equity - equity,
            0.0,
        )

        self._maximum_drawdown = max(
            self._maximum_drawdown,
            drawdown,
        )

        # ------------------------------------------------------------
        # Risk
        # ------------------------------------------------------------

        self._risk_history.append(risk_level)
        self._risk_counts[risk_level] += 1

        return self.snapshot()

    # ================================================================
    # SNAPSHOT
    # ================================================================

    def snapshot(self) -> AnalyticsSnapshot:
        """
        Return a complete current analytics snapshot.
        """

        price = (
            self._prices[-1]
            if self._prices
            else None
        )

        volatility = (
            self._volatilities[-1]
            if self._volatilities
            else 0.0
        )

        spread = (
            self._spreads[-1]
            if self._spreads
            else None
        )

        total_trades = self.total_trades()
        total_volume = self.total_volume()

        average_trade_volume = (
            total_volume / total_trades
            if total_trades > 0
            else 0.0
        )

        inventory = self.current_inventory()

        average_inventory = (
            mean(self._inventory_history)
            if self._inventory_history
            else 0.0
        )

        maximum_inventory = (
            max(
                (abs(value) for value in self._inventory_history),
                default=0,
            )
        )

        equity = self.current_equity()
        pnl = self.current_pnl()

        drawdown = self.current_drawdown()

        return AnalyticsSnapshot(
            step=self._step,
            price=price,
            price_return=self.price_return(),

            total_trades=total_trades,
            total_volume=total_volume,
            average_trade_volume=average_trade_volume,

            inventory=inventory,
            average_inventory=average_inventory,
            maximum_inventory=maximum_inventory,

            equity=equity,
            pnl=pnl,
            drawdown=drawdown,
            maximum_drawdown=self._maximum_drawdown,

            volatility=volatility,
            average_volatility=self.average_volatility(),
            maximum_volatility=self.maximum_volatility(),

            spread=spread,
            average_spread=self.average_spread(),
            minimum_spread=self.minimum_spread(),
            maximum_spread=self.maximum_spread(),

            risk_level=self.current_risk_level(),
            warning_events=self.warning_events(),
            critical_events=self.critical_events(),
            halted_events=self.halted_events(),
        )

    # ================================================================
    # MARKET METRICS
    # ================================================================

    def price_return(self) -> float:
        """
        Return from initial recorded price to current price.

        Example:
            100 -> 105 = +5%
        """

        if len(self._prices) < 2:
            return 0.0

        initial = self._prices[0]
        current = self._prices[-1]

        if initial <= 0:
            return 0.0

        return (current - initial) / initial

    def average_volatility(self) -> float:
        if not self._volatilities:
            return 0.0

        return mean(self._volatilities)

    def maximum_volatility(self) -> float:
        if not self._volatilities:
            return 0.0

        return max(self._volatilities)

    def average_spread(self) -> float:
        if not self._spreads:
            return 0.0

        return mean(self._spreads)

    def minimum_spread(self) -> float:
        if not self._spreads:
            return 0.0

        return min(self._spreads)

    def maximum_spread(self) -> float:
        if not self._spreads:
            return 0.0

        return max(self._spreads)

    # ================================================================
    # TRADING METRICS
    # ================================================================

    def total_trades(self) -> int:
        if not self._trade_count_history:
            return 0

        return self._trade_count_history[-1]

    def total_volume(self) -> int:
        if not self._volume_history:
            return 0

        return self._volume_history[-1]

    def trade_activity(self) -> float:
        """
        Average cumulative trades per simulation step.
        """

        if self._step == 0:
            return 0.0

        return self.total_trades() / self._step

    # ================================================================
    # PORTFOLIO METRICS
    # ================================================================

    def current_inventory(self) -> int:
        if not self._inventory_history:
            return 0

        return self._inventory_history[-1]

    def maximum_inventory(self) -> int:
        if not self._inventory_history:
            return 0

        return max(
            abs(value)
            for value in self._inventory_history
        )

    def average_inventory(self) -> float:
        if not self._inventory_history:
            return 0.0

        return mean(self._inventory_history)

    def current_equity(self) -> float:
        if not self._equity_history:
            return 0.0

        return self._equity_history[-1]

    def current_pnl(self) -> float:
        if not self._pnl_history:
            return 0.0

        return self._pnl_history[-1]

    # ================================================================
    # DRAWDOWN
    # ================================================================

    def current_drawdown(self) -> float:
        if (
            self._peak_equity is None
            or not self._equity_history
        ):
            return 0.0

        return max(
            self._peak_equity - self._equity_history[-1],
            0.0,
        )

    def maximum_drawdown(self) -> float:
        return self._maximum_drawdown

    # ================================================================
    # RISK METRICS
    # ================================================================

    def current_risk_level(self) -> str:
        if not self._risk_history:
            return "UNKNOWN"

        return self._risk_history[-1]

    def risk_event_count(self, risk_level: str) -> int:
        return self._risk_counts.get(risk_level, 0)

    def warning_events(self) -> int:
        return self._risk_counts.get("WARNING", 0)

    def critical_events(self) -> int:
        return (
            self._risk_counts.get("CRITICAL", 0)
            + self._risk_counts.get("HALTED", 0)
        )

    def halted_events(self) -> int:
        return self._risk_counts.get("HALTED", 0)

    def risk_distribution(self) -> dict[str, int]:
        return dict(self._risk_counts)

    # ================================================================
    # SUMMARY
    # ================================================================

    def summary(self) -> dict[str, object]:
        """
        Dashboard-ready dictionary.

        This will later become the data source for the web UI.
        """

        snapshot = self.snapshot()

        return {
            "step": snapshot.step,

            "market": {
                "price": snapshot.price,
                "price_return": snapshot.price_return,
                "volatility": snapshot.volatility,
                "average_volatility": snapshot.average_volatility,
                "maximum_volatility": snapshot.maximum_volatility,
                "spread": snapshot.spread,
                "average_spread": snapshot.average_spread,
                "minimum_spread": snapshot.minimum_spread,
                "maximum_spread": snapshot.maximum_spread,
            },

            "trading": {
                "total_trades": snapshot.total_trades,
                "total_volume": snapshot.total_volume,
                "average_trade_volume": snapshot.average_trade_volume,
                "trade_activity": self.trade_activity(),
            },

            "portfolio": {
                "inventory": snapshot.inventory,
                "average_inventory": snapshot.average_inventory,
                "maximum_inventory": snapshot.maximum_inventory,
                "equity": snapshot.equity,
                "pnl": snapshot.pnl,
                "drawdown": snapshot.drawdown,
                "maximum_drawdown": snapshot.maximum_drawdown,
            },

            "risk": {
                "level": snapshot.risk_level,
                "warning_events": snapshot.warning_events,
                "critical_events": snapshot.critical_events,
                "halted_events": snapshot.halted_events,
                "distribution": self.risk_distribution(),
            },
        }

    # ================================================================
    # DISPLAY
    # ================================================================

    def display(self) -> None:
        """
        Windows-safe ASCII terminal display.
        """

        snapshot = self.snapshot()

        print()
        print("========== NEON GENESIS ANALYTICS ==========")

        print(f"Simulation Step:       {snapshot.step}")

        if snapshot.price is not None:
            print(f"Market Price:          {snapshot.price:,.2f}")
        else:
            print("Market Price:          N/A")

        print(
            f"Price Return:          "
            f"{snapshot.price_return * 100:+.2f}%"
        )

        print()
        print("--- TRADING ---")

        print(
            f"Total Trades:          "
            f"{snapshot.total_trades:,}"
        )

        print(
            f"Total Volume:          "
            f"{snapshot.total_volume:,}"
        )

        print(
            f"Avg Trade Volume:      "
            f"{snapshot.average_trade_volume:.2f}"
        )

        print(
            f"Trade Activity:        "
            f"{self.trade_activity():.2f}/step"
        )

        print()
        print("--- PORTFOLIO ---")

        print(
            f"Inventory:             "
            f"{snapshot.inventory:,}"
        )

        print(
            f"Average Inventory:     "
            f"{snapshot.average_inventory:.2f}"
        )

        print(
            f"Maximum Inventory:     "
            f"{snapshot.maximum_inventory:,}"
        )

        print(
            f"Equity:                "
            f"{snapshot.equity:,.2f}"
        )

        print(
            f"P&L:                   "
            f"{snapshot.pnl:+,.2f}"
        )

        print(
            f"Current Drawdown:      "
            f"{snapshot.drawdown:,.2f}"
        )

        print(
            f"Maximum Drawdown:      "
            f"{snapshot.maximum_drawdown:,.2f}"
        )

        print()
        print("--- MARKET QUALITY ---")

        print(
            f"Volatility:            "
            f"{snapshot.volatility:.4f}"
        )

        print(
            f"Average Volatility:    "
            f"{snapshot.average_volatility:.4f}"
        )

        print(
            f"Maximum Volatility:    "
            f"{snapshot.maximum_volatility:.4f}"
        )

        if snapshot.spread is not None:
            print(
                f"Spread:                "
                f"{snapshot.spread:,.4f}"
            )
        else:
            print("Spread:                N/A")

        print(
            f"Average Spread:        "
            f"{snapshot.average_spread:,.4f}"
        )

        print()
        print("--- RISK ---")

        print(
            f"Risk Level:            "
            f"{snapshot.risk_level}"
        )

        print(
            f"Warning Events:        "
            f"{snapshot.warning_events}"
        )

        print(
            f"Critical Events:       "
            f"{snapshot.critical_events}"
        )

        print(
            f"Halted Events:        "
            f"{snapshot.halted_events}"
        )

        print("=============================================")

    # ================================================================
    # HISTORY
    # ================================================================

    @property
    def price_history(self) -> tuple[float, ...]:
        return tuple(self._prices)

    @property
    def inventory_history(self) -> tuple[int, ...]:
        return tuple(self._inventory_history)

    @property
    def equity_history(self) -> tuple[float, ...]:
        return tuple(self._equity_history)

    @property
    def pnl_history(self) -> tuple[float, ...]:
        return tuple(self._pnl_history)

    @property
    def volatility_history(self) -> tuple[float, ...]:
        return tuple(self._volatilities)

    @property
    def spread_history(self) -> tuple[float, ...]:
        return tuple(self._spreads)

    @property
    def risk_history(self) -> tuple[str, ...]:
        return tuple(self._risk_history)

    @property
    def steps(self) -> int:
        return self._step

    # ================================================================
    # VALIDATION
    # ================================================================

    @staticmethod
    def _validate(
        *,
        price: Optional[float],
        total_trades: int,
        total_volume: int,
        inventory: int,
        equity: float,
        pnl: float,
        volatility: float,
        spread: Optional[float],
        risk_level: str,
    ) -> None:

        if price is not None and price <= 0:
            raise ValueError("Price must be positive.")

        if total_trades < 0:
            raise ValueError(
                "Total trades cannot be negative."
            )

        if total_volume < 0:
            raise ValueError(
                "Total volume cannot be negative."
            )

        if volatility < 0:
            raise ValueError(
                "Volatility cannot be negative."
            )

        if spread is not None and spread < 0:
            raise ValueError(
                "Spread cannot be negative."
            )

        if not isinstance(risk_level, str):
            raise TypeError(
                "Risk level must be a string."
            )


# ====================================================================
# STANDALONE VALIDATION
# ====================================================================

if __name__ == "__main__":

    analytics = Analytics()

    analytics.record(
        price=100.00,
        total_trades=2,
        total_volume=20,
        inventory=10,
        equity=10_000.00,
        pnl=0.00,
        volatility=0.005,
        spread=0.50,
        risk_level="SAFE",
    )

    analytics.record(
        price=102.00,
        total_trades=5,
        total_volume=50,
        inventory=15,
        equity=10_040.00,
        pnl=40.00,
        volatility=0.009,
        spread=0.55,
        risk_level="SAFE",
    )

    analytics.record(
        price=98.00,
        total_trades=8,
        total_volume=85,
        inventory=-20,
        equity=9_900.00,
        pnl=-100.00,
        volatility=0.026,
        spread=0.90,
        risk_level="WARNING",
    )

    analytics.display()

    print()
    print("Dashboard Summary:")
    print(analytics.summary())