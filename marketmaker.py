from __future__ import annotations

from dataclasses import dataclass

from aipredictor import (
    AIPredictor,
    DirectionalBias,
    Prediction,
)
from marketfeatures import MarketFeatures
from matchingengine import MatchingEngine
from order import Order, OrderSide
from trade import Trade


@dataclass(frozen=True, slots=True)
class Quote:
    """
    Represents a complete market-maker quote.

    The quote combines:
        - market reference price
        - AI valuation
        - inventory skew
        - dynamic spread
        - risk-controlled sizing
    """

    fair_price: float
    bid_price: float | None
    ask_price: float | None

    bid_quantity: int
    ask_quantity: int

    spread: float

    inventory_skew: float
    ai_price_adjustment: float

    risk_size_multiplier: float
    risk_trading_allowed: bool

    prediction: Prediction


@dataclass(frozen=True, slots=True)
class MarketMakerState:
    """
    Immutable snapshot of market-maker state.
    """

    cash: float
    inventory: int
    equity: float
    pnl: float

    active_bid_order_id: int | None
    active_ask_order_id: int | None

    quote: Quote | None


class MarketMaker:
    """
    AI-driven market-making engine.

    Strategy pipeline:

        MarketFeatures
             ↓
        AIPredictor
             ↓
        Fair Value
             ↓
        AI Adjustment
             ↓
        Inventory Skew
             ↓
        Dynamic Spread
             ↓
        Risk-Controlled Sizing
             ↓
        Bid / Ask Quotes
             ↓
        MatchingEngine
    """

    def __init__(
        self,
        matching_engine: MatchingEngine,
        predictor: AIPredictor,
        initial_fair_price: float = 100.00,
        base_spread: float = 0.0025,
        quote_size: int = 5,
        max_inventory: int = 100,
        inventory_skew_factor: float = 0.010,
        max_ai_price_adjustment: float = 0.020,
        starting_cash: float = 100_000.00,
        initial_inventory: int = 0,
        order_id_start: int = 1_000_000,
    ) -> None:

        if not isinstance(matching_engine, MatchingEngine):
            raise TypeError(
                "matching_engine must be a MatchingEngine."
            )

        if not isinstance(predictor, AIPredictor):
            raise TypeError(
                "predictor must be an AIPredictor."
            )

        if initial_fair_price <= 0:
            raise ValueError(
                "Initial fair price must be greater than zero."
            )

        if base_spread <= 0:
            raise ValueError(
                "Base spread must be greater than zero."
            )

        if quote_size <= 0:
            raise ValueError(
                "Quote size must be greater than zero."
            )

        if max_inventory <= 0:
            raise ValueError(
                "Maximum inventory must be greater than zero."
            )

        if inventory_skew_factor < 0:
            raise ValueError(
                "Inventory skew factor cannot be negative."
            )

        if max_ai_price_adjustment < 0:
            raise ValueError(
                "AI price adjustment cannot be negative."
            )

        if starting_cash < 0:
            raise ValueError(
                "Starting cash cannot be negative."
            )

        if not (
            -max_inventory
            <= initial_inventory
            <= max_inventory
        ):
            raise ValueError(
                "Initial inventory exceeds the inventory limit."
            )

        if order_id_start <= 0:
            raise ValueError(
                "Order ID start must be positive."
            )

        self._matching_engine = matching_engine
        self._predictor = predictor

        self._initial_fair_price = initial_fair_price

        self._base_spread = base_spread
        self._quote_size = quote_size
        self._max_inventory = max_inventory

        self._inventory_skew_factor = (
            inventory_skew_factor
        )

        self._max_ai_price_adjustment = (
            max_ai_price_adjustment
        )

        self._cash = starting_cash
        self._inventory = initial_inventory

        self._initial_equity = (
            starting_cash
            + initial_inventory * initial_fair_price
        )

        self._next_order_id = order_id_start

        self._active_bid_order_id: int | None = None
        self._active_ask_order_id: int | None = None

        self._last_quote: Quote | None = None
        self._last_prediction: Prediction | None = None

    # =========================================================
    # PROPERTIES
    # =========================================================

    @property
    def matching_engine(self) -> MatchingEngine:
        return self._matching_engine

    @property
    def inventory(self) -> int:
        return self._inventory

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def last_quote(self) -> Quote | None:
        return self._last_quote

    @property
    def last_prediction(self) -> Prediction | None:
        return self._last_prediction

    @property
    def active_bid_order_id(self) -> int | None:
        return self._active_bid_order_id

    @property
    def active_ask_order_id(self) -> int | None:
        return self._active_ask_order_id

    # =========================================================
    # MARKET REFERENCE
    # =========================================================

    def _reference_price(
        self,
        features: MarketFeatures,
    ) -> float:

        if (
            features.last_price is not None
            and features.last_price > 0
        ):
            return features.last_price

        order_book = self._matching_engine.order_book

        best_bid = order_book.best_bid
        best_ask = order_book.best_ask

        if (
            best_bid is not None
            and best_ask is not None
        ):
            return (
                best_bid.price
                + best_ask.price
            ) / 2.0

        if best_bid is not None:
            return best_bid.price

        if best_ask is not None:
            return best_ask.price

        return self._initial_fair_price

    # =========================================================
    # AI VALUATION
    # =========================================================

    def _ai_adjustment(
        self,
        prediction: Prediction,
    ) -> float:

        adjustment = (
            prediction.expected_movement
            * self._max_ai_price_adjustment
        )

        return max(
            -self._max_ai_price_adjustment,
            min(
                adjustment,
                self._max_ai_price_adjustment,
            ),
        )

    # =========================================================
    # INVENTORY CONTROL
    # =========================================================

    def _inventory_skew(self) -> float:

        inventory_ratio = (
            self._inventory
            / self._max_inventory
        )

        return -(
            inventory_ratio
            * self._inventory_skew_factor
        )

    def _inventory_ratios(
        self,
    ) -> tuple[float, float]:

        buy_capacity = (
            self._max_inventory
            - self._inventory
        )

        sell_capacity = (
            self._max_inventory
            + self._inventory
        )

        buy_ratio = (
            buy_capacity
            / (2 * self._max_inventory)
        )

        sell_ratio = (
            sell_capacity
            / (2 * self._max_inventory)
        )

        return (
            max(0.0, min(buy_ratio, 1.0)),
            max(0.0, min(sell_ratio, 1.0)),
        )

    # =========================================================
    # FAIR VALUE
    # =========================================================

    def _fair_value(
        self,
        reference_price: float,
        prediction: Prediction,
    ) -> tuple[float, float, float]:

        ai_adjustment = self._ai_adjustment(
            prediction
        )

        inventory_skew = self._inventory_skew()

        total_adjustment = (
            ai_adjustment
            + inventory_skew
        )

        fair_price = (
            reference_price
            * (1.0 + total_adjustment)
        )

        return (
            max(fair_price, 0.01),
            ai_adjustment,
            inventory_skew,
        )

    # =========================================================
    # SPREAD ENGINE
    # =========================================================

    def _dynamic_spread(
        self,
        fair_price: float,
        prediction: Prediction,
    ) -> float:

        spread = (
            fair_price
            * self._base_spread
            * prediction.suggested_spread_multiplier
        )

        return max(
            spread,
            0.01,
        )

    # =========================================================
    # QUOTE SIZING
    # =========================================================

    def _quote_quantities(
        self,
        risk_size_multiplier: float = 1.0,
    ) -> tuple[int, int]:

        if not 0.0 <= risk_size_multiplier <= 1.0:
            raise ValueError(
                "Risk size multiplier must be between 0 and 1."
            )

        buy_ratio, sell_ratio = (
            self._inventory_ratios()
        )

        base_bid_quantity = max(
            0,
            int(
                round(
                    self._quote_size
                    * buy_ratio
                    * 2.0
                )
            ),
        )

        base_ask_quantity = max(
            0,
            int(
                round(
                    self._quote_size
                    * sell_ratio
                    * 2.0
                )
            ),
        )

        bid_quantity = int(
            round(
                base_bid_quantity
                * risk_size_multiplier
            )
        )

        ask_quantity = int(
            round(
                base_ask_quantity
                * risk_size_multiplier
            )
        )

        maximum_buy = (
            self._max_inventory
            - self._inventory
        )

        maximum_sell = (
            self._max_inventory
            + self._inventory
        )

        bid_quantity = min(
            bid_quantity,
            maximum_buy,
        )

        ask_quantity = min(
            ask_quantity,
            maximum_sell,
        )

        return (
            max(0, bid_quantity),
            max(0, ask_quantity),
        )

    # =========================================================
    # QUOTE GENERATION
    # =========================================================

    def generate_quote(
        self,
        features: MarketFeatures,
        risk_size_multiplier: float = 1.0,
        risk_trading_allowed: bool = True,
    ) -> Quote:

        if not isinstance(features, MarketFeatures):
            raise TypeError(
                "features must be a MarketFeatures instance."
            )

        if not 0.0 <= risk_size_multiplier <= 1.0:
            raise ValueError(
                "Risk size multiplier must be between 0 and 1."
            )

        prediction = self._predictor.predict(
            features
        )

        reference_price = self._reference_price(
            features
        )

        (
            fair_price,
            ai_adjustment,
            inventory_skew,
        ) = self._fair_value(
            reference_price,
            prediction,
        )

        spread = self._dynamic_spread(
            fair_price,
            prediction,
        )

        half_spread = spread / 2.0

        raw_bid = (
            fair_price
            - half_spread
        )

        raw_ask = (
            fair_price
            + half_spread
        )

        bid_price = round(
            max(raw_bid, 0.01),
            2,
        )

        ask_price = round(
            max(raw_ask, 0.01),
            2,
        )

        # -----------------------------------------------------
        # Directional intelligence
        # -----------------------------------------------------

        if (
            prediction.directional_bias
            == DirectionalBias.BULLISH
        ):
            bid_price = round(
                bid_price
                + fair_price * 0.0005,
                2,
            )

        elif (
            prediction.directional_bias
            == DirectionalBias.BEARISH
        ):
            ask_price = round(
                ask_price
                - fair_price * 0.0005,
                2,
            )

        # -----------------------------------------------------
        # Prevent accidental self-crossing.
        # -----------------------------------------------------

        if bid_price >= ask_price:
            ask_price = round(
                bid_price + 0.01,
                2,
            )

        if risk_trading_allowed:
            bid_quantity, ask_quantity = (
                self._quote_quantities(
                    risk_size_multiplier
                )
            )
        else:
            bid_quantity = 0
            ask_quantity = 0

        # -----------------------------------------------------
        # Hard inventory protection.
        # -----------------------------------------------------

        if self._inventory >= self._max_inventory:
            bid_price = None
            bid_quantity = 0

        if self._inventory <= -self._max_inventory:
            ask_price = None
            ask_quantity = 0

        # -----------------------------------------------------
        # Risk halt protection.
        # -----------------------------------------------------

        if not risk_trading_allowed:
            bid_price = None
            ask_price = None
            bid_quantity = 0
            ask_quantity = 0

        return Quote(
            fair_price=round(
                fair_price,
                2,
            ),
            bid_price=bid_price,
            ask_price=ask_price,
            bid_quantity=bid_quantity,
            ask_quantity=ask_quantity,
            spread=round(
                spread,
                2,
            ),
            inventory_skew=inventory_skew,
            ai_price_adjustment=ai_adjustment,
            risk_size_multiplier=(
                risk_size_multiplier
            ),
            risk_trading_allowed=(
                risk_trading_allowed
            ),
            prediction=prediction,
        )

    # =========================================================
    # ORDER ID
    # =========================================================

    def _next_order_id_value(self) -> int:

        order_id = self._next_order_id

        self._next_order_id += 1

        return order_id

    # =========================================================
    # QUOTE LIFECYCLE
    # =========================================================

    def cancel_quotes(self) -> None:

        if self._active_bid_order_id is not None:
            self._matching_engine.cancel_order(
                self._active_bid_order_id
            )

        if self._active_ask_order_id is not None:
            self._matching_engine.cancel_order(
                self._active_ask_order_id
            )

        self._active_bid_order_id = None
        self._active_ask_order_id = None

    # =========================================================
    # EXECUTION ACCOUNTING
    # =========================================================

    def _process_trade(
        self,
        trade: Trade,
    ) -> None:

        if (
            trade.buy_order_id
            == self._active_bid_order_id
        ):
            self._inventory += trade.quantity
            self._cash -= trade.value

        if (
            trade.sell_order_id
            == self._active_ask_order_id
        ):
            self._inventory -= trade.quantity
            self._cash += trade.value

    def _process_trades(
        self,
        trades: list[Trade],
    ) -> None:

        for trade in trades:
            self._process_trade(trade)

    # =========================================================
    # QUOTE REFRESH
    # =========================================================

    def refresh_quotes(
        self,
        features: MarketFeatures,
        risk_size_multiplier: float = 1.0,
        risk_trading_allowed: bool = True,
    ) -> list[Trade]:
        """
        Cancel old quotes and publish new quotes.

        Risk controls are applied before new orders
        are submitted.

        Returns:
            Trades generated while publishing the quote.
        """

        self.cancel_quotes()

        quote = self.generate_quote(
            features,
            risk_size_multiplier=(
                risk_size_multiplier
            ),
            risk_trading_allowed=(
                risk_trading_allowed
            ),
        )

        self._last_quote = quote
        self._last_prediction = quote.prediction

        trades: list[Trade] = []

        # -----------------------------------------------------
        # BID
        # -----------------------------------------------------

        if (
            quote.bid_price is not None
            and quote.bid_quantity > 0
        ):
            bid_order = Order(
                order_id=(
                    self._next_order_id_value()
                ),
                side=OrderSide.BUY,
                price=quote.bid_price,
                quantity=quote.bid_quantity,
            )

            self._active_bid_order_id = (
                bid_order.order_id
            )

            report = (
                self._matching_engine.submit_order(
                    bid_order
                )
            )

            trades.extend(
                report.trades
            )

        # -----------------------------------------------------
        # ASK
        # -----------------------------------------------------

        if (
            quote.ask_price is not None
            and quote.ask_quantity > 0
        ):
            ask_order = Order(
                order_id=(
                    self._next_order_id_value()
                ),
                side=OrderSide.SELL,
                price=quote.ask_price,
                quantity=quote.ask_quantity,
            )

            self._active_ask_order_id = (
                ask_order.order_id
            )

            report = (
                self._matching_engine.submit_order(
                    ask_order
                )
            )

            trades.extend(
                report.trades
            )

        # -----------------------------------------------------
        # Financial accounting.
        # -----------------------------------------------------

        self._process_trades(trades)

        # -----------------------------------------------------
        # Clear completely filled orders.
        # -----------------------------------------------------

        order_book = (
            self._matching_engine.order_book
        )

        if (
            self._active_bid_order_id is not None
            and order_book.get_remaining_quantity(
                self._active_bid_order_id
            ) == 0
        ):
            self._active_bid_order_id = None

        if (
            self._active_ask_order_id is not None
            and order_book.get_remaining_quantity(
                self._active_ask_order_id
            ) == 0
        ):
            self._active_ask_order_id = None

        return trades

    # =========================================================
    # FINANCIAL METRICS
    # =========================================================

    def equity(
        self,
        market_price: float | None = None,
    ) -> float:

        if market_price is None:

            if self._last_quote is not None:
                market_price = (
                    self._last_quote.fair_price
                )
            else:
                market_price = (
                    self._initial_fair_price
                )

        if market_price <= 0:
            raise ValueError(
                "Market price must be greater than zero."
            )

        return (
            self._cash
            + self._inventory * market_price
        )

    def pnl(
        self,
        market_price: float | None = None,
    ) -> float:

        return (
            self.equity(market_price)
            - self._initial_equity
        )

    # =========================================================
    # STATE
    # =========================================================

    def state(
        self,
        market_price: float | None = None,
    ) -> MarketMakerState:

        return MarketMakerState(
            cash=self._cash,
            inventory=self._inventory,
            equity=self.equity(
                market_price
            ),
            pnl=self.pnl(
                market_price
            ),
            active_bid_order_id=(
                self._active_bid_order_id
            ),
            active_ask_order_id=(
                self._active_ask_order_id
            ),
            quote=self._last_quote,
        )

    # =========================================================
    # DISPLAY
    # =========================================================

    def display(self) -> None:

        print("\n========== MARKET MAKER ==========")

        if self._last_quote is None:
            print("No quote generated yet.")
            print("===================================")
            return

        quote = self._last_quote

        print(
            f"Fair Price: "
            f"{quote.fair_price:.2f}"
        )

        print(
            f"Bid: "
            f"{quote.bid_price} "
            f"x {quote.bid_quantity}"
        )

        print(
            f"Ask: "
            f"{quote.ask_price} "
            f"x {quote.ask_quantity}"
        )

        print(
            f"Spread: "
            f"{quote.spread:.2f}"
        )

        print(
            f"Risk Size: "
            f"{quote.risk_size_multiplier:.2f}x"
        )

        print(
            f"Risk Trading: "
            f"{quote.risk_trading_allowed}"
        )

        print(
            f"Inventory: "
            f"{self._inventory}"
        )

        print(
            f"Cash: "
            f"{self._cash:,.2f}"
        )

        print(
            f"Equity: "
            f"{self.equity():,.2f}"
        )

        print(
            f"P&L: "
            f"{self.pnl():,.2f}"
        )

        print(
            f"AI Regime: "
            f"{quote.prediction.regime}"
        )

        print(
            f"AI Bias: "
            f"{quote.prediction.directional_bias}"
        )

        print(
            f"AI Confidence: "
            f"{quote.prediction.confidence:.2%}"
        )

        print(
            f"AI Expected Movement: "
            f"{quote.prediction.expected_movement:.4%}"
        )

        print(
            f"AI Spread Multiplier: "
            f"{quote.prediction.suggested_spread_multiplier:.2f}x"
        )

        print("===================================")


# =============================================================
# STANDALONE TEST
# =============================================================

if __name__ == "__main__":
    from marketdata import MarketData

    print("\n==========================================")
    print("       NEON GENESIS MARKET MAKER")
    print("==========================================")

    engine = MatchingEngine()

    market_data = MarketData()

    engine.submit_order(
        Order(
            order_id=1,
            side=OrderSide.BUY,
            price=99.50,
            quantity=10,
        )
    )

    engine.submit_order(
        Order(
            order_id=2,
            side=OrderSide.SELL,
            price=100.50,
            quantity=10,
        )
    )

    features = MarketFeatures(
        market_data,
        engine.order_book,
    )

    predictor = AIPredictor()

    market_maker = MarketMaker(
        matching_engine=engine,
        predictor=predictor,
        initial_fair_price=100.00,
        base_spread=0.0025,
        quote_size=5,
        max_inventory=100,
        inventory_skew_factor=0.010,
        max_ai_price_adjustment=0.020,
        starting_cash=100_000.00,
        initial_inventory=0,
    )

    quote = market_maker.generate_quote(
        features
    )

    print("\n---------- AI-DRIVEN QUOTE ----------")

    print(
        f"Fair Price: "
        f"{quote.fair_price:.2f}"
    )

    print(
        f"Bid: "
        f"{quote.bid_price} "
        f"x {quote.bid_quantity}"
    )

    print(
        f"Ask: "
        f"{quote.ask_price} "
        f"x {quote.ask_quantity}"
    )

    print(
        f"Spread: "
        f"{quote.spread:.2f}"
    )

    print(
        f"Risk Size: "
        f"{quote.risk_size_multiplier:.2f}x"
    )

    print(
        f"Risk Trading: "
        f"{quote.risk_trading_allowed}"
    )

    print(
        f"AI Regime: "
        f"{quote.prediction.regime}"
    )

    print(
        f"AI Bias: "
        f"{quote.prediction.directional_bias}"
    )

    print(
        f"AI Confidence: "
        f"{quote.prediction.confidence:.2%}"
    )

    print("-------------------------------------")

    trades = market_maker.refresh_quotes(
        features
    )

    print(
        f"\nTrades Generated: "
        f"{len(trades)}"
    )

    for trade in trades:
        print(trade)

    market_maker.display()