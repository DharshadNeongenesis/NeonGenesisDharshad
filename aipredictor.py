from __future__ import annotations

from dataclasses import dataclass

from marketfeatures import MarketFeatures


class MarketRegime:
    """Possible market volatility regimes."""

    CALM = "CALM"
    NORMAL = "NORMAL"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"


class DirectionalBias:
    """Possible short-term market directions."""

    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"


@dataclass(frozen=True, slots=True)
class Prediction:
    """
    Represents the AI layer's assessment of the market.

    The prediction contains:
        - market regime
        - directional bias
        - confidence
        - expected movement
        - suggested spread multiplier
    """

    regime: str
    directional_bias: str
    confidence: float
    expected_movement: float
    suggested_spread_multiplier: float

    def __post_init__(self) -> None:
        """Validate prediction values."""

        valid_regimes = {
            MarketRegime.CALM,
            MarketRegime.NORMAL,
            MarketRegime.HIGH_VOLATILITY,
        }

        valid_biases = {
            DirectionalBias.BULLISH,
            DirectionalBias.NEUTRAL,
            DirectionalBias.BEARISH,
        }

        if self.regime not in valid_regimes:
            raise ValueError(
                f"Invalid market regime: {self.regime}"
            )

        if self.directional_bias not in valid_biases:
            raise ValueError(
                f"Invalid directional bias: "
                f"{self.directional_bias}"
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "Confidence must be between 0.0 and 1.0."
            )

        if not -1.0 <= self.expected_movement <= 1.0:
            raise ValueError(
                "Expected movement must be between "
                "-1.0 and 1.0."
            )

        if self.suggested_spread_multiplier <= 0:
            raise ValueError(
                "Spread multiplier must be greater than zero."
            )


class AIPredictor:
    """
    Interpretable AI-style market prediction layer.

    Inputs:
        - price change
        - order-book imbalance
        - volatility
        - spread

    Outputs:
        - market regime
        - directional bias
        - confidence
        - expected movement
        - suggested spread multiplier

    The current implementation is intentionally
    deterministic and interpretable. It provides the
    interface that a trained ML model can later replace.
    """

    def __init__(
        self,
        volatility_normal_threshold: float = 0.005,
        volatility_high_threshold: float = 0.015,
        neutral_threshold: float = 0.25,
    ) -> None:

        if volatility_normal_threshold < 0:
            raise ValueError(
                "Normal volatility threshold "
                "cannot be negative."
            )

        if (
            volatility_high_threshold
            <= volatility_normal_threshold
        ):
            raise ValueError(
                "High volatility threshold must be "
                "greater than normal volatility threshold."
            )

        if not 0.0 <= neutral_threshold <= 1.0:
            raise ValueError(
                "Neutral threshold must be between "
                "0.0 and 1.0."
            )

        self._volatility_normal_threshold = (
            volatility_normal_threshold
        )

        self._volatility_high_threshold = (
            volatility_high_threshold
        )

        self._neutral_threshold = neutral_threshold

    def _classify_regime(
        self,
        volatility: float,
    ) -> str:
        """
        Classify the market according to volatility.
        """

        if volatility >= self._volatility_high_threshold:
            return MarketRegime.HIGH_VOLATILITY

        if volatility >= self._volatility_normal_threshold:
            return MarketRegime.NORMAL

        return MarketRegime.CALM

    def _classify_direction(
        self,
        price_change: float,
        imbalance: float,
    ) -> str:
        """
        Determine directional pressure.

        Price momentum and order-book imbalance are
        combined into bullish and bearish scores.
        """

        bullish_score = 0.0
        bearish_score = 0.0

        # Price momentum signal.
        price_signal = min(
            abs(price_change) * 10.0,
            1.0,
        )

        if price_change > 0:
            bullish_score += price_signal

        elif price_change < 0:
            bearish_score += price_signal

        # Order-book pressure signal.
        imbalance_signal = min(
            abs(imbalance),
            1.0,
        )

        if imbalance > 0:
            bullish_score += imbalance_signal

        elif imbalance < 0:
            bearish_score += imbalance_signal

        # Determine dominant direction.
        if (
            bullish_score > bearish_score
            and bullish_score >= self._neutral_threshold
        ):
            return DirectionalBias.BULLISH

        if (
            bearish_score > bullish_score
            and bearish_score >= self._neutral_threshold
        ):
            return DirectionalBias.BEARISH

        return DirectionalBias.NEUTRAL

    def _calculate_confidence(
        self,
        price_change: float,
        imbalance: float,
        volatility: float,
    ) -> float:
        """
        Calculate confidence from available signals.

        Stronger price movement and stronger order-book
        imbalance increase confidence.

        Volatility contributes a smaller amount because
        high volatility does not automatically imply
        higher directional certainty.
        """

        price_signal = min(
            abs(price_change) * 10.0,
            1.0,
        )

        imbalance_signal = min(
            abs(imbalance),
            1.0,
        )

        volatility_signal = min(
            volatility * 20.0,
            1.0,
        )

        confidence = (
            price_signal * 0.35
            + imbalance_signal * 0.45
            + volatility_signal * 0.20
        )

        return max(
            0.0,
            min(confidence, 1.0),
        )

    def _calculate_expected_movement(
        self,
        price_change: float,
        imbalance: float,
    ) -> float:
        """
        Estimate short-term directional movement.

        Positive values indicate upward pressure.
        Negative values indicate downward pressure.
        """

        expected_movement = (
            price_change * 0.60
            + imbalance * 0.40
        )

        return max(
            -1.0,
            min(expected_movement, 1.0),
        )

    def _calculate_spread_multiplier(
        self,
        regime: str,
        confidence: float,
        spread: float | None,
    ) -> float:
        """
        Determine how much the market maker should
        adjust its quoted spread.

        Higher volatility widens the spread.

        Lower confidence also widens the spread.

        An unusually large existing spread produces
        a small additional risk adjustment.
        """

        regime_multiplier = {
            MarketRegime.CALM: 1.00,
            MarketRegime.NORMAL: 1.25,
            MarketRegime.HIGH_VOLATILITY: 1.75,
        }[regime]

        confidence_adjustment = (
            1.0
            + (1.0 - confidence) * 0.50
        )

        spread_adjustment = 1.0

        if spread is not None and spread > 2.0:
            spread_adjustment = 1.10

        multiplier = (
            regime_multiplier
            * confidence_adjustment
            * spread_adjustment
        )

        return max(
            0.10,
            multiplier,
        )

    def predict(
        self,
        features: MarketFeatures,
    ) -> Prediction:
        """
        Generate a prediction from market features.
        """

        if not isinstance(features, MarketFeatures):
            raise TypeError(
                "features must be a MarketFeatures instance."
            )

        snapshot = features.snapshot()

        price_change = snapshot["price_change"]
        imbalance = snapshot["order_book_imbalance"]
        volatility = snapshot["volatility"]
        spread = snapshot["spread"]

        if price_change is None:
            price_change = 0.0

        if imbalance is None:
            imbalance = 0.0

        if volatility is None:
            volatility = 0.0

        if spread is not None and spread < 0:
            spread = None

        regime = self._classify_regime(
            volatility
        )

        directional_bias = self._classify_direction(
            price_change,
            imbalance,
        )

        confidence = self._calculate_confidence(
            price_change,
            imbalance,
            volatility,
        )

        expected_movement = (
            self._calculate_expected_movement(
                price_change,
                imbalance,
            )
        )

        spread_multiplier = (
            self._calculate_spread_multiplier(
                regime,
                confidence,
                spread,
            )
        )

        return Prediction(
            regime=regime,
            directional_bias=directional_bias,
            confidence=confidence,
            expected_movement=expected_movement,
            suggested_spread_multiplier=spread_multiplier,
        )

    def display(
        self,
        prediction: Prediction,
    ) -> None:
        """Display the AI prediction."""

        print("\n========== AI PREDICTION ==========")

        print(
            f"Market Regime: "
            f"{prediction.regime}"
        )

        print(
            f"Directional Bias: "
            f"{prediction.directional_bias}"
        )

        print(
            f"Confidence: "
            f"{prediction.confidence:.2%}"
        )

        print(
            f"Expected Movement: "
            f"{prediction.expected_movement:.4%}"
        )

        print(
            f"Suggested Spread Multiplier: "
            f"{prediction.suggested_spread_multiplier:.2f}x"
        )

        print("===================================")


if __name__ == "__main__":
    from order import Order, OrderSide
    from orderbook import OrderBook
    from marketdata import MarketData
    from trade import Trade

    # Create an isolated order book.
    book = OrderBook()

    book.add_order(
        Order(
            order_id=1,
            side=OrderSide.BUY,
            price=100.00,
            quantity=10,
        )
    )

    book.add_order(
        Order(
            order_id=2,
            side=OrderSide.SELL,
            price=102.00,
            quantity=8,
        )
    )

    # Create market data.
    market_data = MarketData()

    # Synthetic trade used only for testing
    # the MarketData / AI pipeline in isolation.
    market_data.update_from_trade(
        Trade(
            trade_id=1,
            price=101.00,
            quantity=5,
            buy_order_id=1,
            sell_order_id=2,
        )
    )

    # Convert raw market information into features.
    features = MarketFeatures(
        market_data,
        book,
    )

    # Create AI predictor.
    predictor = AIPredictor()

    # Generate prediction.
    prediction = predictor.predict(
        features
    )

    # Display prediction.
    predictor.display(
        prediction
    )