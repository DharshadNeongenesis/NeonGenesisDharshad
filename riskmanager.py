from __future__ import annotations

from dataclasses import dataclass

from aipredictor import MarketRegime, Prediction
from marketfeatures import MarketFeatures


class RiskLevel:
    """Risk states used by the risk manager."""

    SAFE = "SAFE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    HALTED = "HALTED"


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """
    Complete assessment of the market maker's current risk.

    inventory:
        Current net position.

    inventory_utilization:
        Absolute inventory as a fraction of the
        permitted maximum.

    exposure:
        Absolute mark-to-market position value.

    exposure_utilization:
        Exposure as a fraction of the permitted maximum.

    volatility:
        Current market volatility.

    loss:
        Current P&L.

    quote_size_multiplier:
        Recommended multiplier for market-maker
        quote quantities.

    allow_trading:
        Whether new quotes should be allowed.

    level:
        Overall risk classification.
    """

    inventory: int
    inventory_utilization: float
    exposure: float
    exposure_utilization: float
    volatility: float
    loss: float
    quote_size_multiplier: float
    allow_trading: bool
    level: str


class RiskManager:
    """
    Risk-control layer for the NeonGenesis market maker.

    The Risk Manager does not execute orders itself.

    It evaluates:
        - inventory
        - exposure
        - volatility
        - P&L
        - AI market regime

    and produces controls that the Market Maker can use.
    """

    def __init__(
        self,
        max_inventory: int = 100,
        max_exposure: float = 10_000.0,
        warning_inventory_ratio: float = 0.70,
        critical_inventory_ratio: float = 0.90,
        warning_exposure_ratio: float = 0.70,
        critical_exposure_ratio: float = 0.90,
        max_loss: float = 1_000.0,
        volatility_warning_threshold: float = 0.015,
        volatility_critical_threshold: float = 0.030,
        minimum_quote_size_multiplier: float = 0.10,
    ) -> None:
        """
        Configure risk limits.

        Ratios are expressed between 0.0 and 1.0.
        """

        if max_inventory <= 0:
            raise ValueError(
                "Max inventory must be greater than zero."
            )

        if max_exposure <= 0:
            raise ValueError(
                "Max exposure must be greater than zero."
            )

        if not 0.0 <= warning_inventory_ratio <= 1.0:
            raise ValueError(
                "Warning inventory ratio must be between 0 and 1."
            )

        if not (
            warning_inventory_ratio
            <= critical_inventory_ratio
            <= 1.0
        ):
            raise ValueError(
                "Inventory risk ratios are invalid."
            )

        if not 0.0 <= warning_exposure_ratio <= 1.0:
            raise ValueError(
                "Warning exposure ratio must be between 0 and 1."
            )

        if not (
            warning_exposure_ratio
            <= critical_exposure_ratio
            <= 1.0
        ):
            raise ValueError(
                "Exposure risk ratios are invalid."
            )

        if max_loss <= 0:
            raise ValueError(
                "Max loss must be greater than zero."
            )

        if volatility_warning_threshold < 0:
            raise ValueError(
                "Volatility warning threshold cannot be negative."
            )

        if (
            volatility_critical_threshold
            <= volatility_warning_threshold
        ):
            raise ValueError(
                "Critical volatility threshold must be greater "
                "than the warning threshold."
            )

        if not (
            0.0
            < minimum_quote_size_multiplier
            <= 1.0
        ):
            raise ValueError(
                "Minimum quote size multiplier must be "
                "greater than 0 and at most 1."
            )

        self._max_inventory = max_inventory
        self._max_exposure = max_exposure

        self._warning_inventory_ratio = (
            warning_inventory_ratio
        )

        self._critical_inventory_ratio = (
            critical_inventory_ratio
        )

        self._warning_exposure_ratio = (
            warning_exposure_ratio
        )

        self._critical_exposure_ratio = (
            critical_exposure_ratio
        )

        self._max_loss = max_loss

        self._volatility_warning_threshold = (
            volatility_warning_threshold
        )

        self._volatility_critical_threshold = (
            volatility_critical_threshold
        )

        self._minimum_quote_size_multiplier = (
            minimum_quote_size_multiplier
        )

    # =====================================================
    # BASIC RISK UTILITIES
    # =====================================================

    def inventory_utilization(
        self,
        inventory: int,
    ) -> float:
        """
        Return absolute inventory utilization.

        Example:
            inventory = -50
            max_inventory = 100

            utilization = 0.50
        """

        return min(
            abs(inventory)
            / self._max_inventory,
            1.0,
        )

    def exposure(
        self,
        inventory: int,
        market_price: float,
    ) -> float:
        """
        Return absolute mark-to-market inventory exposure.
        """

        if market_price <= 0:
            raise ValueError(
                "Market price must be greater than zero."
            )

        return abs(inventory * market_price)

    def exposure_utilization(
        self,
        inventory: int,
        market_price: float,
    ) -> float:
        """
        Return exposure utilization as a fraction
        of the configured maximum exposure.
        """

        return min(
            self.exposure(
                inventory,
                market_price,
            )
            / self._max_exposure,
            1.0,
        )

    # =====================================================
    # QUOTE SIZE CONTROL
    # =====================================================

    def quote_size_multiplier(
        self,
        inventory: int,
        market_price: float,
        volatility: float,
    ) -> float:
        """
        Calculate how aggressively the market maker
        should quote.

        Higher risk reduces quote size.

        The multiplier is always between the configured
        minimum and 1.0.
        """

        if volatility < 0:
            raise ValueError(
                "Volatility cannot be negative."
            )

        inventory_ratio = self.inventory_utilization(
            inventory
        )

        exposure_ratio = self.exposure_utilization(
            inventory,
            market_price,
        )

        multiplier = 1.0

        # -------------------------------------------------
        # Inventory pressure
        # -------------------------------------------------

        if inventory_ratio >= self._critical_inventory_ratio:
            multiplier *= 0.20

        elif inventory_ratio >= self._warning_inventory_ratio:
            multiplier *= 0.60

        # -------------------------------------------------
        # Exposure pressure
        # -------------------------------------------------

        if exposure_ratio >= self._critical_exposure_ratio:
            multiplier *= 0.25

        elif exposure_ratio >= self._warning_exposure_ratio:
            multiplier *= 0.70

        # -------------------------------------------------
        # Volatility pressure
        # -------------------------------------------------

        if volatility >= self._volatility_critical_threshold:
            multiplier *= 0.25

        elif volatility >= self._volatility_warning_threshold:
            multiplier *= 0.65

        return max(
            self._minimum_quote_size_multiplier,
            min(multiplier, 1.0),
        )

    # =====================================================
    # RISK LEVEL
    # =====================================================

    def assess(
        self,
        inventory: int,
        market_price: float,
        pnl: float,
        features: MarketFeatures,
        prediction: Prediction,
    ) -> RiskAssessment:
        """
        Evaluate the complete current risk state.
        """

        if not isinstance(features, MarketFeatures):
            raise TypeError(
                "features must be a MarketFeatures instance."
            )

        if not isinstance(prediction, Prediction):
            raise TypeError(
                "prediction must be a Prediction instance."
            )

        if market_price <= 0:
            raise ValueError(
                "Market price must be greater than zero."
            )

        volatility = features.volatility

        inventory_ratio = self.inventory_utilization(
            inventory
        )

        exposure_value = self.exposure(
            inventory,
            market_price,
        )

        exposure_ratio = min(
            exposure_value
            / self._max_exposure,
            1.0,
        )

        loss = max(
            -pnl,
            0.0,
        )

        # -------------------------------------------------
        # Determine risk level
        # -------------------------------------------------

        level = RiskLevel.SAFE

        if loss >= self._max_loss:
            level = RiskLevel.HALTED

        elif (
            inventory_ratio >= self._critical_inventory_ratio
            or exposure_ratio >= self._critical_exposure_ratio
            or volatility >= self._volatility_critical_threshold
        ):
            level = RiskLevel.CRITICAL

        elif (
            inventory_ratio >= self._warning_inventory_ratio
            or exposure_ratio >= self._warning_exposure_ratio
            or volatility >= self._volatility_warning_threshold
        ):
            level = RiskLevel.WARNING

        # -------------------------------------------------
        # AI high-volatility regime adds risk awareness.
        #
        # It does not independently halt trading.
        # The actual volatility measurement and hard
        # limits remain the authoritative controls.
        # -------------------------------------------------

        if (
            prediction.regime == MarketRegime.HIGH_VOLATILITY
            and level == RiskLevel.SAFE
        ):
            level = RiskLevel.WARNING

        # -------------------------------------------------
        # Trading permission
        # -------------------------------------------------

        allow_trading = level not in {
            RiskLevel.CRITICAL,
            RiskLevel.HALTED,
        }

        # -------------------------------------------------
        # Quote-size recommendation
        # -------------------------------------------------

        size_multiplier = self.quote_size_multiplier(
            inventory=inventory,
            market_price=market_price,
            volatility=volatility,
        )

        if level == RiskLevel.CRITICAL:
            size_multiplier = max(
                self._minimum_quote_size_multiplier,
                min(size_multiplier, 0.10),
            )

        elif level == RiskLevel.HALTED:
            size_multiplier = 0.0

        return RiskAssessment(
            inventory=inventory,
            inventory_utilization=inventory_ratio,
            exposure=exposure_value,
            exposure_utilization=exposure_ratio,
            volatility=volatility,
            loss=loss,
            quote_size_multiplier=size_multiplier,
            allow_trading=allow_trading,
            level=level,
        )

    # =====================================================
    # DISPLAY
    # =====================================================

    def display(
        self,
        assessment: RiskAssessment,
    ) -> None:
        """
        Display the current risk assessment.
        """

        print()
        print(
            "---------------------- RISK MANAGER ----------------------"
        )

        print(
            f" Risk Level:         "
            f"{assessment.level}"
        )

        print(
            f" Inventory:          "
            f"{assessment.inventory}"
        )

        print(
            f" Inventory Usage:    "
            f"{assessment.inventory_utilization:.2%}"
        )

        print(
            f" Exposure:           "
            f"{assessment.exposure:,.2f}"
        )

        print(
            f" Exposure Usage:     "
            f"{assessment.exposure_utilization:.2%}"
        )

        print(
            f" Volatility:         "
            f"{assessment.volatility:.4%}"
        )

        print(
            f" Loss:               "
            f"{assessment.loss:,.2f}"
        )

        print(
            f" Quote Size:         "
            f"{assessment.quote_size_multiplier:.2f}x"
        )

        print(
            f" Trading Allowed:    "
            f"{assessment.allow_trading}"
        )

        print(
            "------------------------------------------------------------"
        )


# =========================================================
# STANDALONE TEST
# =========================================================

if __name__ == "__main__":

    from aipredictor import AIPredictor
    from marketdata import MarketData
    from order import Order, OrderSide
    from orderbook import OrderBook

    # -----------------------------------------------------
    # Build a small valid market state
    # -----------------------------------------------------

    order_book = OrderBook()

    order_book.add_order(
        Order(
            order_id=1,
            side=OrderSide.BUY,
            price=99.50,
            quantity=10,
        )
    )

    order_book.add_order(
        Order(
            order_id=2,
            side=OrderSide.SELL,
            price=100.50,
            quantity=10,
        )
    )

    market_data = MarketData()

    features = MarketFeatures(
        market_data,
        order_book,
    )

    predictor = AIPredictor()

    prediction = predictor.predict(
        features
    )

    # -----------------------------------------------------
    # Create Risk Manager
    # -----------------------------------------------------

    risk_manager = RiskManager(
        max_inventory=100,
        max_exposure=10_000.0,
        max_loss=1_000.0,
    )

    # -----------------------------------------------------
    # Evaluate a normal portfolio
    # -----------------------------------------------------

    assessment = risk_manager.assess(
        inventory=20,
        market_price=100.0,
        pnl=50.0,
        features=features,
        prediction=prediction,
    )

    risk_manager.display(
        assessment
    )