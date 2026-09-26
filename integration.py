from __future__ import annotations

import time

from aipredictor import AIPredictor
from analytics import Analytics
from marketfeatures import MarketFeatures
from marketmaker import MarketMaker
from marketsimulator import MarketSimulator
from riskmanager import RiskManager


class NeonGenesis:
    """
    Main backend controller for NeonGenesis.

    System pipeline:

        Market Simulator
              |
              v
        Market Features
              |
              v
         AI Predictor
              |
              v
         Risk Manager
              |
              v
         Market Maker
              |
              v
          Analytics
    """

    def __init__(self) -> None:

        self.simulator = MarketSimulator(
            initial_price=100.0
        )

        self.ai_predictor = AIPredictor()

        self.market_features = MarketFeatures(
            self.simulator.market_data,
            self.simulator.matching_engine.order_book,
        )

        self.market_maker = MarketMaker(
            self.simulator.matching_engine,
            self.ai_predictor,
        )

        self.risk_manager = RiskManager(
            max_inventory=100,
            max_exposure=10_000.0,
            warning_inventory_ratio=0.70,
            critical_inventory_ratio=0.90,
            warning_exposure_ratio=0.70,
            critical_exposure_ratio=0.90,
            max_loss=1_000.0,
            volatility_warning_threshold=0.015,
            volatility_critical_threshold=0.030,
            minimum_quote_size_multiplier=0.10,
        )

        self.analytics = Analytics()

        self.step_number = 0
        self._risk_level_counts: dict[str, int] = {}

    def step(self) -> None:

        self.step_number += 1

        # --------------------------------------------------
        # 1. Advance market simulation
        # --------------------------------------------------

        self.simulator.step()

        # --------------------------------------------------
        # 2. Rebuild market features
        # --------------------------------------------------

        self.market_features = MarketFeatures(
            self.simulator.market_data,
            self.simulator.matching_engine.order_book,
        )

        # --------------------------------------------------
        # 3. AI prediction
        # --------------------------------------------------

        prediction = self.ai_predictor.predict(
            self.market_features
        )

        # --------------------------------------------------
        # 4. Current market price
        # --------------------------------------------------

        current_price = self._current_market_price()

        # --------------------------------------------------
        # 5. Current market-maker state
        # --------------------------------------------------

        current_state = self.market_maker.state(
            current_price
        )

        # --------------------------------------------------
        # 6. Risk assessment
        # --------------------------------------------------

        risk_assessment = self.risk_manager.assess(
            inventory=current_state.inventory,
            market_price=current_price,
            pnl=current_state.pnl,
            features=self.market_features,
            prediction=prediction,
        )

        # RiskAssessment.level is already a string.
        risk_level = str(risk_assessment.level)

        # --------------------------------------------------
        # 7. Market maker reacts to risk
        # --------------------------------------------------

        trades = self.market_maker.refresh_quotes(
            self.market_features,
            risk_size_multiplier=(
                risk_assessment.quote_size_multiplier
            ),
            risk_trading_allowed=(
                risk_assessment.allow_trading
            ),
        )

        # --------------------------------------------------
        # 8. Refresh features after market-maker activity
        # --------------------------------------------------

        self.market_features = MarketFeatures(
            self.simulator.market_data,
            self.simulator.matching_engine.order_book,
        )

        # --------------------------------------------------
        # 9. Final market-maker state
        # --------------------------------------------------

        final_state = self.market_maker.state(
            current_price
        )

        # --------------------------------------------------
        # 10. Analytics
        # --------------------------------------------------

        self.analytics.record(
            price=current_price,
            total_trades=self._total_market_trades(),
            total_volume=self._total_market_volume(),
            inventory=final_state.inventory,
            equity=final_state.equity,
            pnl=final_state.pnl,
            volatility=self.market_features.volatility,
            spread=self.market_features.spread,
            risk_level=risk_level,
        )

        self._risk_level_counts[risk_level] = (
            self._risk_level_counts.get(risk_level, 0) + 1
        )

        # --------------------------------------------------
        # 11. Console output
        # --------------------------------------------------

        print()
        print(
            f"---------------- STEP {self.step_number} ----------------"
        )

        print(
            f"Price:       {current_price:,.2f}"
        )

        print(
            f"AI Regime:   {str(prediction.regime)}"
        )

        print(
            f"AI Bias:     {str(prediction.directional_bias)}"
        )

        print(
            f"Confidence:  "
            f"{prediction.confidence * 100:.2f}%"
        )

        print(
            f"Risk:        {risk_level}"
        )

        print(
            f"Risk Size:   "
            f"{risk_assessment.quote_size_multiplier:.2f}x"
        )

        print(
            f"Inventory:   {final_state.inventory}"
        )

        print(
            f"Equity:      {final_state.equity:,.2f}"
        )

        print(
            f"P&L:         {final_state.pnl:+,.2f}"
        )

        print(
            f"Trades:      {len(trades)}"
        )

    def run(
        self,
        steps: int = 25,
        delay: float = 0.05,
    ) -> None:

        if steps <= 0:
            raise ValueError(
                "Steps must be greater than zero."
            )

        if delay < 0:
            raise ValueError(
                "Delay cannot be negative."
            )

        print()
        print("==============================================")
        print("          NEON GENESIS MARKET ENGINE")
        print("==============================================")
        print()
        print(f"Simulation Steps: {steps}")
        print()

        for _ in range(steps):

            self.step()

            if delay > 0:
                time.sleep(delay)

        self.display_summary()

    def display_summary(self) -> None:

        analytics = self.analytics.summary()

        market = analytics["market"]
        trading = analytics["trading"]
        portfolio = analytics["portfolio"]
        risk = analytics["risk"]

        print()
        print()
        print("==============================================")
        print("          NEON GENESIS FINAL REPORT")
        print("==============================================")

        print()
        print("--- MARKET ---")

        price = market["price"]

        if price is not None:
            print(
                f"Final Price:           {price:,.2f}"
            )
        else:
            print(
                "Final Price:           N/A"
            )

        print(
            f"Price Return:          "
            f"{market['price_return'] * 100:+.2f}%"
        )

        print(
            f"Average Volatility:    "
            f"{market['average_volatility']:.4f}"
        )

        print(
            f"Maximum Volatility:    "
            f"{market['maximum_volatility']:.4f}"
        )

        print(
            f"Average Spread:        "
            f"{market['average_spread']:.4f}"
        )

        print()
        print("--- TRADING ---")

        print(
            f"Total Trades:          "
            f"{trading['total_trades']:,}"
        )

        print(
            f"Total Volume:          "
            f"{trading['total_volume']:,}"
        )

        print(
            f"Trade Activity:        "
            f"{trading['trade_activity']:.2f}/step"
        )

        print()
        print("--- MARKET MAKER ---")

        print(
            f"Final Inventory:       "
            f"{portfolio['inventory']:,}"
        )

        print(
            f"Maximum Inventory:     "
            f"{portfolio['maximum_inventory']:,}"
        )

        print(
            f"Average Inventory:     "
            f"{portfolio['average_inventory']:.2f}"
        )

        print(
            f"Final Equity:          "
            f"{portfolio['equity']:,.2f}"
        )

        print(
            f"Final P&L:             "
            f"{portfolio['pnl']:+,.2f}"
        )

        print(
            f"Maximum Drawdown:      "
            f"{portfolio['maximum_drawdown']:,.2f}"
        )

        print()
        print("--- RISK ---")

        print(
            f"Final Risk Level:      "
            f"{risk['level']}"
        )

        print(
            f"Warning Events:        "
            f"{risk['warning_events']}"
        )

        print(
            f"Critical Events:       "
            f"{risk['critical_events']}"
        )

        print(
            f"Halted Events:        "
            f"{risk['halted_events']}"
        )

        print()
        print("Risk Distribution:")

        for level, count in risk["distribution"].items():
            print(
                f"  {level:<10} {count}"
            )

        print()
        print("==============================================")
        print("             SIMULATION COMPLETE")
        print("==============================================")

    def _current_market_price(self) -> float:

        last_price = self.simulator.market_data.last_price

        if last_price is not None and last_price > 0:
            return last_price

        reference_price = self.simulator.reference_price

        if reference_price <= 0:
            raise RuntimeError(
                "Market simulator produced an invalid price."
            )

        return reference_price

    def _total_market_trades(self) -> int:
        return self.simulator.market_data.total_trades

    def _total_market_volume(self) -> int:
        return self.simulator.market_data.total_volume


if __name__ == "__main__":

    engine = NeonGenesis()

    engine.run(
        steps=25,
        delay=0.05,
    )