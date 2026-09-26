from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from integration import NeonGenesis


class StressPhase(Enum):
    NORMAL = "NORMAL"
    HIGH_VOLATILITY = "HIGH VOLATILITY"
    CRASH = "CRASH"
    RECOVERY = "RECOVERY"


@dataclass(frozen=True, slots=True)
class StressSnapshot:
    phase: StressPhase
    step: int
    price: float
    volatility: float
    risk_level: str
    inventory: int
    equity: float
    pnl: float
    trades: int
    quote_size_multiplier: float
    trading_allowed: bool


@dataclass(frozen=True, slots=True)
class PhaseResult:
    phase: StressPhase
    steps: int
    starting_price: float
    ending_price: float
    price_return: float
    maximum_volatility: float
    final_inventory: int
    final_equity: float
    final_pnl: float
    total_trades: int
    minimum_quote_size_multiplier: float
    trading_halted: bool
    highest_risk_level: str


class StressTestEngine:
    """
    Controlled market stress-testing engine for NeonGenesis.

    Scenarios:

        NORMAL
            ->
        HIGH VOLATILITY
            ->
        CRASH
            ->
        RECOVERY
    """

    DEFAULT_PHASE_STEPS = 10

    def __init__(
        self,
        *,
        steps_per_phase: int = DEFAULT_PHASE_STEPS,
    ) -> None:

        if steps_per_phase <= 0:
            raise ValueError(
                "steps_per_phase must be greater than zero."
            )

        self.steps_per_phase = steps_per_phase

        self.engine = NeonGenesis()

        self._snapshots: list[StressSnapshot] = []
        self._phase_results: list[PhaseResult] = []

    # =========================================================
    # PUBLIC API
    # =========================================================

    def run(self) -> None:
        """
        Execute the complete four-phase stress test.
        """

        self._print_header()

        self._run_phase(
            StressPhase.NORMAL,
            self._normal_shock,
        )

        self._run_phase(
            StressPhase.HIGH_VOLATILITY,
            self._high_volatility_shock,
        )

        self._run_phase(
            StressPhase.CRASH,
            self._crash_shock,
        )

        self._run_phase(
            StressPhase.RECOVERY,
            self._recovery_shock,
        )

        self._print_final_report()

    def snapshots(self) -> tuple[StressSnapshot, ...]:
        return tuple(self._snapshots)

    def phase_results(self) -> tuple[PhaseResult, ...]:
        return tuple(self._phase_results)

    def validation_report(self) -> dict[str, object]:
        """
        Return a machine-readable validation report.
        """

        return {
            "passed": self._overall_passed(),
            "phases": [
                {
                    "phase": result.phase.value,
                    "steps": result.steps,
                    "starting_price": result.starting_price,
                    "ending_price": result.ending_price,
                    "price_return": result.price_return,
                    "maximum_volatility": (
                        result.maximum_volatility
                    ),
                    "final_inventory": result.final_inventory,
                    "final_equity": result.final_equity,
                    "final_pnl": result.final_pnl,
                    "total_trades": result.total_trades,
                    "minimum_quote_size_multiplier": (
                        result.minimum_quote_size_multiplier
                    ),
                    "trading_halted": result.trading_halted,
                    "highest_risk_level": (
                        result.highest_risk_level
                    ),
                }
                for result in self._phase_results
            ],
        }

    # =========================================================
    # PHASE EXECUTION
    # =========================================================

    def _run_phase(
        self,
        phase: StressPhase,
        shock_function,
    ) -> None:

        print()
        print("==================================================")
        print(f"                 {phase.value}")
        print("==================================================")

        phase_snapshots: list[StressSnapshot] = []

        starting_price = self._current_price()

        for _ in range(self.steps_per_phase):

            shock_function()

            self.engine.step()

            snapshot = self._capture_snapshot(
                phase
            )

            self._snapshots.append(snapshot)
            phase_snapshots.append(snapshot)

        result = self._build_phase_result(
            phase,
            phase_snapshots,
            starting_price,
        )

        self._phase_results.append(result)

        self._print_phase_result(result)

    # =========================================================
    # MARKET SHOCKS
    # =========================================================

    def _normal_shock(self) -> None:
        """
        Normal market conditions.

        No artificial directional shock is applied.
        The underlying simulator controls normal movement.
        """

        return

    def _high_volatility_shock(self) -> None:
        """
        High-volatility regime.

        Apply alternating price shocks before the
        integrated engine processes the step.
        """

        simulator = self.engine.simulator

        current_price = self._current_price()

        movement = self._alternating_shock(
            current_price,
            magnitude=0.012,
        )

        simulator.set_reference_price(
            max(
                0.01,
                current_price + movement,
            )
        )

    def _crash_shock(self) -> None:
        """
        Crash regime.

        Apply a controlled downward shock.
        """

        simulator = self.engine.simulator

        current_price = self._current_price()

        crash_move = current_price * 0.035

        simulator.set_reference_price(
            max(
                0.01,
                current_price - crash_move,
            )
        )

    def _recovery_shock(self) -> None:
        """
        Recovery regime.

        Gradually move the reference price upward.
        """

        simulator = self.engine.simulator

        current_price = self._current_price()

        recovery_move = current_price * 0.012

        simulator.set_reference_price(
            current_price + recovery_move
        )

    # =========================================================
    # SNAPSHOT COLLECTION
    # =========================================================

    def _capture_snapshot(
        self,
        phase: StressPhase,
    ) -> StressSnapshot:

        current_price = self._current_price()

        features = self.engine.market_features

        state = self.engine.market_maker.state(
            current_price
        )

        risk_level = self._current_risk_level()

        quote_size_multiplier = (
            self._latest_quote_size_multiplier()
        )

        trading_allowed = (
            self._latest_trading_allowed()
        )

        snapshot = StressSnapshot(
            phase=phase,
            step=self.engine.step_number,
            price=current_price,
            volatility=features.volatility,
            risk_level=risk_level,
            inventory=state.inventory,
            equity=state.equity,
            pnl=state.pnl,
            trades=self._total_trades(),
            quote_size_multiplier=quote_size_multiplier,
            trading_allowed=trading_allowed,
        )

        self._print_snapshot(snapshot)

        return snapshot

    # =========================================================
    # PHASE ANALYSIS
    # =========================================================

    def _build_phase_result(
        self,
        phase: StressPhase,
        snapshots: list[StressSnapshot],
        starting_price: float,
    ) -> PhaseResult:

        if not snapshots:
            raise RuntimeError(
                "Cannot build a phase result without snapshots."
            )

        ending_price = snapshots[-1].price

        price_return = 0.0

        if starting_price > 0:
            price_return = (
                ending_price - starting_price
            ) / starting_price

        maximum_volatility = max(
            snapshot.volatility
            for snapshot in snapshots
        )

        minimum_quote_size_multiplier = min(
            snapshot.quote_size_multiplier
            for snapshot in snapshots
        )

        highest_risk_level = (
            self._highest_risk_level(
                snapshots
            )
        )

        trading_halted = any(
            not snapshot.trading_allowed
            for snapshot in snapshots
        )

        final_snapshot = snapshots[-1]

        return PhaseResult(
            phase=phase,
            steps=len(snapshots),
            starting_price=starting_price,
            ending_price=ending_price,
            price_return=price_return,
            maximum_volatility=maximum_volatility,
            final_inventory=final_snapshot.inventory,
            final_equity=final_snapshot.equity,
            final_pnl=final_snapshot.pnl,
            total_trades=final_snapshot.trades,
            minimum_quote_size_multiplier=(
                minimum_quote_size_multiplier
            ),
            trading_halted=trading_halted,
            highest_risk_level=highest_risk_level,
        )

    # =========================================================
    # VALIDATION
    # =========================================================

    def _overall_passed(self) -> bool:

        if len(self._phase_results) != 4:
            return False

        phases = {
            result.phase: result
            for result in self._phase_results
        }

        required_phases = (
            StressPhase.NORMAL,
            StressPhase.HIGH_VOLATILITY,
            StressPhase.CRASH,
            StressPhase.RECOVERY,
        )

        if any(
            phase not in phases
            for phase in required_phases
        ):
            return False

        normal = phases[StressPhase.NORMAL]
        high_vol = phases[StressPhase.HIGH_VOLATILITY]
        crash = phases[StressPhase.CRASH]
        recovery = phases[StressPhase.RECOVERY]

        checks = [
            normal.steps > 0,
            high_vol.steps > 0,
            crash.steps > 0,
            recovery.steps > 0,

            high_vol.maximum_volatility
            >= normal.maximum_volatility,

            crash.price_return < 0,

            all(
                result.starting_price > 0
                and result.ending_price > 0
                for result in self._phase_results
            ),
        ]

        return all(checks)

    # =========================================================
    # STATE HELPERS
    # =========================================================

    def _current_price(self) -> float:

        price = (
            self.engine
            .simulator
            .market_data
            .last_price
        )

        if price is not None and price > 0:
            return price

        reference_price = (
            self.engine.simulator.reference_price
        )

        if reference_price <= 0:
            raise RuntimeError(
                "Stress test encountered an invalid market price."
            )

        return reference_price

    def _current_risk_level(self) -> str:

        if not self.engine.analytics.risk_history:
            return "UNKNOWN"

        return self.engine.analytics.risk_history[-1]

    def _latest_quote_size_multiplier(self) -> float:

        if not self._snapshots:
            return 1.0

        return self._snapshots[-1].quote_size_multiplier

    def _latest_trading_allowed(self) -> bool:

        risk_level = self._current_risk_level()

        return risk_level not in {
            "CRITICAL",
            "HALTED",
        }

    def _total_trades(self) -> int:

        return (
            self.engine
            .simulator
            .market_data
            .total_trades
        )

    # =========================================================
    # RISK / SHOCK HELPERS
    # =========================================================

    @staticmethod
    def _alternating_shock(
        current_price: float,
        *,
        magnitude: float,
    ) -> float:

        if current_price <= 0:
            raise ValueError(
                "Current price must be positive."
            )

        if magnitude < 0:
            raise ValueError(
                "Shock magnitude cannot be negative."
            )

        step_index = (
            int(current_price * 1000)
            % 2
        )

        direction = (
            1.0
            if step_index == 0
            else -1.0
        )

        return (
            current_price
            * magnitude
            * direction
        )

    @staticmethod
    def _highest_risk_level(
        snapshots: list[StressSnapshot],
    ) -> str:

        priority = {
            "SAFE": 0,
            "WARNING": 1,
            "CRITICAL": 2,
            "HALTED": 3,
            "UNKNOWN": -1,
        }

        highest = "UNKNOWN"

        for snapshot in snapshots:

            if priority.get(
                snapshot.risk_level,
                -1,
            ) > priority.get(
                highest,
                -1,
            ):
                highest = snapshot.risk_level

        return highest

    # =========================================================
    # OUTPUT
    # =========================================================

    @staticmethod
    def _print_header() -> None:

        print()
        print("==================================================")
        print("          NEON GENESIS STRESS TEST")
        print("==================================================")
        print()

        print(
            "Scenario: NORMAL -> HIGH VOLATILITY"
        )

        print(
            "          -> CRASH -> RECOVERY"
        )

        print()

        print("Testing:")

        print("  AI regime detection")
        print("  Risk escalation")
        print("  Dynamic quote sizing")
        print("  Trading controls")
        print("  Inventory response")
        print("  P&L behaviour")
        print("  Market recovery")

    @staticmethod
    def _print_snapshot(
        snapshot: StressSnapshot,
    ) -> None:

        print(
            f"[{snapshot.phase.value:<16}] "
            f"Step {snapshot.step:>3} | "
            f"Price {snapshot.price:>8.2f} | "
            f"Vol {snapshot.volatility:.4f} | "
            f"Risk {snapshot.risk_level:<8} | "
            f"Inv {snapshot.inventory:>5} | "
            f"P&L {snapshot.pnl:>9.2f} | "
            f"Quote {snapshot.quote_size_multiplier:.2f}x | "
            f"Trading "
            f"{'ON' if snapshot.trading_allowed else 'HALTED'}"
        )

    @staticmethod
    def _print_phase_result(
        result: PhaseResult,
    ) -> None:

        print()
        print(
            f"--- {result.phase.value} RESULT ---"
        )

        print(
            f"Steps:                    "
            f"{result.steps}"
        )

        print(
            f"Starting Price:           "
            f"{result.starting_price:,.2f}"
        )

        print(
            f"Ending Price:             "
            f"{result.ending_price:,.2f}"
        )

        print(
            f"Price Return:             "
            f"{result.price_return * 100:+.2f}%"
        )

        print(
            f"Maximum Volatility:       "
            f"{result.maximum_volatility:.4f}"
        )

        print(
            f"Final Inventory:          "
            f"{result.final_inventory}"
        )

        print(
            f"Final Equity:             "
            f"{result.final_equity:,.2f}"
        )

        print(
            f"Final P&L:                "
            f"{result.final_pnl:+,.2f}"
        )

        print(
            f"Minimum Quote Size:       "
            f"{result.minimum_quote_size_multiplier:.2f}x"
        )

        print(
            f"Highest Risk Level:       "
            f"{result.highest_risk_level}"
        )

        print(
            f"Trading Halted:           "
            f"{'YES' if result.trading_halted else 'NO'}"
        )

    def _print_final_report(self) -> None:

        passed = self._overall_passed()

        print()
        print()
        print("==================================================")
        print("          NEON GENESIS STRESS TEST REPORT")
        print("==================================================")

        print()

        for result in self._phase_results:

            print(
                f"{result.phase.value:<18} "
                f"| Return "
                f"{result.price_return * 100:+7.2f}% "
                f"| Max Vol "
                f"{result.maximum_volatility:.4f} "
                f"| Risk "
                f"{result.highest_risk_level:<8} "
                f"| Quote "
                f"{result.minimum_quote_size_multiplier:.2f}x"
            )

        print()
        print("-----------------------------------------------")

        print(
            f"Validation Status: "
            f"{'PASSED' if passed else 'FAILED'}"
        )

        print(
            f"Phases Completed:  "
            f"{len(self._phase_results)}/4"
        )

        print()

        if passed:
            print(
                "NeonGenesis successfully completed "
                "the four-phase stress scenario."
            )
        else:
            print(
                "Stress validation detected a condition "
                "that requires investigation."
            )

        print()
        print("==================================================")
        print("             STRESS TEST COMPLETE")
        print("==================================================")


if __name__ == "__main__":

    stress_test = StressTestEngine(
        steps_per_phase=10,
    )

    stress_test.run()