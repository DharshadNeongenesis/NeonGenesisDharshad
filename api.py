"""
NEON GENESIS - Browser API
Serves the frontend dashboard and exposes the trading simulation API.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from integration import NeonGenesis
from stresstest import StressTestEngine


# ============================================================
# APP CONFIG
# ============================================================

APP_NAME = "NEON GENESIS"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = (
    "Real-time market-making simulation API for the NEON GENESIS trading system."
)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_FILE = FRONTEND_DIR / "index.html"


# ============================================================
# REQUEST MODELS
# ============================================================

class SimulationStartRequest(BaseModel):
    interval: float = Field(default=0.50, ge=0.05, le=60.0)


class StressTestStartRequest(BaseModel):
    steps_per_phase: int = Field(default=10, ge=1, le=10_000)


# ============================================================
# CONTROLLER
# ============================================================

class NeonAPIController:

    def __init__(self) -> None:
        self.engine = NeonGenesis()

        self._state_lock = threading.RLock()

        self._simulation_task: asyncio.Task | None = None
        self._simulation_running = False
        self._simulation_interval = 0.50

        self._stress_engine: StressTestEngine | None = None
        self._stress_thread: threading.Thread | None = None
        self._stress_running = False
        self._stress_error: str | None = None

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    @property
    def simulation_running(self) -> bool:
        return self._simulation_running

    @property
    def simulation_interval(self) -> float:
        return self._simulation_interval

    async def start_simulation(self, interval: float) -> None:
        self._simulation_interval = interval

        if (
            self._simulation_task is not None
            and not self._simulation_task.done()
        ):
            self._simulation_running = True
            return

        self._simulation_running = True

        self._simulation_task = asyncio.create_task(
            self._simulation_loop()
        )

    async def stop_simulation(self) -> None:
        self._simulation_running = False

        task = self._simulation_task
        self._simulation_task = None

        if task is not None and not task.done():
            task.cancel()

            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _simulation_loop(self) -> None:
        try:
            while self._simulation_running:
                self.step()
                await asyncio.sleep(self._simulation_interval)

        except asyncio.CancelledError:
            raise

        finally:
            self._simulation_running = False

    def step(self) -> dict[str, Any]:
        with self._state_lock:
            self.engine.step()
            return self.state()

    def reset(self) -> dict[str, Any]:
        with self._state_lock:
            self.engine = NeonGenesis()
            return self.state()

    # --------------------------------------------------------
    # Stress Test
    # --------------------------------------------------------

    @property
    def stress_running(self) -> bool:
        return self._stress_running

    def start_stress_test(self, steps_per_phase: int) -> None:

        if self._stress_running:
            raise RuntimeError("A stress test is already running.")

        if (
            self._stress_thread is not None
            and self._stress_thread.is_alive()
        ):
            raise RuntimeError("A stress test is already running.")

        self._stress_engine = StressTestEngine(
            steps_per_phase=steps_per_phase
        )

        self._stress_error = None
        self._stress_running = True

        self._stress_thread = threading.Thread(
            target=self._run_stress_test,
            daemon=True,
            name="neon-genesis-stress-test",
        )

        self._stress_thread.start()

    def _run_stress_test(self) -> None:

        engine = self._stress_engine

        if engine is None:
            self._stress_running = False
            return

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                engine.run()

        except Exception as exc:
            self._stress_error = str(exc)

        finally:
            self._stress_running = False

    # --------------------------------------------------------
    # Complete State
    # --------------------------------------------------------

    def state(self) -> dict[str, Any]:

        with self._state_lock:

            engine = self.engine

            price = self._current_market_price()

            features = engine.market_features.snapshot()

            prediction = engine.ai_predictor.predict(
                engine.market_features
            )

            maker_state = engine.market_maker.state(price)

            quote = maker_state.quote

            risk = engine.risk_manager.assess(
                inventory=maker_state.inventory,
                market_price=price,
                pnl=maker_state.pnl,
                features=engine.market_features,
                prediction=prediction,
            )

            analytics = engine.analytics.summary()

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),

                "application": {
                    "name": APP_NAME,
                    "version": APP_VERSION,
                },

                "simulation": {
                    "running": self._simulation_running,
                    "interval": self._simulation_interval,
                    "step": engine.step_number,
                },

                "market": self._serialize_market(
                    price,
                    features,
                ),

                "order_book": self._serialize_order_book(
                    engine.simulator.matching_engine.order_book
                ),

                "trades": self._serialize_recent_trades(),

                "ai": self._serialize_prediction(
                    prediction
                ),

                "market_maker": self._serialize_market_maker(
                    maker_state,
                    quote,
                ),

                "risk": self._serialize_risk(
                    risk
                ),

                "analytics": self._json_safe(
                    analytics
                ),
            }

    # --------------------------------------------------------
    # Individual API State
    # --------------------------------------------------------

    def market_state(self) -> dict[str, Any]:
        state = self.state()

        return {
            "timestamp": state["timestamp"],
            "simulation": state["simulation"],
            "market": state["market"],
        }

    def order_book_state(self) -> dict[str, Any]:
        with self._state_lock:
            return self._serialize_order_book(
                self.engine.simulator.matching_engine.order_book
            )

    def trades_state(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),

                "trades": self._serialize_recent_trades(),

                "total_trades": (
                    self.engine.simulator.market_data.total_trades
                ),

                "total_volume": (
                    self.engine.simulator.market_data.total_volume
                ),
            }

    def analytics_state(self) -> dict[str, Any]:
        with self._state_lock:
            return self._json_safe(
                self.engine.analytics.summary()
            )

    # --------------------------------------------------------
    # Stress State
    # --------------------------------------------------------

    def stress_state(self) -> dict[str, Any]:

        engine = self._stress_engine

        if engine is None:
            return {
                "running": False,
                "available": False,
                "phase": None,
                "step": 0,
                "snapshots": [],
                "phase_results": [],
                "validation": None,
                "error": self._stress_error,
            }

        snapshots = list(engine.snapshots())
        phase_results = list(engine.phase_results())

        latest = snapshots[-1] if snapshots else None

        return {
            "running": self._stress_running,
            "available": True,

            "phase": (
                latest.phase.value
                if latest is not None
                else None
            ),

            "step": (
                latest.step
                if latest is not None
                else 0
            ),

            "latest": (
                self._json_safe(latest)
                if latest is not None
                else None
            ),

            "snapshots": [
                self._json_safe(item)
                for item in snapshots
            ],

            "phase_results": [
                self._json_safe(item)
                for item in phase_results
            ],

            "validation": (
                self._json_safe(
                    engine.validation_report()
                )
                if not self._stress_running
                else None
            ),

            "error": self._stress_error,
        }

    # ========================================================
    # SERIALIZATION
    # ========================================================

    def _current_market_price(self) -> float:

        price = self.engine.simulator.market_data.last_price

        if price is not None and price > 0:
            return float(price)

        reference = self.engine.simulator.reference_price

        if reference <= 0:
            raise RuntimeError(
                "Market price is unavailable."
            )

        return float(reference)

    def _serialize_market(
        self,
        price: float,
        features: dict[str, Any],
    ) -> dict[str, Any]:

        def number(value: Any) -> float | None:
            if value is None:
                return None

            return float(value)

        return {
            "price": float(price),

            "reference_price": float(
                self.engine.simulator.reference_price
            ),

            "spread": number(
                features.get("spread")
            ),

            "volatility": number(
                features.get("volatility")
            ),

            "order_book_imbalance": number(
                features.get("order_book_imbalance")
            ),

            "volume": int(
                features.get("volume", 0)
            ),

            "trade_count": int(
                features.get("trade_count", 0)
            ),

            "price_change": number(
                features.get("price_change")
            ),
        }

    def _serialize_prediction(
        self,
        prediction: Any,
    ) -> dict[str, Any]:

        return {
            "regime": self._enum_value(
                prediction.regime
            ),

            "directional_bias": self._enum_value(
                prediction.directional_bias
            ),

            "confidence": float(
                prediction.confidence
            ),

            "expected_movement": float(
                prediction.expected_movement
            ),

            "suggested_spread_multiplier": float(
                prediction.suggested_spread_multiplier
            ),
        }

    def _serialize_risk(
        self,
        risk: Any,
    ) -> dict[str, Any]:

        return {
            "level": self._enum_value(
                risk.level
            ),

            "inventory": int(
                risk.inventory
            ),

            "inventory_utilization": float(
                risk.inventory_utilization
            ),

            "exposure": float(
                risk.exposure
            ),

            "exposure_utilization": float(
                risk.exposure_utilization
            ),

            "volatility": float(
                risk.volatility
            ),

            "loss": float(
                risk.loss
            ),

            "quote_size_multiplier": float(
                risk.quote_size_multiplier
            ),

            "trading_allowed": bool(
                risk.allow_trading
            ),
        }

    def _serialize_market_maker(
        self,
        state: Any,
        quote: Any,
    ) -> dict[str, Any]:

        serialized_quote = None

        if quote is not None:

            serialized_quote = {
                "fair_price": float(
                    quote.fair_price
                ),

                "bid_price": (
                    float(quote.bid_price)
                    if quote.bid_price is not None
                    else None
                ),

                "ask_price": (
                    float(quote.ask_price)
                    if quote.ask_price is not None
                    else None
                ),

                "bid_quantity": int(
                    quote.bid_quantity
                ),

                "ask_quantity": int(
                    quote.ask_quantity
                ),

                "spread": float(
                    quote.spread
                ),

                "inventory_skew": float(
                    quote.inventory_skew
                ),

                "ai_price_adjustment": float(
                    quote.ai_price_adjustment
                ),

                "risk_size_multiplier": float(
                    quote.risk_size_multiplier
                ),

                "risk_trading_allowed": bool(
                    quote.risk_trading_allowed
                ),

                "prediction": self._serialize_prediction(
                    quote.prediction
                ),
            }

        return {
            "cash": float(state.cash),
            "inventory": int(state.inventory),
            "equity": float(state.equity),
            "pnl": float(state.pnl),

            "active_bid_order_id": (
                state.active_bid_order_id
            ),

            "active_ask_order_id": (
                state.active_ask_order_id
            ),

            "quote": serialized_quote,
        }

    def _serialize_order_book(
        self,
        order_book: Any,
    ) -> dict[str, Any]:

        bids = []
        asks = []

        for price, orders in getattr(
            order_book,
            "_bids",
            {},
        ).items():

            quantity = sum(
                int(order.quantity)
                for order in orders
                if order.quantity > 0
            )

            if quantity > 0:
                bids.append({
                    "price": float(price),
                    "quantity": quantity,
                    "order_count": len(orders),
                })

        for price, orders in getattr(
            order_book,
            "_asks",
            {},
        ).items():

            quantity = sum(
                int(order.quantity)
                for order in orders
                if order.quantity > 0
            )

            if quantity > 0:
                asks.append({
                    "price": float(price),
                    "quantity": quantity,
                    "order_count": len(orders),
                })

        bids.sort(
            key=lambda x: x["price"],
            reverse=True,
        )

        asks.sort(
            key=lambda x: x["price"]
        )

        best_bid = order_book.best_bid
        best_ask = order_book.best_ask

        return {
            "best_bid": (
                float(best_bid.price)
                if best_bid is not None
                else None
            ),

            "best_ask": (
                float(best_ask.price)
                if best_ask is not None
                else None
            ),

            "spread": (
                float(order_book.spread)
                if order_book.spread is not None
                else None
            ),

            "bids": bids,
            "asks": asks,
        }

    def _serialize_recent_trades(
        self,
    ) -> list[dict[str, Any]]:

        trade_history = getattr(
            self.engine.simulator.market_data,
            "trade_history",
            [],
        )

        return [
            {
                "trade_id": int(
                    trade.trade_id
                ),

                "price": float(
                    trade.price
                ),

                "quantity": int(
                    trade.quantity
                ),

                "buy_order_id": int(
                    trade.buy_order_id
                ),

                "sell_order_id": int(
                    trade.sell_order_id
                ),

                "value": float(
                    trade.value
                ),
            }

            for trade in trade_history
        ]

    # ========================================================
    # JSON HELPERS
    # ========================================================

    @staticmethod
    def _enum_value(value: Any) -> Any:
        return getattr(
            value,
            "value",
            value,
        )

    @classmethod
    def _json_safe(
        cls,
        value: Any,
    ) -> Any:

        if value is None:
            return None

        if isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        if hasattr(value, "value") and not isinstance(
            value,
            (dict, list, tuple, set),
        ):
            return cls._json_safe(
                value.value
            )

        if is_dataclass(value):
            return cls._json_safe(
                asdict(value)
            )

        if isinstance(value, dict):
            return {
                str(key): cls._json_safe(item)
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return [
                cls._json_safe(item)
                for item in value
            ]

        return str(value)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

controller = NeonAPIController()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    if FRONTEND_FILE.exists():
        return FileResponse(
            FRONTEND_FILE,
            media_type="text/html",
        )

    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "online",
        "description": APP_DESCRIPTION,
        "docs": "/docs",
        "websocket": "/ws",
        "frontend": "frontend/index.html not found",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
async def health():

    return {
        "status": "healthy",
        "application": APP_NAME,
        "version": APP_VERSION,
        "simulation_running": (
            controller.simulation_running
        ),
        "stress_test_running": (
            controller.stress_running
        ),
    }


# ============================================================
# DASHBOARD API
# ============================================================

@app.get("/api/state")
async def get_state():
    return controller.state()


@app.get("/api/market")
async def get_market():
    return controller.market_state()


@app.get("/api/orderbook")
async def get_orderbook():
    return controller.order_book_state()


@app.get("/api/trades")
async def get_trades():
    return controller.trades_state()


@app.get("/api/analytics")
async def get_analytics():
    return controller.analytics_state()


# ============================================================
# SIMULATION CONTROLS
# ============================================================

@app.post("/api/simulation/start")
async def start_simulation(
    request: SimulationStartRequest,
):

    await controller.start_simulation(
        request.interval
    )

    return {
        "status": "running",
        "interval": controller.simulation_interval,
        "state": controller.state(),
    }


@app.post("/api/simulation/stop")
async def stop_simulation():

    await controller.stop_simulation()

    return {
        "status": "stopped",
        "state": controller.state(),
    }


@app.post("/api/simulation/step")
async def simulation_step():

    if controller.simulation_running:
        raise HTTPException(
            status_code=409,
            detail=(
                "Stop the automatic simulation "
                "before stepping manually."
            ),
        )

    return controller.step()


@app.post("/api/simulation/reset")
async def reset_simulation():

    if controller.simulation_running:
        raise HTTPException(
            status_code=409,
            detail=(
                "Stop the simulation "
                "before resetting it."
            ),
        )

    return controller.reset()


# ============================================================
# STRESS TEST
# ============================================================

@app.get("/api/stress-test")
async def get_stress_test():
    return controller.stress_state()


@app.post("/api/stress-test/start")
async def start_stress_test(
    request: StressTestStartRequest,
):

    if controller.simulation_running:
        raise HTTPException(
            status_code=409,
            detail=(
                "Stop the live simulation "
                "before starting the stress test."
            ),
        )

    try:

        controller.start_stress_test(
            request.steps_per_phase
        )

    except RuntimeError as exc:

        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    return controller.stress_state()


# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
):

    await websocket.accept()

    try:

        while True:

            await websocket.send_json({
                "type": "state",
                "data": controller.state(),
            })

            await websocket.send_json({
                "type": "stress_test",
                "data": controller.stress_state(),
            })

            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        return

    except Exception:

        with contextlib.suppress(Exception):
            await websocket.close()


# ============================================================
# SHUTDOWN
# ============================================================

@app.on_event("shutdown")
async def shutdown_event():

    await controller.stop_simulation()


# ============================================================
# DIRECT RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )