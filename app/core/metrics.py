import logging

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

logger = logging.getLogger(__name__)

_chat_turns: metrics.Counter | None = None
_hitl_interrupts: metrics.Counter | None = None
_mcp_search_duration: metrics.Histogram | None = None
_chat_turn_duration: metrics.Histogram | None = None


def init_metrics(otlp_endpoint: str) -> None:
    global _chat_turns, _hitl_interrupts, _mcp_search_duration, _chat_turn_duration

    try:
        endpoint = otlp_endpoint.rstrip("/") + "/v1/metrics"
        exporter = OTLPMetricExporter(endpoint=endpoint)
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=30_000)
        provider = MeterProvider(metric_readers=[reader])
        metrics.set_meter_provider(provider)

        meter = metrics.get_meter("shelter.chat-api")

        _chat_turns = meter.create_counter(
            "chat.turns.total",
            description="Chat turns by resolved intent",
        )
        _hitl_interrupts = meter.create_counter(
            "chat.hitl_interrupts.total",
            description="HITL interrupt events emitted (intake_request / context_clarify_request)",
        )
        _mcp_search_duration = meter.create_histogram(
            "mcp.search.duration_ms",
            unit="ms",
            description="Duration of search_services MCP tool calls per group",
        )
        _chat_turn_duration = meter.create_histogram(
            "chat.turn.duration_ms",
            unit="ms",
            description="Wall-clock SSE turn duration from start to finish / error / interrupt",
        )
        logger.info("Custom OTel metrics initialised — exporting to %s", endpoint)
    except Exception:
        logger.exception("Failed to initialise custom metrics — continuing without them")


def record_turn(intent: str) -> None:
    if _chat_turns is not None:
        _chat_turns.add(1, {"intent": intent})


def record_hitl_interrupt(interrupt_type: str) -> None:
    if _hitl_interrupts is not None:
        _hitl_interrupts.add(1, {"type": interrupt_type})


def record_mcp_search_duration(duration_ms: float, status: str) -> None:
    if _mcp_search_duration is not None:
        _mcp_search_duration.record(duration_ms, {"status": status})


def record_turn_duration(duration_ms: float, outcome: str) -> None:
    if _chat_turn_duration is not None:
        _chat_turn_duration.record(duration_ms, {"outcome": outcome})
