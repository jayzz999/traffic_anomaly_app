"""
Network Monitor Module
Monitors network connectivity and bandwidth for adaptive transmission.
Designed for Raspberry Pi edge deployment with RTA Dubai integration.
"""

import socket
import time
import threading
import logging
from typing import Optional, Callable, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import urllib.request
import json

logger = logging.getLogger(__name__)


class NetworkStatus(Enum):
    """Network connectivity status levels"""
    EXCELLENT = "excellent"     # >10 Mbps, <50ms latency
    GOOD = "good"               # >5 Mbps, <100ms latency
    FAIR = "fair"               # >1 Mbps, <200ms latency
    POOR = "poor"               # >256 Kbps, <500ms latency
    CRITICAL = "critical"       # <256 Kbps or >500ms latency
    OFFLINE = "offline"         # No connectivity


@dataclass
class NetworkMetrics:
    """Container for network metrics"""
    timestamp: datetime
    status: NetworkStatus
    latency_ms: float
    packet_loss: float
    estimated_bandwidth_kbps: float
    is_connected: bool
    server_reachable: bool
    consecutive_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'latency_ms': self.latency_ms,
            'packet_loss': self.packet_loss,
            'estimated_bandwidth_kbps': self.estimated_bandwidth_kbps,
            'is_connected': self.is_connected,
            'server_reachable': self.server_reachable,
            'consecutive_failures': self.consecutive_failures
        }


@dataclass
class NetworkConfig:
    """Network monitoring configuration"""
    # Primary server endpoints (RTA Dubai)
    primary_server: str = "https://api.rta.ae"
    fallback_servers: List[str] = field(default_factory=lambda: [
        "https://backup1.rta.ae",
        "https://backup2.rta.ae"
    ])

    # Health check endpoints
    health_check_path: str = "/health"
    ping_hosts: List[str] = field(default_factory=lambda: [
        "8.8.8.8",  # Google DNS
        "1.1.1.1",  # Cloudflare DNS
    ])

    # Monitoring parameters
    check_interval_seconds: float = 5.0
    timeout_seconds: float = 5.0
    ping_count: int = 3
    bandwidth_test_size_kb: int = 10

    # Thresholds
    latency_excellent_ms: float = 50.0
    latency_good_ms: float = 100.0
    latency_fair_ms: float = 200.0
    latency_poor_ms: float = 500.0

    bandwidth_excellent_kbps: float = 10000.0  # 10 Mbps
    bandwidth_good_kbps: float = 5000.0        # 5 Mbps
    bandwidth_fair_kbps: float = 1000.0        # 1 Mbps
    bandwidth_poor_kbps: float = 256.0         # 256 Kbps

    # Failure thresholds
    max_consecutive_failures: int = 3
    offline_after_failures: int = 5


class NetworkMonitor:
    """
    Monitors network connectivity and quality for adaptive transmission decisions.
    Runs background checks and provides real-time status updates.
    """

    def __init__(
        self,
        config: Optional[NetworkConfig] = None,
        on_status_change: Optional[Callable[[NetworkStatus, NetworkStatus], None]] = None,
        on_offline: Optional[Callable[[], None]] = None,
        on_reconnect: Optional[Callable[[], None]] = None
    ):
        """
        Initialize network monitor.

        Args:
            config: Network configuration
            on_status_change: Callback when status changes (old_status, new_status)
            on_offline: Callback when network goes offline
            on_reconnect: Callback when network reconnects
        """
        self.config = config or NetworkConfig()
        self.on_status_change = on_status_change
        self.on_offline = on_offline
        self.on_reconnect = on_reconnect

        self._current_status = NetworkStatus.OFFLINE
        self._current_metrics: Optional[NetworkMetrics] = None
        self._consecutive_failures = 0
        self._metrics_history: deque = deque(maxlen=1000)

        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Active server tracking
        self._active_server = self.config.primary_server
        self._server_failures: Dict[str, int] = {}

    @property
    def status(self) -> NetworkStatus:
        """Get current network status"""
        with self._lock:
            return self._current_status

    @property
    def metrics(self) -> Optional[NetworkMetrics]:
        """Get current network metrics"""
        with self._lock:
            return self._current_metrics

    @property
    def is_online(self) -> bool:
        """Check if network is online"""
        return self.status != NetworkStatus.OFFLINE

    @property
    def active_server(self) -> str:
        """Get currently active server"""
        with self._lock:
            return self._active_server

    def _ping_host(self, host: str) -> Tuple[bool, float]:
        """
        Ping a host and measure latency.

        Returns:
            Tuple of (success, latency_ms)
        """
        try:
            start = time.time()
            socket.setdefaulttimeout(self.config.timeout_seconds)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, 53))
            latency = (time.time() - start) * 1000
            return True, latency
        except Exception:
            return False, float('inf')

    def _check_server_health(self, server_url: str) -> Tuple[bool, float]:
        """
        Check if server is reachable and healthy.

        Returns:
            Tuple of (is_healthy, latency_ms)
        """
        try:
            url = f"{server_url}{self.config.health_check_path}"
            start = time.time()

            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'TrafficMonitor/1.0'}
            )
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_seconds
            ) as response:
                if response.status == 200:
                    latency = (time.time() - start) * 1000
                    return True, latency

            return False, float('inf')
        except Exception:
            return False, float('inf')

    def _estimate_bandwidth(self) -> float:
        """
        Estimate available bandwidth in Kbps.

        Returns:
            Estimated bandwidth in Kbps
        """
        # Simple bandwidth estimation using small data transfer
        try:
            # Use a small test endpoint or fallback to timing a known request
            test_data = b'x' * (self.config.bandwidth_test_size_kb * 1024)

            start = time.time()
            # Simulate by measuring response time
            success, latency = self._ping_host(self.config.ping_hosts[0])
            elapsed = time.time() - start

            if success and elapsed > 0:
                # Rough estimation based on latency
                # Lower latency generally correlates with higher bandwidth
                if latency < 50:
                    return 10000.0  # 10 Mbps
                elif latency < 100:
                    return 5000.0   # 5 Mbps
                elif latency < 200:
                    return 1000.0   # 1 Mbps
                elif latency < 500:
                    return 256.0    # 256 Kbps
                else:
                    return 64.0     # 64 Kbps

            return 0.0
        except Exception:
            return 0.0

    def _determine_status(
        self,
        latency: float,
        bandwidth: float,
        is_connected: bool
    ) -> NetworkStatus:
        """Determine network status based on metrics"""
        if not is_connected:
            return NetworkStatus.OFFLINE

        # Check latency first
        if latency <= self.config.latency_excellent_ms and bandwidth >= self.config.bandwidth_excellent_kbps:
            return NetworkStatus.EXCELLENT
        elif latency <= self.config.latency_good_ms and bandwidth >= self.config.bandwidth_good_kbps:
            return NetworkStatus.GOOD
        elif latency <= self.config.latency_fair_ms and bandwidth >= self.config.bandwidth_fair_kbps:
            return NetworkStatus.FAIR
        elif latency <= self.config.latency_poor_ms and bandwidth >= self.config.bandwidth_poor_kbps:
            return NetworkStatus.POOR
        else:
            return NetworkStatus.CRITICAL

    def _select_best_server(self) -> str:
        """Select best available server based on failures"""
        servers = [self.config.primary_server] + self.config.fallback_servers

        for server in servers:
            failures = self._server_failures.get(server, 0)
            if failures < self.config.max_consecutive_failures:
                return server

        # All servers have failures, reset and try primary
        self._server_failures.clear()
        return self.config.primary_server

    def check_connectivity(self) -> NetworkMetrics:
        """
        Perform a single connectivity check.

        Returns:
            Current network metrics
        """
        timestamp = datetime.now()

        # Ping tests
        successful_pings = 0
        total_latency = 0.0

        for host in self.config.ping_hosts[:self.config.ping_count]:
            success, latency = self._ping_host(host)
            if success:
                successful_pings += 1
                total_latency += latency

        is_connected = successful_pings > 0
        avg_latency = total_latency / successful_pings if successful_pings > 0 else float('inf')
        packet_loss = 1 - (successful_pings / len(self.config.ping_hosts[:self.config.ping_count]))

        # Check server health
        server_reachable = False
        if is_connected:
            self._active_server = self._select_best_server()
            server_reachable, server_latency = self._check_server_health(self._active_server)

            if server_reachable:
                # Use server latency if available
                avg_latency = (avg_latency + server_latency) / 2
                self._server_failures[self._active_server] = 0
            else:
                self._server_failures[self._active_server] = \
                    self._server_failures.get(self._active_server, 0) + 1

        # Estimate bandwidth
        bandwidth = self._estimate_bandwidth() if is_connected else 0.0

        # Update failure tracking
        if is_connected and server_reachable:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

        # Determine status
        if self._consecutive_failures >= self.config.offline_after_failures:
            status = NetworkStatus.OFFLINE
        else:
            status = self._determine_status(avg_latency, bandwidth, is_connected)

        metrics = NetworkMetrics(
            timestamp=timestamp,
            status=status,
            latency_ms=avg_latency if avg_latency != float('inf') else -1,
            packet_loss=packet_loss,
            estimated_bandwidth_kbps=bandwidth,
            is_connected=is_connected,
            server_reachable=server_reachable,
            consecutive_failures=self._consecutive_failures
        )

        return metrics

    def _update_status(self, metrics: NetworkMetrics):
        """Update status and trigger callbacks"""
        with self._lock:
            old_status = self._current_status
            self._current_status = metrics.status
            self._current_metrics = metrics
            self._metrics_history.append(metrics)

        # Trigger callbacks
        if old_status != metrics.status:
            if self.on_status_change:
                try:
                    self.on_status_change(old_status, metrics.status)
                except Exception as e:
                    logger.error(f"Status change callback error: {e}")

            # Check for offline/reconnect
            if metrics.status == NetworkStatus.OFFLINE and old_status != NetworkStatus.OFFLINE:
                if self.on_offline:
                    try:
                        self.on_offline()
                    except Exception as e:
                        logger.error(f"Offline callback error: {e}")

            elif old_status == NetworkStatus.OFFLINE and metrics.status != NetworkStatus.OFFLINE:
                if self.on_reconnect:
                    try:
                        self.on_reconnect()
                    except Exception as e:
                        logger.error(f"Reconnect callback error: {e}")

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                metrics = self.check_connectivity()
                self._update_status(metrics)
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

            time.sleep(self.config.check_interval_seconds)

    def start_monitoring(self):
        """Start background network monitoring"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Network monitoring started")

    def stop_monitoring(self):
        """Stop background network monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
            self._monitor_thread = None
        logger.info("Network monitoring stopped")

    def get_history(self, minutes: int = 60) -> List[NetworkMetrics]:
        """Get metrics history for the last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        with self._lock:
            return [m for m in self._metrics_history if m.timestamp >= cutoff]

    def get_statistics(self) -> Dict[str, Any]:
        """Get network statistics summary"""
        history = self.get_history(60)

        if not history:
            return {
                'samples': 0,
                'uptime_percent': 0.0,
                'avg_latency_ms': 0.0,
                'avg_packet_loss': 0.0,
                'status_distribution': {}
            }

        online_samples = sum(1 for m in history if m.status != NetworkStatus.OFFLINE)
        latencies = [m.latency_ms for m in history if m.latency_ms >= 0]
        packet_losses = [m.packet_loss for m in history]

        status_counts = {}
        for m in history:
            status_counts[m.status.value] = status_counts.get(m.status.value, 0) + 1

        return {
            'samples': len(history),
            'uptime_percent': (online_samples / len(history)) * 100 if history else 0,
            'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
            'avg_packet_loss': sum(packet_losses) / len(packet_losses) if packet_losses else 0,
            'status_distribution': status_counts,
            'current_status': self.status.value,
            'active_server': self._active_server
        }


