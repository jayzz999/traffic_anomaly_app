"""
RTA Dubai API Client
Handles communication with Roads and Transport Authority (RTA) Dubai traffic management system.
Implements secure, authenticated API calls with retry logic and error handling.
"""

import json
import time
import logging
import hashlib
import hmac
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib import request, parse, error
from enum import Enum
import ssl
import base64

logger = logging.getLogger(__name__)


class RTAEventType(Enum):
    """RTA event types for traffic incidents"""
    ACCIDENT = "ACCIDENT"
    FIRE = "FIRE"
    CONGESTION = "CONGESTION"
    ROAD_CLOSURE = "ROAD_CLOSURE"
    HAZARD = "HAZARD"
    NORMAL = "NORMAL"


class RTASeverity(Enum):
    """RTA severity levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5


@dataclass
class RTAConfig:
    """Configuration for RTA API client"""
    # API endpoints
    base_url: str = "https://api.rta.ae/traffic/v1"
    auth_url: str = "https://auth.rta.ae/oauth2/token"

    # Authentication
    client_id: str = ""
    client_secret: str = ""
    api_key: str = ""

    # Camera registration
    camera_id: str = ""
    location_id: str = ""
    intersection_name: str = ""
    coordinates: Tuple[float, float] = (0.0, 0.0)  # lat, lon

    # Timeouts and retries
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 2.0

    # Rate limiting
    max_requests_per_minute: int = 60
    burst_limit: int = 10

    # TLS/SSL
    verify_ssl: bool = True
    cert_path: Optional[str] = None


@dataclass
class RTAEvent:
    """RTA traffic event structure"""
    event_id: str
    event_type: RTAEventType
    severity: RTASeverity
    timestamp: datetime
    camera_id: str
    location_id: str
    coordinates: Tuple[float, float]
    confidence: float
    description: str
    image_data: Optional[str] = None  # Base64 encoded
    video_clip_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'camera_id': self.camera_id,
            'location_id': self.location_id,
            'coordinates': {
                'latitude': self.coordinates[0],
                'longitude': self.coordinates[1]
            },
            'confidence': self.confidence,
            'description': self.description,
            'image_data': self.image_data,
            'video_clip_url': self.video_clip_url,
            'metadata': self.metadata
        }


class RTAAuthenticator:
    """Handles OAuth2 authentication with RTA API"""

    def __init__(self, config: RTAConfig):
        self.config = config
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._refresh_token: Optional[str] = None

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for HTTPS requests"""
        context = ssl.create_default_context()
        if not self.config.verify_ssl:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        elif self.config.cert_path:
            context.load_verify_locations(self.config.cert_path)
        return context

    def authenticate(self) -> bool:
        """
        Authenticate with RTA API using OAuth2 client credentials.

        Returns:
            True if authentication successful
        """
        try:
            data = parse.urlencode({
                'grant_type': 'client_credentials',
                'client_id': self.config.client_id,
                'client_secret': self.config.client_secret,
                'scope': 'traffic.write traffic.read'
            }).encode('utf-8')

            req = request.Request(
                self.config.auth_url,
                data=data,
                headers={
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Accept': 'application/json'
                },
                method='POST'
            )

            context = self._create_ssl_context()

            with request.urlopen(req, timeout=self.config.timeout_seconds, context=context) as response:
                if response.status == 200:
                    result = json.loads(response.read().decode('utf-8'))
                    self._access_token = result.get('access_token')
                    expires_in = result.get('expires_in', 3600)
                    self._token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
                    self._refresh_token = result.get('refresh_token')
                    logger.info("RTA authentication successful")
                    return True

            return False

        except Exception as e:
            logger.error(f"RTA authentication failed: {e}")
            return False

    def get_token(self) -> Optional[str]:
        """Get valid access token, refreshing if necessary"""
        if self._access_token is None:
            self.authenticate()
        elif self._token_expiry and datetime.now() >= self._token_expiry:
            self.authenticate()

        return self._access_token

    def is_authenticated(self) -> bool:
        """Check if we have a valid token"""
        return (
            self._access_token is not None and
            self._token_expiry is not None and
            datetime.now() < self._token_expiry
        )


class RTAClient:
    """
    Client for RTA Dubai Traffic Management API.
    Handles event reporting, frame uploads, and status queries.
    """

    def __init__(self, config: RTAConfig):
        """
        Initialize RTA client.

        Args:
            config: RTA configuration
        """
        self.config = config
        self.authenticator = RTAAuthenticator(config)

        # Rate limiting
        self._request_times: List[datetime] = []

        # Statistics
        self._stats = {
            'requests_made': 0,
            'requests_successful': 0,
            'requests_failed': 0,
            'events_reported': 0,
            'frames_uploaded': 0,
            'bytes_transferred': 0
        }

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for HTTPS requests"""
        context = ssl.create_default_context()
        if not self.config.verify_ssl:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        elif self.config.cert_path:
            context.load_verify_locations(self.config.cert_path)
        return context

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Clean old entries
        self._request_times = [t for t in self._request_times if t > minute_ago]

        # Check limit
        if len(self._request_times) >= self.config.max_requests_per_minute:
            return False

        return True

    def _make_request(
        self,
        endpoint: str,
        method: str = 'GET',
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Make authenticated request to RTA API.

        Returns:
            Tuple of (success, response_data, error_message)
        """
        if not self._check_rate_limit():
            return False, None, "Rate limit exceeded"

        # Get auth token
        token = self.authenticator.get_token()
        if not token and self.config.client_id:  # Only require token if OAuth configured
            return False, None, "Authentication failed"

        # Build URL
        url = f"{self.config.base_url}{endpoint}"

        # Build headers
        req_headers = {
            'Accept': 'application/json',
            'User-Agent': 'TrafficAnomalyDetector/1.0',
            'X-Camera-ID': self.config.camera_id,
            'X-Location-ID': self.config.location_id
        }

        if token:
            req_headers['Authorization'] = f'Bearer {token}'
        elif self.config.api_key:
            req_headers['X-API-Key'] = self.config.api_key

        if headers:
            req_headers.update(headers)

        # Prepare data
        body = None
        if data:
            body = json.dumps(data).encode('utf-8')
            req_headers['Content-Type'] = 'application/json'

        # Make request with retries
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                req = request.Request(url, data=body, headers=req_headers, method=method)
                context = self._create_ssl_context()

                self._request_times.append(datetime.now())
                self._stats['requests_made'] += 1

                with request.urlopen(
                    req,
                    timeout=self.config.timeout_seconds,
                    context=context
                ) as response:
                    if response.status in [200, 201, 202]:
                        result = json.loads(response.read().decode('utf-8'))
                        self._stats['requests_successful'] += 1
                        return True, result, None
                    else:
                        last_error = f"HTTP {response.status}"

            except error.HTTPError as e:
                last_error = f"HTTP {e.code}: {e.reason}"
                if e.code == 401:
                    # Token expired, re-authenticate
                    self.authenticator.authenticate()
                elif e.code == 429:
                    # Rate limited, wait longer
                    time.sleep(self.config.retry_delay_seconds * 2)
                    continue

            except error.URLError as e:
                last_error = f"URL Error: {e.reason}"

            except Exception as e:
                last_error = str(e)

            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_delay_seconds * (attempt + 1))

        self._stats['requests_failed'] += 1
        return False, None, last_error

    def register_camera(self) -> Tuple[bool, Optional[str]]:
        """
        Register camera with RTA system.

        Returns:
            Tuple of (success, registration_id or error)
        """
        data = {
            'camera_id': self.config.camera_id,
            'location_id': self.config.location_id,
            'intersection_name': self.config.intersection_name,
            'coordinates': {
                'latitude': self.config.coordinates[0],
                'longitude': self.config.coordinates[1]
            },
            'capabilities': {
                'anomaly_detection': True,
                'edge_processing': True,
                'classes': ['accident', 'fire', 'dense_traffic', 'sparse_traffic']
            },
            'status': 'ACTIVE'
        }

        success, response, error = self._make_request('/cameras/register', method='POST', data=data)

        if success and response:
            return True, response.get('registration_id')
        return False, error

    def report_event(self, event: RTAEvent) -> Tuple[bool, Optional[str]]:
        """
        Report traffic event to RTA.

        Args:
            event: Traffic event to report

        Returns:
            Tuple of (success, event_id or error)
        """
        success, response, error = self._make_request(
            '/events',
            method='POST',
            data=event.to_dict()
        )

        if success:
            self._stats['events_reported'] += 1
            return True, response.get('event_id') if response else event.event_id

        return False, error

    def upload_frames(self, frames: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """
        Upload processed frames to RTA.

        Args:
            frames: List of processed frame dictionaries

        Returns:
            Tuple of (success, batch_id or error)
        """
        data = {
            'camera_id': self.config.camera_id,
            'location_id': self.config.location_id,
            'timestamp': datetime.now().isoformat(),
            'frame_count': len(frames),
            'frames': frames
        }

        # Calculate bytes
        bytes_transferred = len(json.dumps(data))

        success, response, error = self._make_request(
            '/frames/batch',
            method='POST',
            data=data
        )

        if success:
            self._stats['frames_uploaded'] += len(frames)
            self._stats['bytes_transferred'] += bytes_transferred
            return True, response.get('batch_id') if response else None

        return False, error

    def send_heartbeat(self) -> bool:
        """
        Send heartbeat to RTA to indicate camera is active.

        Returns:
            True if successful
        """
        data = {
            'camera_id': self.config.camera_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'ACTIVE',
            'metrics': {
                'uptime_seconds': int(time.time()),
                'frames_processed': self._stats.get('frames_uploaded', 0),
                'events_reported': self._stats.get('events_reported', 0)
            }
        }

        success, _, _ = self._make_request('/cameras/heartbeat', method='POST', data=data)
        return success

    def get_camera_status(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Get camera status from RTA.

        Returns:
            Tuple of (success, status_data or None)
        """
        success, response, _ = self._make_request(f'/cameras/{self.config.camera_id}/status')
        return success, response

    def get_events_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> Tuple[bool, Optional[List[Dict[str, Any]]]]:
        """
        Get event history for this camera.

        Returns:
            Tuple of (success, events_list or None)
        """
        params = {'limit': limit}
        if start_time:
            params['start_time'] = start_time.isoformat()
        if end_time:
            params['end_time'] = end_time.isoformat()

        query = parse.urlencode(params)
        success, response, _ = self._make_request(
            f'/cameras/{self.config.camera_id}/events?{query}'
        )

        if success and response:
            return True, response.get('events', [])
        return False, None

    def create_event_from_detection(
        self,
        anomaly_class: str,
        confidence: float,
        image_base64: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RTAEvent:
        """
        Create RTA event from anomaly detection result.

        Args:
            anomaly_class: Detected anomaly class
            confidence: Detection confidence
            image_base64: Optional base64 encoded image
            metadata: Additional metadata

        Returns:
            RTAEvent object
        """
        # Map anomaly class to RTA event type
        event_type_map = {
            'accident': RTAEventType.ACCIDENT,
            'fire': RTAEventType.FIRE,
            'dense_traffic': RTAEventType.CONGESTION,
            'sparse_traffic': RTAEventType.NORMAL
        }

        # Map to severity
        severity_map = {
            'accident': RTASeverity.CRITICAL,
            'fire': RTASeverity.CRITICAL,
            'dense_traffic': RTASeverity.MEDIUM,
            'sparse_traffic': RTASeverity.INFO
        }

        # Generate descriptions
        description_map = {
            'accident': 'Traffic accident detected by AI system',
            'fire': 'Fire or smoke detected near roadway',
            'dense_traffic': 'Heavy traffic congestion detected',
            'sparse_traffic': 'Normal traffic flow'
        }

        # Generate unique event ID
        event_id = f"{self.config.camera_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        return RTAEvent(
            event_id=event_id,
            event_type=event_type_map.get(anomaly_class, RTAEventType.NORMAL),
            severity=severity_map.get(anomaly_class, RTASeverity.INFO),
            timestamp=datetime.now(),
            camera_id=self.config.camera_id,
            location_id=self.config.location_id,
            coordinates=self.config.coordinates,
            confidence=confidence,
            description=description_map.get(anomaly_class, 'Traffic event detected'),
            image_data=image_base64,
            metadata=metadata or {}
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            **self._stats,
            'is_authenticated': self.authenticator.is_authenticated(),
            'camera_id': self.config.camera_id,
            'location_id': self.config.location_id
        }


class RTAIntegration:
    """
    High-level integration class that combines RTA client with edge processing.
    Provides simple interface for the main application.
    """

    def __init__(
        self,
        config: RTAConfig,
        heartbeat_interval_seconds: int = 60
    ):
        """
        Initialize RTA integration.

        Args:
            config: RTA configuration
            heartbeat_interval_seconds: Interval for heartbeat messages
        """
        self.config = config
        self.client = RTAClient(config)
        self.heartbeat_interval = heartbeat_interval_seconds

        self._running = False
        self._heartbeat_thread = None
        self._registered = False

    def start(self) -> bool:
        """Start RTA integration"""
        # Register camera
        success, result = self.client.register_camera()
        if success:
            self._registered = True
            logger.info(f"Camera registered with RTA: {result}")
        else:
            logger.warning(f"Camera registration failed: {result}")
            # Continue anyway, might work without registration

        # Start heartbeat thread
        self._running = True
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        return True

    def stop(self):
        """Stop RTA integration"""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
            self._heartbeat_thread = None

    def _heartbeat_loop(self):
        """Background heartbeat loop"""
        while self._running:
            try:
                self.client.send_heartbeat()
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")

            time.sleep(self.heartbeat_interval)

    def report_anomaly(
        self,
        anomaly_class: str,
        confidence: float,
        image_base64: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Report detected anomaly to RTA.

        Args:
            anomaly_class: Detected class
            confidence: Detection confidence
            image_base64: Optional image data
            metadata: Additional metadata

        Returns:
            True if reported successfully
        """
        event = self.client.create_event_from_detection(
            anomaly_class=anomaly_class,
            confidence=confidence,
            image_base64=image_base64,
            metadata=metadata
        )

        success, result = self.client.report_event(event)

        if success:
            logger.info(f"Anomaly reported to RTA: {event.event_id}")
        else:
            logger.error(f"Failed to report anomaly: {result}")

        return success

    def upload_frames(self, frames: List[Dict[str, Any]]) -> bool:
        """Upload frames to RTA"""
        success, result = self.client.upload_frames(frames)
        return success

    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            'registered': self._registered,
            'running': self._running,
            **self.client.get_statistics()
        }


# Import threading at the top level
import threading
