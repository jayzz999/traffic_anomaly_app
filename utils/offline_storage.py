"""
Offline Storage Module
Provides persistent storage for frames and events when network is unavailable.
Implements SQLite-based queue with automatic sync when connection is restored.
"""

import sqlite3
import json
import time
import threading
import logging
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue, Empty
import os

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Configuration for offline storage"""
    db_path: str = "offline_storage.db"
    max_storage_mb: int = 500  # Maximum storage size
    max_age_hours: int = 72    # Maximum age of stored items
    batch_size: int = 50       # Items per sync batch
    cleanup_interval_minutes: int = 30
    sync_interval_seconds: float = 5.0


class OfflineStorage:
    """
    SQLite-based persistent storage for offline operation.
    Stores frames and events when network is unavailable.
    """

    def __init__(self, config: Optional[StorageConfig] = None, db_path: Optional[str] = None):
        """
        Initialize offline storage.

        Args:
            config: Storage configuration
            db_path: Override database path
        """
        self.config = config or StorageConfig()
        if db_path:
            self.config.db_path = db_path

        self._db_path = Path(self.config.db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(str(self._db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Frames table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame_id TEXT UNIQUE NOT NULL,
                    camera_id TEXT NOT NULL,
                    location_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    is_priority INTEGER DEFAULT 0,
                    compression_level TEXT,
                    data_size INTEGER,
                    data_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    sync_attempts INTEGER DEFAULT 0,
                    last_sync_attempt TEXT,
                    synced INTEGER DEFAULT 0
                )
            ''')

            # Events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    camera_id TEXT NOT NULL,
                    location_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    confidence REAL,
                    data_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    sync_attempts INTEGER DEFAULT 0,
                    last_sync_attempt TEXT,
                    synced INTEGER DEFAULT 0
                )
            ''')

            # Sync log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sync_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sync_type TEXT NOT NULL,
                    item_count INTEGER NOT NULL,
                    success INTEGER NOT NULL,
                    error_message TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_frames_synced ON frames(synced)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_frames_priority ON frames(is_priority)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_frames_timestamp ON frames(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_synced ON events(synced)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity)')

            conn.commit()

    def store_frame(self, frame_data: Dict[str, Any], is_priority: bool = False) -> bool:
        """
        Store a frame for later transmission.

        Args:
            frame_data: Frame data dictionary
            is_priority: Whether this is a high-priority frame

        Returns:
            True if stored successfully
        """
        try:
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    cursor.execute('''
                        INSERT OR REPLACE INTO frames
                        (frame_id, camera_id, location_id, timestamp, is_priority,
                         compression_level, data_size, data_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        frame_data.get('frame_id'),
                        frame_data.get('metadata', {}).get('camera_id', ''),
                        frame_data.get('metadata', {}).get('location_id', ''),
                        frame_data.get('timestamp'),
                        1 if is_priority else 0,
                        frame_data.get('compression_level'),
                        len(frame_data.get('data_base64', '')),
                        json.dumps(frame_data)
                    ))

                    conn.commit()
                    return True

        except Exception as e:
            logger.error(f"Failed to store frame: {e}")
            return False

    def store_event(self, event_data: Dict[str, Any]) -> bool:
        """
        Store an event for later transmission.

        Args:
            event_data: Event data dictionary

        Returns:
            True if stored successfully
        """
        try:
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    cursor.execute('''
                        INSERT OR REPLACE INTO events
                        (event_id, camera_id, location_id, event_type, severity,
                         timestamp, confidence, data_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        event_data.get('event_id'),
                        event_data.get('camera_id', ''),
                        event_data.get('location_id', ''),
                        event_data.get('event_type', ''),
                        event_data.get('severity', 5),
                        event_data.get('timestamp'),
                        event_data.get('confidence', 0),
                        json.dumps(event_data)
                    ))

                    conn.commit()
                    return True

        except Exception as e:
            logger.error(f"Failed to store event: {e}")
            return False

    def get_pending_frames(self, limit: int = 50, priority_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get pending frames for transmission.

        Args:
            limit: Maximum number of frames to retrieve
            priority_only: Only retrieve priority frames

        Returns:
            List of frame data dictionaries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                if priority_only:
                    cursor.execute('''
                        SELECT id, data_json FROM frames
                        WHERE synced = 0 AND is_priority = 1
                        ORDER BY severity ASC, timestamp ASC
                        LIMIT ?
                    ''', (limit,))
                else:
                    cursor.execute('''
                        SELECT id, data_json FROM frames
                        WHERE synced = 0
                        ORDER BY is_priority DESC, timestamp ASC
                        LIMIT ?
                    ''', (limit,))

                rows = cursor.fetchall()
                return [(row['id'], json.loads(row['data_json'])) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get pending frames: {e}")
            return []

    def get_pending_events(self, limit: int = 50) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Get pending events for transmission.

        Args:
            limit: Maximum number of events to retrieve

        Returns:
            List of (id, event_data) tuples
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT id, data_json FROM events
                    WHERE synced = 0
                    ORDER BY severity ASC, timestamp ASC
                    LIMIT ?
                ''', (limit,))

                rows = cursor.fetchall()
                return [(row['id'], json.loads(row['data_json'])) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get pending events: {e}")
            return []

    def mark_frames_synced(self, frame_ids: List[int], success: bool = True):
        """Mark frames as synced or increment retry count"""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    if success:
                        cursor.execute(f'''
                            UPDATE frames SET synced = 1
                            WHERE id IN ({','.join('?' * len(frame_ids))})
                        ''', frame_ids)
                    else:
                        cursor.execute(f'''
                            UPDATE frames SET
                                sync_attempts = sync_attempts + 1,
                                last_sync_attempt = ?
                            WHERE id IN ({','.join('?' * len(frame_ids))})
                        ''', [datetime.now().isoformat()] + frame_ids)

                    conn.commit()

        except Exception as e:
            logger.error(f"Failed to mark frames synced: {e}")

    def mark_events_synced(self, event_ids: List[int], success: bool = True):
        """Mark events as synced or increment retry count"""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    if success:
                        cursor.execute(f'''
                            UPDATE events SET synced = 1
                            WHERE id IN ({','.join('?' * len(event_ids))})
                        ''', event_ids)
                    else:
                        cursor.execute(f'''
                            UPDATE events SET
                                sync_attempts = sync_attempts + 1,
                                last_sync_attempt = ?
                            WHERE id IN ({','.join('?' * len(event_ids))})
                        ''', [datetime.now().isoformat()] + event_ids)

                    conn.commit()

        except Exception as e:
            logger.error(f"Failed to mark events synced: {e}")

    def cleanup_old_data(self):
        """Remove old synced data and enforce storage limits"""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    # Remove old synced items
                    cutoff = (datetime.now() - timedelta(hours=self.config.max_age_hours)).isoformat()

                    cursor.execute('DELETE FROM frames WHERE synced = 1 AND created_at < ?', (cutoff,))
                    cursor.execute('DELETE FROM events WHERE synced = 1 AND created_at < ?', (cutoff,))

                    # Check storage size
                    db_size = os.path.getsize(self._db_path) / (1024 * 1024)  # MB

                    if db_size > self.config.max_storage_mb:
                        # Remove oldest synced items first
                        cursor.execute('''
                            DELETE FROM frames WHERE id IN (
                                SELECT id FROM frames WHERE synced = 1
                                ORDER BY created_at ASC LIMIT 1000
                            )
                        ''')

                        # If still too large, remove old unsynced items
                        db_size = os.path.getsize(self._db_path) / (1024 * 1024)
                        if db_size > self.config.max_storage_mb:
                            cursor.execute('''
                                DELETE FROM frames WHERE id IN (
                                    SELECT id FROM frames
                                    ORDER BY is_priority ASC, created_at ASC LIMIT 500
                                )
                            ''')

                    # Vacuum to reclaim space
                    cursor.execute('VACUUM')
                    conn.commit()

                    logger.info(f"Cleanup completed. DB size: {db_size:.2f} MB")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('SELECT COUNT(*) as count FROM frames WHERE synced = 0')
                pending_frames = cursor.fetchone()['count']

                cursor.execute('SELECT COUNT(*) as count FROM frames WHERE synced = 1')
                synced_frames = cursor.fetchone()['count']

                cursor.execute('SELECT COUNT(*) as count FROM events WHERE synced = 0')
                pending_events = cursor.fetchone()['count']

                cursor.execute('SELECT COUNT(*) as count FROM events WHERE synced = 1')
                synced_events = cursor.fetchone()['count']

                cursor.execute('SELECT SUM(data_size) as total FROM frames WHERE synced = 0')
                row = cursor.fetchone()
                pending_bytes = row['total'] if row['total'] else 0

                db_size = os.path.getsize(self._db_path) / (1024 * 1024)

                return {
                    'pending_frames': pending_frames,
                    'synced_frames': synced_frames,
                    'pending_events': pending_events,
                    'synced_events': synced_events,
                    'pending_bytes': pending_bytes,
                    'db_size_mb': db_size,
                    'max_storage_mb': self.config.max_storage_mb
                }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def log_sync(self, sync_type: str, item_count: int, success: bool, error: Optional[str] = None):
        """Log sync attempt"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO sync_log (sync_type, item_count, success, error_message)
                    VALUES (?, ?, ?, ?)
                ''', (sync_type, item_count, 1 if success else 0, error))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log sync: {e}")


class OfflineSyncManager:
    """
    Manages synchronization between offline storage and remote server.
    Automatically syncs when network becomes available.
    """

    def __init__(
        self,
        storage: OfflineStorage,
        send_frames_func: Callable[[List[Dict[str, Any]]], bool],
        send_events_func: Callable[[List[Dict[str, Any]]], bool],
        config: Optional[StorageConfig] = None
    ):
        """
        Initialize sync manager.

        Args:
            storage: Offline storage instance
            send_frames_func: Function to send frames to server
            send_events_func: Function to send events to server
            config: Storage configuration
        """
        self.storage = storage
        self.send_frames_func = send_frames_func
        self.send_events_func = send_events_func
        self.config = config or StorageConfig()

        self._running = False
        self._sync_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None
        self._is_syncing = False
        self._network_available = False

        # Statistics
        self._stats = {
            'sync_attempts': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'frames_synced': 0,
            'events_synced': 0
        }

    def set_network_available(self, available: bool):
        """Set network availability status"""
        self._network_available = available

    def _sync_loop(self):
        """Background sync loop"""
        while self._running:
            try:
                if self._network_available and not self._is_syncing:
                    self._perform_sync()
            except Exception as e:
                logger.error(f"Sync loop error: {e}")

            time.sleep(self.config.sync_interval_seconds)

    def _cleanup_loop(self):
        """Background cleanup loop"""
        while self._running:
            try:
                self.storage.cleanup_old_data()
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

            time.sleep(self.config.cleanup_interval_minutes * 60)

    def _perform_sync(self):
        """Perform synchronization"""
        self._is_syncing = True
        self._stats['sync_attempts'] += 1

        try:
            # Sync events first (higher priority)
            events = self.storage.get_pending_events(self.config.batch_size)
            if events:
                event_ids = [e[0] for e in events]
                event_data = [e[1] for e in events]

                try:
                    success = self.send_events_func(event_data)
                    self.storage.mark_events_synced(event_ids, success)
                    self.storage.log_sync('events', len(events), success)

                    if success:
                        self._stats['events_synced'] += len(events)
                        self._stats['successful_syncs'] += 1
                    else:
                        self._stats['failed_syncs'] += 1

                except Exception as e:
                    logger.error(f"Event sync failed: {e}")
                    self.storage.mark_events_synced(event_ids, False)
                    self.storage.log_sync('events', len(events), False, str(e))

            # Sync frames
            frames = self.storage.get_pending_frames(self.config.batch_size)
            if frames:
                frame_ids = [f[0] for f in frames]
                frame_data = [f[1] for f in frames]

                try:
                    success = self.send_frames_func(frame_data)
                    self.storage.mark_frames_synced(frame_ids, success)
                    self.storage.log_sync('frames', len(frames), success)

                    if success:
                        self._stats['frames_synced'] += len(frames)
                        self._stats['successful_syncs'] += 1
                    else:
                        self._stats['failed_syncs'] += 1

                except Exception as e:
                    logger.error(f"Frame sync failed: {e}")
                    self.storage.mark_frames_synced(frame_ids, False)
                    self.storage.log_sync('frames', len(frames), False, str(e))

        finally:
            self._is_syncing = False

    def start(self):
        """Start sync manager"""
        if self._running:
            return

        self._running = True

        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()

        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

        logger.info("Offline sync manager started")

    def stop(self):
        """Stop sync manager"""
        self._running = False

        if self._sync_thread:
            self._sync_thread.join(timeout=10)
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=10)

        logger.info("Offline sync manager stopped")

    def force_sync(self) -> bool:
        """Force immediate sync"""
        if not self._network_available:
            return False

        self._perform_sync()
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get sync statistics"""
        return {
            **self._stats,
            'storage': self.storage.get_statistics(),
            'network_available': self._network_available,
            'is_syncing': self._is_syncing
        }
