"""
Connection Pool — manages persistent TCP connections to downstream nodes.

The Connection Pool maintains a map of target_addr → TCP connection, reusing
connections across requests. It handles connection establishment, failure
detection, and cleanup.

Design:
- One TCP connection per node pair (not per request)
- Blocking I/O acceptable for POC
- No connection multiplexing — sequential request processing per connection
- Connection timeout: 5 seconds for establishment, 30 seconds for idle
- Connection lifecycle states: Disconnected → Connecting → Connected → Failed → Disconnected
- Single retry attempt on connection failure before reporting to Coordinator

Validates: Requirements 2.1, 2.2, 2.3
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

CONNECTION_ESTABLISHMENT_TIMEOUT = 5.0  # seconds
"""Timeout for TCP connection establishment."""

IDLE_CONNECTION_TIMEOUT = 30.0  # seconds
"""Timeout for idle connections before they are closed."""

MAX_RETRY_ATTEMPTS = 1
"""Maximum number of retry attempts for failed connections."""

# ── Enumerations ───────────────────────────────────────────────────────────

class ConnectionState(IntEnum):
    """Lifecycle state of a TCP connection in the pool."""
    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2
    FAILED = 3


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ConnectionInfo:
    """Metadata for a connection in the pool."""
    target_addr: Tuple[str, int]
    """Target address as (host, port)."""
    
    state: ConnectionState
    """Current connection state."""
    
    socket: Optional[socket.socket] = None
    """Active socket if state is CONNECTED, None otherwise."""
    
    last_activity: float = field(default_factory=time.time)
    """Timestamp of last successful send/receive (seconds since epoch)."""
    
    retry_count: int = 0
    """Number of retry attempts made for this connection."""
    
    error_message: Optional[str] = None
    """Error message if state is FAILED."""


# ── Connection Pool ──────────────────────────────────────────────────────────

class ConnectionPool:
    """Manages persistent TCP connections to downstream nodes.
    
    The pool maintains a map of target_addr → TCP connection, reusing
    connections across requests. Connections are established once and reused
    for all subsequent Forward operations to that target.
    
    Attributes:
        _connections: Dict mapping target_addr → ConnectionInfo
        _lock: Thread lock for thread-safe access to the connections dict
        _listener: Optional TCP listener socket for accepting incoming connections
        _listener_thread: Optional thread running the accept loop
        _shutdown_flag: Flag indicating shutdown has been requested
    """
    
    def __init__(self) -> None:
        """Initialize an empty connection pool."""
        self._connections: Dict[Tuple[str, int], ConnectionInfo] = {}
        self._lock = threading.Lock()
        self._listener: Optional[socket.socket] = None
        self._listener_thread: Optional[threading.Thread] = None
        self._shutdown_flag = threading.Event()
        self._incoming_connections: List[Tuple[socket.socket, Tuple[str, int]]] = []
        self._incoming_lock = threading.Lock()
        self._on_connection: Optional[Callable[[socket.socket, Tuple[str, int]], None]] = None
        self._listen_addr: Optional[Tuple[str, int]] = None
    
    # ── Public Interface ─────────────────────────────────────────────────
    
    def get_connection(self, target_addr: Tuple[str, int]) -> Optional[socket.socket]:
        """Get or establish a TCP connection to the target address.
        
        If a live connection already exists for this address, it is returned.
        Otherwise, a new connection is established with a 5-second timeout.
        
        Args:
            target_addr: Target address as (host, port).
            
        Returns:
            A connected socket if successful, None if connection failed
            (including after a single retry attempt).
            
        Raises:
            ValueError: If target_addr is not a valid (host, port) tuple.
        """
        if not isinstance(target_addr, tuple) or len(target_addr) != 2:
            raise ValueError(f"target_addr must be (host, port) tuple, got {target_addr}")
        
        host, port = target_addr
        if not isinstance(port, int) or port < 1 or port > 65535:
            raise ValueError(f"Invalid port {port}, must be 1-65535")
        
        with self._lock:
            # Check for existing connection
            if target_addr in self._connections:
                conn_info = self._connections[target_addr]
                
                # Return existing connected socket
                if conn_info.state == ConnectionState.CONNECTED and conn_info.socket:
                    # Update last activity timestamp
                    self._connections[target_addr] = ConnectionInfo(
                        target_addr=conn_info.target_addr,
                        state=conn_info.state,
                        socket=conn_info.socket,
                        last_activity=time.time(),
                        retry_count=conn_info.retry_count,
                        error_message=conn_info.error_message,
                    )
                    return conn_info.socket
                
                # Handle failed connection with retry
                if conn_info.state == ConnectionState.FAILED:
                    if conn_info.retry_count < MAX_RETRY_ATTEMPTS:
                        return self._retry_connection(target_addr, conn_info)
                    else:
                        # Max retries exhausted
                        return None
            
            # No existing connection or needs new connection
            return self._establish_connection(target_addr)
    
    def close_connection(self, target_addr: Tuple[str, int]) -> None:
        """Close a specific connection and remove it from the pool.
        
        Args:
            target_addr: Target address as (host, port).
            
        Raises:
            KeyError: If no connection exists for the target address.
        """
        with self._lock:
            if target_addr not in self._connections:
                raise KeyError(f"No connection found for {target_addr}")
            
            conn_info = self._connections.pop(target_addr)
            if conn_info.socket:
                try:
                    conn_info.socket.close()
                except OSError:
                    pass  # Socket already closed
    
    def close_all(self) -> None:
        """Gracefully close all connections in the pool.
        
        This method is called during node shutdown to clean up all resources,
        including outbound connections, the TCP listener, and any accepted
        incoming connections.
        """
        # Signal the accept loop to stop
        self._shutdown_flag.set()

        # Close outbound connections
        with self._lock:
            for conn_info in self._connections.values():
                if conn_info.socket:
                    try:
                        conn_info.socket.close()
                    except OSError:
                        pass  # Socket already closed
            self._connections.clear()

        # Close accepted incoming connections
        with self._incoming_lock:
            for client_sock, _ in self._incoming_connections:
                try:
                    client_sock.close()
                except OSError:
                    pass
            self._incoming_connections.clear()
        
        # Stop the listener thread
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=5.0)
        
        if self._listener:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None
        self._listener_thread = None
        self._on_connection = None
        self._listen_addr = None
    
    def is_connected(self, target_addr: Tuple[str, int]) -> bool:
        """Check if a live connection exists to the target address.
        
        Verifies both that the pool has a CONNECTED entry and that the
        underlying socket is still alive by performing a zero-byte,
        non-blocking peek.  If the peer has closed the connection the
        peek will reveal it and the entry is moved to FAILED state so
        subsequent callers see an accurate picture.
        
        Args:
            target_addr: Target address as (host, port).
            
        Returns:
            True if a connected, live socket exists for this address,
            False otherwise.
        """
        with self._lock:
            if target_addr not in self._connections:
                return False

            conn_info = self._connections[target_addr]
            if conn_info.state != ConnectionState.CONNECTED or conn_info.socket is None:
                return False

            # Probe the socket to detect a closed/broken connection.
            sock = conn_info.socket
            try:
                original_timeout = sock.gettimeout()
                sock.setblocking(False)
                try:
                    data = sock.recv(1, socket.MSG_PEEK)
                    # recv returns b'' when the peer has closed the connection.
                    if data == b"":
                        self._connections[target_addr] = ConnectionInfo(
                            target_addr=conn_info.target_addr,
                            state=ConnectionState.FAILED,
                            socket=None,
                            last_activity=conn_info.last_activity,
                            retry_count=conn_info.retry_count,
                            error_message="Peer closed connection",
                        )
                        try:
                            sock.close()
                        except OSError:
                            pass
                        return False
                except BlockingIOError:
                    # No data available — socket is still alive.
                    pass
                except (OSError, ConnectionError):
                    # Socket error — mark as failed.
                    self._connections[target_addr] = ConnectionInfo(
                        target_addr=conn_info.target_addr,
                        state=ConnectionState.FAILED,
                        socket=None,
                        last_activity=conn_info.last_activity,
                        retry_count=conn_info.retry_count,
                        error_message="Socket error during liveness check",
                    )
                    try:
                        sock.close()
                    except OSError:
                        pass
                    return False
                finally:
                    # Restore the original timeout so callers aren't surprised.
                    try:
                        sock.settimeout(original_timeout)
                    except OSError:
                        pass
            except OSError:
                # gettimeout / setblocking failed — socket is dead.
                self._connections[target_addr] = ConnectionInfo(
                    target_addr=conn_info.target_addr,
                    state=ConnectionState.FAILED,
                    socket=None,
                    last_activity=conn_info.last_activity,
                    retry_count=conn_info.retry_count,
                    error_message="Socket unusable",
                )
                return False

            return True
    
    def accept_incoming(
        self,
        listen_addr: Tuple[str, int],
        on_connection: Optional[Callable[[socket.socket, Tuple[str, int]], None]] = None,
        backlog: int = 5,
    ) -> None:
        """Start a TCP listener to accept incoming connections.

        Binds a server socket on *listen_addr* and spawns a daemon thread
        that continuously accepts inbound TCP connections.  Each accepted
        connection is:

        1. Recorded in the internal ``_incoming_connections`` list so the
           pool can track and clean up all accepted sockets.
        2. Handed off to the optional *on_connection* callback (typically
           the Message Handler) for processing.  The callback is invoked
           in a dedicated daemon thread so the accept loop is never
           blocked by slow consumers.

        Multiple upstream nodes can connect simultaneously — the listener
        uses a configurable *backlog* and each accepted connection is
        processed independently.

        Args:
            listen_addr: Address to listen on as (host, port).
            on_connection: Optional callback invoked for every accepted
                connection.  Signature: ``(sock, addr) -> None``.  If
                ``None``, accepted sockets are stored but no handler is
                called.
            backlog: Maximum number of queued connections passed to
                ``socket.listen()``.  Defaults to 5.

        Raises:
            RuntimeError: If the listener thread is already running.
            OSError: If the listener socket cannot be created or bound.
        """
        if self._listener_thread and self._listener_thread.is_alive():
            raise RuntimeError("Listener thread is already running")

        self._shutdown_flag.clear()
        self._on_connection = on_connection
        self._listen_addr = listen_addr

        # Create and bind listener socket
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(listen_addr)
        self._listener.listen(backlog)

        logger.info("TCP listener bound to %s:%d", listen_addr[0], listen_addr[1])

        # Start listener thread
        self._listener_thread = threading.Thread(
            target=self._accept_loop,
            args=(listen_addr,),
            daemon=True,
            name="tcp-accept-loop",
        )
        self._listener_thread.start()

    def get_incoming_connections(self) -> List[Tuple[socket.socket, Tuple[str, int]]]:
        """Return a snapshot of all accepted incoming connections.

        Returns:
            List of (socket, remote_addr) tuples for every connection
            accepted by the listener that has not been explicitly closed.
        """
        with self._incoming_lock:
            return list(self._incoming_connections)
    
    # ── Private Methods ──────────────────────────────────────────────────
    
    def _establish_connection(
        self, target_addr: Tuple[str, int], retry_count: int = 0
    ) -> Optional[socket.socket]:
        """Establish a new TCP connection with timeout handling.
        
        Args:
            target_addr: Target address as (host, port).
            retry_count: Current retry count to preserve across attempts.
            
        Returns:
            Connected socket if successful, None if connection failed.
        """
        host, port = target_addr
        
        # Update connection state to CONNECTING
        self._connections[target_addr] = ConnectionInfo(
            target_addr=target_addr,
            state=ConnectionState.CONNECTING,
            socket=None,
            last_activity=time.time(),
            retry_count=retry_count,
            error_message=None,
        )
        
        sock: Optional[socket.socket] = None
        try:
            # Create socket with timeout
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(CONNECTION_ESTABLISHMENT_TIMEOUT)
            
            # Attempt connection
            sock.connect((host, port))
            
            # Connection successful — clear the timeout so data transfers
            # use the default (blocking, no timeout) behaviour.
            sock.settimeout(None)
            
            self._connections[target_addr] = ConnectionInfo(
                target_addr=target_addr,
                state=ConnectionState.CONNECTED,
                socket=sock,
                last_activity=time.time(),
                retry_count=retry_count,
                error_message=None,
            )
            
            return sock
            
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            # Connection failed — clean up socket
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
            
            self._connections[target_addr] = ConnectionInfo(
                target_addr=target_addr,
                state=ConnectionState.FAILED,
                socket=None,
                last_activity=time.time(),
                retry_count=retry_count,
                error_message=str(e),
            )
            
            return None
    
    def _retry_connection(
        self,
        target_addr: Tuple[str, int],
        conn_info: ConnectionInfo,
    ) -> Optional[socket.socket]:
        """Retry a failed connection (single attempt).
        
        Args:
            target_addr: Target address as (host, port).
            conn_info: Current connection info for this address.
            
        Returns:
            Connected socket if retry successful, None if retry failed.
        """
        new_retry_count = conn_info.retry_count + 1
        return self._establish_connection(target_addr, retry_count=new_retry_count)
    
    def _accept_loop(self, listen_addr: Tuple[str, int]) -> None:
        """Background thread loop that accepts incoming TCP connections.

        Each accepted connection is stored in ``_incoming_connections`` and,
        if an ``on_connection`` callback was provided, handed off in a
        separate daemon thread so the accept loop remains responsive.

        Args:
            listen_addr: Address the listener is bound to (used for
                logging only).
        """
        if not self._listener:
            return

        logger.info(
            "Accept loop started on %s:%d", listen_addr[0], listen_addr[1]
        )

        while not self._shutdown_flag.is_set():
            try:
                # Use a short timeout so we can periodically check the
                # shutdown flag without blocking forever.
                self._listener.settimeout(1.0)
                client_sock, client_addr = self._listener.accept()

                logger.info("Accepted connection from %s:%d", client_addr[0], client_addr[1])

                # Track the accepted connection
                with self._incoming_lock:
                    self._incoming_connections.append((client_sock, client_addr))

                # Hand off to the callback in a dedicated thread so the
                # accept loop is never blocked by a slow handler.
                if self._on_connection is not None:
                    handler_thread = threading.Thread(
                        target=self._handle_incoming,
                        args=(client_sock, client_addr),
                        daemon=True,
                        name=f"tcp-handler-{client_addr[0]}:{client_addr[1]}",
                    )
                    handler_thread.start()

            except socket.timeout:
                # Expected — loop back and re-check shutdown flag.
                continue
            except OSError:
                # Listener socket was closed (e.g. during shutdown).
                if not self._shutdown_flag.is_set():
                    logger.warning("Listener socket error, exiting accept loop")
                break

        logger.info("Accept loop stopped")

    def _handle_incoming(
        self, client_sock: socket.socket, client_addr: Tuple[str, int]
    ) -> None:
        """Invoke the on_connection callback, catching exceptions.

        Args:
            client_sock: The accepted client socket.
            client_addr: Remote address as (host, port).
        """
        try:
            if self._on_connection is not None:
                self._on_connection(client_sock, client_addr)
        except Exception:
            logger.exception(
                "Error in connection handler for %s:%d",
                client_addr[0],
                client_addr[1],
            )
            try:
                client_sock.close()
            except OSError:
                pass
    
    def _cleanup_idle_connections(self) -> None:
        """Close connections that have been idle for too long.
        
        This method should be called periodically to clean up idle connections.
        """
        current_time = time.time()
        to_remove = []
        
        with self._lock:
            for target_addr, conn_info in self._connections.items():
                if (conn_info.state == ConnectionState.CONNECTED and 
                    (current_time - conn_info.last_activity) > IDLE_CONNECTION_TIMEOUT):
                    to_remove.append(target_addr)
            
            for target_addr in to_remove:
                conn_info = self._connections.pop(target_addr)
                if conn_info.socket:
                    try:
                        conn_info.socket.close()
                    except OSError:
                        pass