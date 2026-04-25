"""Unit tests for Header.validate() — Task 1.4."""

import struct
import pytest

from meshrun.worker.protocol import (
    DTYPE_SIZE,
    DType,
    Header,
    MessageType,
    MAX_DIMS,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _valid_header(**overrides) -> Header:
    """Return a valid Header, with optional field overrides."""
    defaults = dict(
        message_type=MessageType.FORWARD,
        request_id=1,
        step_id=0,
        payload_size=2 * 10,  # 10 fp16 elements = 20 bytes
        dtype=DType.FP16,
        num_dims=1,
        dims=(10, 0, 0, 0),
    )
    defaults.update(overrides)
    return Header(**defaults)


# ── Valid headers pass validation ────────────────────────────────────────────

class TestValidateAccepts:
    def test_valid_1d_fp16(self):
        h = _valid_header()
        h.validate()  # should not raise

    def test_valid_2d_int8(self):
        h = _valid_header(
            dtype=DType.INT8,
            num_dims=2,
            dims=(4, 8, 0, 0),
            payload_size=4 * 8 * 1,
        )
        h.validate()

    def test_valid_4d_fp16(self):
        h = _valid_header(
            num_dims=4,
            dims=(2, 3, 4, 5),
            payload_size=2 * 3 * 4 * 5 * 2,
        )
        h.validate()

    def test_all_message_types(self):
        for mt in MessageType:
            h = _valid_header(message_type=mt)
            h.validate()


# ── Invalid message_type ─────────────────────────────────────────────────────

class TestValidateMessageType:
    @pytest.mark.parametrize("bad_mt", [0, 5, 255])
    def test_rejects_invalid_message_type(self, bad_mt):
        h = _valid_header(message_type=bad_mt)
        with pytest.raises(ValueError, match="message_type"):
            h.validate()


# ── Invalid dtype ────────────────────────────────────────────────────────────

class TestValidateDtype:
    @pytest.mark.parametrize("bad_dt", [0, 3, 255])
    def test_rejects_invalid_dtype(self, bad_dt):
        h = _valid_header(dtype=bad_dt)
        with pytest.raises(ValueError, match="dtype"):
            h.validate()


# ── Invalid num_dims ─────────────────────────────────────────────────────────

class TestValidateNumDims:
    @pytest.mark.parametrize("bad_nd", [0, 5, 255])
    def test_rejects_invalid_num_dims(self, bad_nd):
        h = _valid_header(num_dims=bad_nd, dims=(1, 0, 0, 0))
        with pytest.raises(ValueError, match="num_dims"):
            h.validate()


# ── Dims consistency ─────────────────────────────────────────────────────────

class TestValidateDimsConsistency:
    def test_rejects_zero_active_dim(self):
        h = _valid_header(num_dims=2, dims=(0, 5, 0, 0), payload_size=0)
        with pytest.raises(ValueError, match=r"dims\[0\].*> 0"):
            h.validate()

    def test_rejects_nonzero_unused_dim(self):
        h = _valid_header(
            num_dims=1,
            dims=(10, 99, 0, 0),
            payload_size=10 * 2,
        )
        with pytest.raises(ValueError, match=r"dims\[1\].*0"):
            h.validate()

    def test_rejects_nonzero_trailing_dim(self):
        h = _valid_header(
            num_dims=2,
            dims=(2, 3, 0, 7),
            payload_size=2 * 3 * 2,
        )
        with pytest.raises(ValueError, match=r"dims\[3\].*0"):
            h.validate()


# ── Payload size mismatch ────────────────────────────────────────────────────

class TestValidatePayloadSize:
    def test_rejects_wrong_payload_size(self):
        h = _valid_header(
            num_dims=1,
            dims=(10, 0, 0, 0),
            payload_size=999,
        )
        with pytest.raises(ValueError, match="payload_size"):
            h.validate()

    def test_payload_size_int8(self):
        # 4*8 = 32 elements, int8 = 1 byte each → 32 bytes
        h = _valid_header(
            dtype=DType.INT8,
            num_dims=2,
            dims=(4, 8, 0, 0),
            payload_size=32,
        )
        h.validate()  # should pass

    def test_payload_size_fp16(self):
        # 4*8 = 32 elements, fp16 = 2 bytes each → 64 bytes
        h = _valid_header(
            dtype=DType.FP16,
            num_dims=2,
            dims=(4, 8, 0, 0),
            payload_size=64,
        )
        h.validate()  # should pass


# ── unpack() auto-validates ──────────────────────────────────────────────────

class TestUnpackValidation:
    def test_unpack_rejects_invalid_header(self):
        """unpack() should call validate() and reject bad headers."""
        bad = _valid_header(message_type=0)
        raw = bad.pack()
        with pytest.raises(ValueError, match="message_type"):
            Header.unpack(raw)


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1.5 — Round-trip, size invariant, and validation-via-unpack tests
# ═══════════════════════════════════════════════════════════════════════════════

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from meshrun.worker.protocol import HEADER_SIZE


# ── Hypothesis strategies ────────────────────────────────────────────────────

_message_types = st.sampled_from([int(mt) for mt in MessageType])
_dtypes = st.sampled_from([int(dt) for dt in DType])


@st.composite
def valid_headers(draw):
    """Generate an arbitrary valid Header."""
    msg_type = draw(_message_types)
    request_id = draw(st.integers(min_value=0, max_value=2**32 - 1))
    step_id = draw(st.integers(min_value=0, max_value=2**32 - 1))
    dtype = draw(_dtypes)
    num_dims = draw(st.integers(min_value=1, max_value=MAX_DIMS))

    # Active dims > 0, keep product reasonable to avoid overflow in payload_size (uint32)
    active = draw(
        st.lists(
            st.integers(min_value=1, max_value=256),
            min_size=num_dims,
            max_size=num_dims,
        )
    )
    dims_list = list(active) + [0] * (MAX_DIMS - num_dims)
    dims = tuple(dims_list)

    product = 1
    for d in active:
        product *= d
    payload_size = product * DTYPE_SIZE[DType(dtype)]

    # Ensure payload_size fits in uint32
    assume(payload_size <= 2**32 - 1)

    return Header(
        message_type=msg_type,
        request_id=request_id,
        step_id=step_id,
        payload_size=payload_size,
        dtype=dtype,
        num_dims=num_dims,
        dims=dims,
    )


# ── Property: round-trip (pack → unpack) ─────────────────────────────────────

class TestRoundTrip:
    """**Validates: Requirements 1.1**

    For any valid header, Header.unpack(header.pack()) must produce an
    identical Header (all fields equal).
    """

    @given(header=valid_headers())
    @settings(max_examples=200)
    def test_roundtrip_property(self, header: Header):
        """Property: pack then unpack is identity for valid headers."""
        raw = header.pack()
        restored = Header.unpack(raw)
        assert restored == header

    # Deterministic edge-case round-trips

    def test_roundtrip_forward_fp16_1d(self):
        h = _valid_header()
        assert Header.unpack(h.pack()) == h

    def test_roundtrip_result_int8_2d(self):
        h = _valid_header(
            message_type=MessageType.RESULT,
            dtype=DType.INT8,
            num_dims=2,
            dims=(4, 8, 0, 0),
            payload_size=4 * 8 * 1,
        )
        assert Header.unpack(h.pack()) == h

    def test_roundtrip_error_fp16_3d(self):
        h = _valid_header(
            message_type=MessageType.ERROR,
            dtype=DType.FP16,
            num_dims=3,
            dims=(2, 3, 4, 0),
            payload_size=2 * 3 * 4 * 2,
        )
        assert Header.unpack(h.pack()) == h

    def test_roundtrip_heartbeat_int8_4d(self):
        h = _valid_header(
            message_type=MessageType.HEARTBEAT_DATA,
            dtype=DType.INT8,
            num_dims=4,
            dims=(1, 2, 3, 4),
            payload_size=1 * 2 * 3 * 4 * 1,
        )
        assert Header.unpack(h.pack()) == h

    def test_roundtrip_large_request_id_and_step_id(self):
        h = _valid_header(request_id=2**32 - 1, step_id=2**32 - 1)
        assert Header.unpack(h.pack()) == h

    def test_roundtrip_min_values(self):
        h = _valid_header(
            request_id=0,
            step_id=0,
            num_dims=1,
            dims=(1, 0, 0, 0),
            payload_size=1 * DTYPE_SIZE[DType.FP16],
        )
        assert Header.unpack(h.pack()) == h


# ── Property: 32-byte size invariant ─────────────────────────────────────────

class TestHeaderSizeInvariant:
    """**Validates: Requirements 1.1**

    Every serialized header must be exactly 32 bytes regardless of field values.
    """

    @given(header=valid_headers())
    @settings(max_examples=200)
    def test_pack_size_property(self, header: Header):
        """Property: pack() always produces exactly HEADER_SIZE bytes."""
        assert len(header.pack()) == HEADER_SIZE

    # Deterministic checks across all message types and dtypes

    def test_size_all_message_types(self):
        for mt in MessageType:
            h = _valid_header(message_type=mt)
            assert len(h.pack()) == HEADER_SIZE

    def test_size_all_dtypes(self):
        for dt in DType:
            size = DTYPE_SIZE[dt]
            h = _valid_header(
                dtype=dt,
                num_dims=1,
                dims=(10, 0, 0, 0),
                payload_size=10 * size,
            )
            assert len(h.pack()) == HEADER_SIZE

    def test_size_various_dim_counts(self):
        configs = [
            (1, (5, 0, 0, 0), 5 * 2),
            (2, (3, 4, 0, 0), 3 * 4 * 2),
            (3, (2, 3, 4, 0), 2 * 3 * 4 * 2),
            (4, (1, 2, 3, 4), 1 * 2 * 3 * 4 * 2),
        ]
        for nd, dims, ps in configs:
            h = _valid_header(num_dims=nd, dims=dims, payload_size=ps)
            assert len(h.pack()) == HEADER_SIZE

    def test_size_max_uint32_fields(self):
        """Large field values still produce 32 bytes."""
        h = _valid_header(request_id=2**32 - 1, step_id=2**32 - 1)
        assert len(h.pack()) == HEADER_SIZE


# ── Validation rejection via unpack ──────────────────────────────────────────

class TestUnpackRejectsInvalid:
    """**Validates: Requirements 1.2**

    Header.unpack() calls validate() internally, so feeding it serialized
    bytes from an invalid header must raise ValueError.
    """

    def test_rejects_bad_message_type(self):
        bad = _valid_header(message_type=0)
        with pytest.raises(ValueError, match="message_type"):
            Header.unpack(bad.pack())

    def test_rejects_bad_dtype(self):
        bad = _valid_header(dtype=0)
        with pytest.raises(ValueError, match="dtype"):
            Header.unpack(bad.pack())

    def test_rejects_bad_num_dims_zero(self):
        bad = _valid_header(num_dims=0, dims=(0, 0, 0, 0))
        with pytest.raises(ValueError, match="num_dims"):
            Header.unpack(bad.pack())

    def test_rejects_bad_num_dims_five(self):
        bad = _valid_header(num_dims=5, dims=(1, 1, 1, 1))
        with pytest.raises(ValueError, match="num_dims"):
            Header.unpack(bad.pack())

    def test_rejects_payload_mismatch(self):
        bad = _valid_header(payload_size=999)
        with pytest.raises(ValueError, match="payload_size"):
            Header.unpack(bad.pack())

    def test_rejects_nonzero_unused_dim(self):
        bad = _valid_header(num_dims=1, dims=(10, 7, 0, 0), payload_size=10 * 2)
        with pytest.raises(ValueError, match=r"dims\[1\]"):
            Header.unpack(bad.pack())

    def test_rejects_zero_active_dim(self):
        bad = _valid_header(num_dims=2, dims=(0, 5, 0, 0), payload_size=0)
        with pytest.raises(ValueError, match=r"dims\[0\]"):
            Header.unpack(bad.pack())

    def test_rejects_wrong_data_length(self):
        """unpack() rejects buffers that are not exactly 32 bytes."""
        with pytest.raises(ValueError, match="Expected 32 bytes"):
            Header.unpack(b"\x00" * 16)

        with pytest.raises(ValueError, match="Expected 32 bytes"):
            Header.unpack(b"\x00" * 64)

        with pytest.raises(ValueError, match="Expected 32 bytes"):
            Header.unpack(b"")


# ═══════════════════════════════════════════════════════════════════════════════
# Task 2.1 — read_exact unit tests
# ═══════════════════════════════════════════════════════════════════════════════

import socket
from unittest.mock import MagicMock

from meshrun.worker.protocol import read_exact


class TestReadExactNormalRead:
    """recv() returns all requested bytes in a single call."""

    def test_single_recv_returns_all(self):
        sock = MagicMock()
        sock.recv.return_value = b"hello world"
        result = read_exact(sock, 11)
        assert result == b"hello world"
        sock.recv.assert_called_once_with(11)

    def test_zero_bytes(self):
        sock = MagicMock()
        result = read_exact(sock, 0)
        assert result == b""
        sock.recv.assert_not_called()


class TestReadExactPartialReads:
    """recv() returns data in multiple chunks (simulated partial reads)."""

    def test_two_chunks(self):
        sock = MagicMock()
        sock.recv.side_effect = [b"hel", b"lo"]
        result = read_exact(sock, 5)
        assert result == b"hello"
        assert sock.recv.call_count == 2

    def test_byte_at_a_time(self):
        sock = MagicMock()
        data = b"abcd"
        sock.recv.side_effect = [bytes([b]) for b in data]
        result = read_exact(sock, 4)
        assert result == data
        assert sock.recv.call_count == 4

    def test_three_uneven_chunks(self):
        sock = MagicMock()
        sock.recv.side_effect = [b"ab", b"cde", b"f"]
        result = read_exact(sock, 6)
        assert result == b"abcdef"
        assert sock.recv.call_count == 3


class TestReadExactEOF:
    """recv() returns empty bytes (EOF) before n bytes are read."""

    def test_eof_immediately(self):
        sock = MagicMock()
        sock.recv.return_value = b""
        with pytest.raises(ConnectionError, match="0 of 10"):
            read_exact(sock, 10)

    def test_eof_after_partial(self):
        sock = MagicMock()
        sock.recv.side_effect = [b"abc", b""]
        with pytest.raises(ConnectionError, match="3 of 10"):
            read_exact(sock, 10)


class TestReadExactTimeout:
    """recv() raises socket.timeout."""

    def test_timeout_propagated(self):
        sock = MagicMock()
        sock.recv.side_effect = socket.timeout("timed out")
        with pytest.raises(socket.timeout):
            read_exact(sock, 10)

    def test_timeout_after_partial(self):
        sock = MagicMock()
        sock.recv.side_effect = [b"abc", socket.timeout("timed out")]
        with pytest.raises(socket.timeout):
            read_exact(sock, 10)


# ═══════════════════════════════════════════════════════════════════════════════
# Task 2.2 — write_all unit tests
# ═══════════════════════════════════════════════════════════════════════════════

from meshrun.worker.protocol import write_all


class TestWriteAllSingleSend:
    """send() accepts all bytes in a single call."""

    def test_single_send_writes_all(self):
        sock = MagicMock()
        data = b"hello world"
        sock.send.return_value = len(data)
        write_all(sock, data)
        sock.send.assert_called_once_with(data)


class TestWriteAllEmptyData:
    """Empty data should return immediately without calling send()."""

    def test_empty_bytes(self):
        sock = MagicMock()
        write_all(sock, b"")
        sock.send.assert_not_called()

    def test_empty_bytearray(self):
        sock = MagicMock()
        write_all(sock, bytearray())
        sock.send.assert_not_called()


class TestWriteAllPartialWrites:
    """send() returns fewer bytes than provided, function loops to completion."""

    def test_two_partial_sends(self):
        sock = MagicMock()
        data = b"hello"
        sock.send.side_effect = [3, 2]
        write_all(sock, data)
        assert sock.send.call_count == 2
        # First call sends data[0:], second sends data[3:]
        sock.send.assert_any_call(data[0:])
        sock.send.assert_any_call(data[3:])

    def test_byte_at_a_time(self):
        sock = MagicMock()
        data = b"abcd"
        sock.send.side_effect = [1, 1, 1, 1]
        write_all(sock, data)
        assert sock.send.call_count == 4


class TestWriteAllConnectionBroken:
    """send() returns 0 — connection is broken."""

    def test_send_returns_zero_immediately(self):
        sock = MagicMock()
        sock.send.return_value = 0
        with pytest.raises(ConnectionError, match="0 of 5"):
            write_all(sock, b"hello")

    def test_send_returns_zero_after_partial(self):
        sock = MagicMock()
        sock.send.side_effect = [3, 0]
        with pytest.raises(ConnectionError, match="3 of 5"):
            write_all(sock, b"hello")


class TestWriteAllTimeout:
    """send() raises socket.timeout."""

    def test_timeout_propagated(self):
        sock = MagicMock()
        sock.send.side_effect = socket.timeout("timed out")
        with pytest.raises(socket.timeout):
            write_all(sock, b"hello")

    def test_timeout_after_partial(self):
        sock = MagicMock()
        sock.send.side_effect = [3, socket.timeout("timed out")]
        with pytest.raises(socket.timeout):
            write_all(sock, b"hello")



# ═══════════════════════════════════════════════════════════════════════════════
# Task 2.3 — tensor_to_bytes unit tests
# ═══════════════════════════════════════════════════════════════════════════════

from meshrun.worker.protocol import tensor_to_bytes, DType


class TestTensorToBytesFP16:
    """**Validates: Requirements 1.4**
    
    fp16 tensors are serialized as little-endian IEEE 754 half-precision bytes.
    """

    def test_fp16_single_element(self):
        """Single fp16 element serialization."""
        result = tensor_to_bytes([1.5], DType.FP16)
        # 1.5 in fp16 little-endian: 0x3E00
        assert result == b'\x00\x3E'
        assert len(result) == 2  # fp16 is 2 bytes per element

    def test_fp16_multiple_elements(self):
        """Multiple fp16 elements serialized in row-major order."""
        result = tensor_to_bytes([1.0, 2.0, 3.0, 4.0], DType.FP16)
        # 1.0 = 0x3C00, 2.0 = 0x4000, 3.0 = 0x4200, 4.0 = 0x4400
        # Little-endian: 00 3C 00 40 00 42 00 44
        expected = b'\x00\x3C\x00\x40\x00\x42\x00\x44'
        assert result == expected
        assert len(result) == 8  # 4 elements * 2 bytes

    def test_fp16_negative_values(self):
        """Negative fp16 values."""
        result = tensor_to_bytes([-1.5], DType.FP16)
        # -1.5 in fp16: 0xBE00
        assert result == b'\x00\xBE'

    def test_fp16_zero(self):
        """Zero value."""
        result = tensor_to_bytes([0.0], DType.FP16)
        # 0.0 in fp16: 0x0000
        assert result == b'\x00\x00'

    def test_fp16_special_values(self):
        """Special fp16 values (NaN, infinity)."""
        import math
        result = tensor_to_bytes([math.inf, -math.inf, math.nan], DType.FP16)
        # inf = 0x7C00, -inf = 0xFC00, nan = 0x7E00
        expected = b'\x00\x7C\x00\xFC\x00\x7E'
        assert result == expected

    def test_fp16_empty_list(self):
        """Empty tensor."""
        result = tensor_to_bytes([], DType.FP16)
        assert result == b''
        assert len(result) == 0


class TestTensorToBytesINT8:
    """**Validates: Requirements 1.4**
    
    int8 tensors are serialized as signed 8-bit integer bytes.
    """

    def test_int8_single_element(self):
        """Single int8 element serialization."""
        result = tensor_to_bytes([42], DType.INT8)
        assert result == b'\x2A'  # 42 in hex
        assert len(result) == 1  # int8 is 1 byte per element

    def test_int8_multiple_elements(self):
        """Multiple int8 elements serialized in row-major order."""
        result = tensor_to_bytes([1, -2, 3, -4], DType.INT8)
        # 1 = 0x01, -2 = 0xFE, 3 = 0x03, -4 = 0xFC
        expected = b'\x01\xFE\x03\xFC'
        assert result == expected
        assert len(result) == 4  # 4 elements * 1 byte

    def test_int8_boundary_values(self):
        """int8 boundary values (-128 to 127)."""
        result = tensor_to_bytes([-128, 0, 127], DType.INT8)
        # -128 = 0x80, 0 = 0x00, 127 = 0x7F
        expected = b'\x80\x00\x7F'
        assert result == expected

    def test_int8_empty_list(self):
        """Empty tensor."""
        result = tensor_to_bytes([], DType.INT8)
        assert result == b''
        assert len(result) == 0


class TestTensorToBytesValidation:
    """Validation and error handling for tensor_to_bytes."""

    def test_invalid_dtype(self):
        """Reject invalid dtype values."""
        with pytest.raises(ValueError, match="Invalid dtype"):
            tensor_to_bytes([1.0], 0)
        
        with pytest.raises(ValueError, match="Invalid dtype"):
            tensor_to_bytes([1], 99)

    def test_fp16_with_int_values(self):
        """fp16 accepts integer values (they get converted to float)."""
        result = tensor_to_bytes([1, 2, 3], DType.FP16)
        # Should still serialize correctly
        assert len(result) == 6  # 3 elements * 2 bytes

    def test_int8_with_float_values(self):
        """int8 accepts float values (they get truncated to int)."""
        result = tensor_to_bytes([1.5, 2.7, -3.2], DType.INT8)
        # Should truncate: 1, 2, -3
        expected = b'\x01\x02\xFD'
        assert result == expected

    def test_int8_out_of_range_positive(self):
        """int8 values > 127 should raise ValueError."""
        with pytest.raises(ValueError, match="out of int8 range"):
            tensor_to_bytes([128], DType.INT8)

    def test_int8_out_of_range_negative(self):
        """int8 values < -128 should raise ValueError."""
        with pytest.raises(ValueError, match="out of int8 range"):
            tensor_to_bytes([-129], DType.INT8)


class TestTensorToBytesPropertyBased:
    """Property-based tests for tensor_to_bytes using Hypothesis."""
    
    from hypothesis import given, settings, assume
    from hypothesis import strategies as st
    
    @given(
        elements=st.lists(st.floats(allow_infinity=False, allow_nan=False, min_value=-65504.0, max_value=65504.0), min_size=0, max_size=100),
        dtype=st.sampled_from([DType.FP16])
    )
    @settings(max_examples=100)
    def test_fp16_roundtrip_property(self, elements, dtype):
        """**Validates: Requirements 1.4**
        
        Property: For any list of floats within fp16 range, serializing to fp16 bytes and 
        then deserializing (using struct.unpack) should reproduce the 
        original values (allowing for fp16 precision loss).
        """
        # Skip if empty (nothing to test)
        assume(len(elements) > 0)
        
        # Serialize
        serialized = tensor_to_bytes(elements, dtype)
        
        # Deserialize using struct.unpack
        fmt = f"<{len(elements)}e"
        deserialized = struct.unpack(fmt, serialized)
        
        # Compare (allowing for fp16 precision differences)
        for orig, deser in zip(elements, deserialized):
            # Check they're approximately equal (fp16 has ~3 decimal digits precision)
            # Use relative error for non-zero values, absolute error for near-zero
            if abs(orig) > 1e-5:
                assert abs(orig - deser) / abs(orig) < 1e-3
            else:
                assert abs(orig - deser) < 1e-5
    
    @given(
        elements=st.lists(st.integers(min_value=-128, max_value=127), min_size=0, max_size=100),
        dtype=st.sampled_from([DType.INT8])
    )
    @settings(max_examples=100)
    def test_int8_roundtrip_property(self, elements, dtype):
        """**Validates: Requirements 1.4**
        
        Property: For any list of int8-compatible integers, serializing 
        to int8 bytes and then deserializing should reproduce the exact 
        original values (bitwise identical).
        """
        # Serialize
        serialized = tensor_to_bytes(elements, dtype)
        
        # Deserialize using struct.unpack
        fmt = f"<{len(elements)}b"
        deserialized = struct.unpack(fmt, serialized)
        
        # Should be exactly equal for int8
        assert list(deserialized) == elements
    
    @given(
        elements=st.lists(st.integers(min_value=-128, max_value=127), min_size=1, max_size=100),
        dtype=st.sampled_from([DType.INT8])
    )
    @settings(max_examples=50)
    def test_int8_length_property(self, elements, dtype):
        """Property: Serialized byte length = len(elements) * DTYPE_SIZE[dtype]."""
        serialized = tensor_to_bytes(elements, dtype)
        expected_length = len(elements) * DTYPE_SIZE[dtype]
        assert len(serialized) == expected_length
    
    @given(
        elements=st.lists(st.floats(allow_infinity=False, allow_nan=False, min_value=-65504.0, max_value=65504.0), min_size=1, max_size=100),
        dtype=st.sampled_from([DType.FP16])
    )
    @settings(max_examples=50)
    def test_fp16_length_property(self, elements, dtype):
        """Property: Serialized byte length = len(elements) * DTYPE_SIZE[dtype]."""
        serialized = tensor_to_bytes(elements, dtype)
        expected_length = len(elements) * DTYPE_SIZE[dtype]
        assert len(serialized) == expected_length



# ═══════════════════════════════════════════════════════════════════════════════
# Task 2.4 — bytes_to_tensor unit tests
# ═══════════════════════════════════════════════════════════════════════════════

from meshrun.worker.protocol import bytes_to_tensor, DType


class TestBytesToTensorFP16:
    """**Validates: Requirements 1.4**
    
    fp16 bytes are deserialized to float values.
    """

    def test_fp16_single_element(self):
        """Single fp16 element deserialization."""
        # 1.5 in fp16 little-endian: 0x3E00
        data = b'\x00\x3E'
        result = bytes_to_tensor(data, DType.FP16, dims=(1, 0, 0, 0), num_dims=1)
        # 1.5 in fp16 deserialized to float
        assert len(result) == 1
        assert abs(result[0] - 1.5) < 1e-3  # fp16 precision

    def test_fp16_multiple_elements(self):
        """Multiple fp16 elements deserialized."""
        # 1.0 = 0x3C00, 2.0 = 0x4000, 3.0 = 0x4200, 4.0 = 0x4400
        data = b'\x00\x3C\x00\x40\x00\x42\x00\x44'
        result = bytes_to_tensor(data, DType.FP16, dims=(4, 0, 0, 0), num_dims=1)
        assert len(result) == 4
        assert abs(result[0] - 1.0) < 1e-3
        assert abs(result[1] - 2.0) < 1e-3
        assert abs(result[2] - 3.0) < 1e-3
        assert abs(result[3] - 4.0) < 1e-3

    def test_fp16_negative_values(self):
        """Negative fp16 values."""
        # -1.5 in fp16: 0xBE00
        data = b'\x00\xBE'
        result = bytes_to_tensor(data, DType.FP16, dims=(1, 0, 0, 0), num_dims=1)
        assert len(result) == 1
        assert abs(result[0] - (-1.5)) < 1e-3

    def test_fp16_zero(self):
        """Zero value."""
        data = b'\x00\x00'
        result = bytes_to_tensor(data, DType.FP16, dims=(1, 0, 0, 0), num_dims=1)
        assert len(result) == 1
        assert abs(result[0]) < 1e-5

    def test_fp16_empty_tensor(self):
        """Empty tensor - zero dimension case."""
        data = b''
        # Note: In protocol context, dims[i] must be > 0 for active dimensions
        # But bytes_to_tensor can handle empty tensors
        result = bytes_to_tensor(data, DType.FP16, dims=(0, 0, 0, 0), num_dims=1)
        # With dims[0]=0, product=0, so empty list expected
        assert result == []

    def test_fp16_2d_tensor(self):
        """2D tensor deserialization."""
        # 6 elements: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] in 2x3 shape
        # Serialized values: 1.0=0x3C00, 2.0=0x4000, 3.0=0x4200, 4.0=0x4400, 5.0=0x4500, 6.0=0x4600
        data = b'\x00\x3C\x00\x40\x00\x42\x00\x44\x00\x45\x00\x46'
        result = bytes_to_tensor(data, DType.FP16, dims=(2, 3, 0, 0), num_dims=2)
        assert len(result) == 6  # 2*3 = 6 elements
        # Check first and last values
        assert abs(result[0] - 1.0) < 1e-3
        assert abs(result[5] - 6.0) < 1e-3


class TestBytesToTensorINT8:
    """**Validates: Requirements 1.4**
    
    int8 bytes are deserialized to integer values.
    """

    def test_int8_single_element(self):
        """Single int8 element deserialization."""
        data = b'\x2A'  # 42 in hex
        result = bytes_to_tensor(data, DType.INT8, dims=(1, 0, 0, 0), num_dims=1)
        assert result == [42]

    def test_int8_multiple_elements(self):
        """Multiple int8 elements deserialized."""
        data = b'\x01\xFE\x03\xFC'  # 1, -2, 3, -4
        result = bytes_to_tensor(data, DType.INT8, dims=(4, 0, 0, 0), num_dims=1)
        assert result == [1, -2, 3, -4]

    def test_int8_boundary_values(self):
        """int8 boundary values (-128 to 127)."""
        data = b'\x80\x00\x7F'  # -128, 0, 127
        result = bytes_to_tensor(data, DType.INT8, dims=(3, 0, 0, 0), num_dims=1)
        assert result == [-128, 0, 127]

    def test_int8_empty_tensor(self):
        """Empty tensor - zero dimension case."""
        data = b''
        # Note: In protocol context, dims[i] must be > 0 for active dimensions
        # But bytes_to_tensor can handle empty tensors
        result = bytes_to_tensor(data, DType.INT8, dims=(0, 0, 0, 0), num_dims=1)
        assert result == []

    def test_int8_3d_tensor(self):
        """3D tensor deserialization."""
        # 8 elements: values 1..8 in 2x2x2 shape
        data = b'\x01\x02\x03\x04\x05\x06\x07\x08'
        result = bytes_to_tensor(data, DType.INT8, dims=(2, 2, 2, 0), num_dims=3)
        assert len(result) == 8  # 2*2*2 = 8 elements
        assert result == [1, 2, 3, 4, 5, 6, 7, 8]


class TestBytesToTensorValidation:
    """Validation and error handling for bytes_to_tensor."""

    def test_invalid_dtype(self):
        """Reject invalid dtype values."""
        with pytest.raises(ValueError, match="Invalid dtype"):
            bytes_to_tensor(b'\x00\x00', 0, dims=(1, 0, 0, 0), num_dims=1)
        
        with pytest.raises(ValueError, match="Invalid dtype"):
            bytes_to_tensor(b'\x00', 99, dims=(1, 0, 0, 0), num_dims=1)

    def test_invalid_num_dims(self):
        """Reject invalid num_dims values."""
        with pytest.raises(ValueError, match="num_dims"):
            bytes_to_tensor(b'\x00\x00', DType.FP16, dims=(1, 0, 0, 0), num_dims=0)
        
        with pytest.raises(ValueError, match="num_dims"):
            bytes_to_tensor(b'\x00\x00', DType.FP16, dims=(1, 0, 0, 0), num_dims=5)

    def test_data_length_mismatch(self):
        """Reject data with wrong length."""
        # Expected: 1 element * 2 bytes = 2 bytes
        # Actual: 1 byte
        with pytest.raises(ValueError, match="Data length"):
            bytes_to_tensor(b'\x00', DType.FP16, dims=(1, 0, 0, 0), num_dims=1)
        
        # Expected: 1 element * 1 byte = 1 byte
        # Actual: 2 bytes
        with pytest.raises(ValueError, match="Data length"):
            bytes_to_tensor(b'\x00\x00', DType.INT8, dims=(1, 0, 0, 0), num_dims=1)

    def test_zero_active_dimension(self):
        """Handle zero active dimension (empty tensor)."""
        # bytes_to_tensor now allows zero dimensions (empty tensors)
        result = bytes_to_tensor(b'', DType.FP16, dims=(0, 0, 0, 0), num_dims=1)
        assert result == []

    def test_nonzero_unused_dimension(self):
        """Reject nonzero unused dimension."""
        with pytest.raises(ValueError, match=r"dims\[1\].*0"):
            bytes_to_tensor(b'\x00\x00', DType.FP16, dims=(1, 5, 0, 0), num_dims=1)


class TestBytesToTensorRoundTrip:
    """Round-trip tests: tensor_to_bytes → bytes_to_tensor."""

    def test_fp16_roundtrip_simple(self):
        """Round-trip for fp16 values."""
        original = [1.5, -2.0, 3.14, 0.0]
        serialized = tensor_to_bytes(original, DType.FP16)
        deserialized = bytes_to_tensor(
            serialized, DType.FP16, dims=(len(original), 0, 0, 0), num_dims=1
        )
        # Compare with fp16 precision tolerance
        for orig, deser in zip(original, deserialized):
            if abs(orig) > 1e-5:
                assert abs(orig - deser) / abs(orig) < 1e-3
            else:
                assert abs(orig - deser) < 1e-5

    def test_int8_roundtrip_simple(self):
        """Round-trip for int8 values."""
        original = [42, -17, 0, 127, -128]
        serialized = tensor_to_bytes(original, DType.INT8)
        deserialized = bytes_to_tensor(
            serialized, DType.INT8, dims=(len(original), 0, 0, 0), num_dims=1
        )
        # Should be exactly equal for int8
        assert deserialized == original

    def test_fp16_roundtrip_2d(self):
        """Round-trip for 2D fp16 tensor."""
        # Create 2x3 tensor (6 elements)
        original = [float(i) for i in range(6)]  # [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        serialized = tensor_to_bytes(original, DType.FP16)
        deserialized = bytes_to_tensor(
            serialized, DType.FP16, dims=(2, 3, 0, 0), num_dims=2
        )
        # Compare with tolerance
        for orig, deser in zip(original, deserialized):
            if abs(orig) > 1e-5:
                assert abs(orig - deser) / abs(orig) < 1e-3
            else:
                assert abs(orig - deser) < 1e-5

    def test_int8_roundtrip_3d(self):
        """Round-trip for 3D int8 tensor."""
        # Create 2x2x2 tensor (8 elements)
        original = [i - 4 for i in range(8)]  # [-4, -3, -2, -1, 0, 1, 2, 3]
        serialized = tensor_to_bytes(original, DType.INT8)
        deserialized = bytes_to_tensor(
            serialized, DType.INT8, dims=(2, 2, 2, 0), num_dims=3
        )
        # Should be exactly equal
        assert deserialized == original


class TestBytesToTensorPropertyBased:
    """Property-based tests for bytes_to_tensor using Hypothesis."""
    
    from hypothesis import given, settings, assume
    from hypothesis import strategies as st
    
    @given(
        elements=st.lists(st.floats(allow_infinity=False, allow_nan=False, min_value=-65504.0, max_value=65504.0), min_size=1, max_size=50),
        dtype=st.sampled_from([DType.FP16])
    )
    @settings(max_examples=50)
    def test_fp16_roundtrip_property(self, elements, dtype):
        """**Validates: Requirements 1.4**
        
        Property: For any list of floats within fp16 range, serializing with
        tensor_to_bytes and then deserializing with bytes_to_tensor should
        reproduce the original values (within fp16 precision tolerance).
        """
        # Serialize
        serialized = tensor_to_bytes(elements, dtype)
        
        # Deserialize
        deserialized = bytes_to_tensor(
            serialized, dtype, dims=(len(elements), 0, 0, 0), num_dims=1
        )
        
        # Compare with fp16 precision tolerance
        for orig, deser in zip(elements, deserialized):
            if abs(orig) > 1e-5:
                assert abs(orig - deser) / abs(orig) < 1e-3
            else:
                assert abs(orig - deser) < 1e-5
    
    @given(
        elements=st.lists(st.integers(min_value=-128, max_value=127), min_size=1, max_size=50),
        dtype=st.sampled_from([DType.INT8])
    )
    @settings(max_examples=50)
    def test_int8_roundtrip_property(self, elements, dtype):
        """**Validates: Requirements 1.4**
        
        Property: For any list of int8-compatible integers, serializing with
        tensor_to_bytes and then deserializing with bytes_to_tensor should
        reproduce the exact original values (bitwise identical).
        """
        # Serialize
        serialized = tensor_to_bytes(elements, dtype)
        
        # Deserialize
        deserialized = bytes_to_tensor(
            serialized, dtype, dims=(len(elements), 0, 0, 0), num_dims=1
        )
        
        # Should be exactly equal for int8
        assert deserialized == elements
    
    @given(
        data=st.binary(min_size=0, max_size=100),
        dtype=st.sampled_from([DType.FP16, DType.INT8]),
        dims=st.tuples(
            st.integers(min_value=0, max_value=10),
            st.integers(min_value=0, max_value=10),
            st.integers(min_value=0, max_value=10),
            st.integers(min_value=0, max_value=10)
        ),
        num_dims=st.integers(min_value=1, max_value=4)
    )
    @settings(max_examples=50)
    def test_validation_property(self, data, dtype, dims, num_dims):
        """Property: bytes_to_tensor validates data length matches expected size."""
        # Calculate expected size
        product = 1
        for i in range(num_dims):
            product *= dims[i]
        expected_size = product * DTYPE_SIZE[dtype]
        
        # Check unused dimensions are zero
        has_nonzero_unused = any(dims[i] != 0 for i in range(num_dims, 4))
        
        try:
            result = bytes_to_tensor(data, dtype, dims, num_dims)
            # If we get here, validation passed
            # Check data length matches expected size
            assert len(data) == expected_size
            # Check unused dimensions are zero (function should have validated)
            assert not has_nonzero_unused
            # Check result length matches product
            assert len(result) == product
        except ValueError as e:
            # Should fail for one of these reasons:
            # 1. Data length mismatch
            # 2. Nonzero unused dimension
            # 3. Negative dimension
            error_msg = str(e)
            assert (
                "Data length" in error_msg or
                "must be 0 for unused dimension" in error_msg or
                "must be >= 0" in error_msg
            )
# ═══════════════════════════════════════════════════════════════════════════════
# Task 2.5 — read_message unit tests
# ═══════════════════════════════════════════════════════════════════════════════

from meshrun.worker.protocol import read_message, Header, DType, MessageType
from unittest.mock import MagicMock, patch


class TestReadMessage:
    """**Validates: Requirements 1.3, 1.4**
    
    Tests for the full message read flow:
    read_exact(32) → validate → read_exact(payload_size) → reconstruct tensor
    """

    def test_read_message_fp16_1d(self):
        """Read a simple 1D fp16 message."""
        # Create a valid header
        header = Header(
            message_type=MessageType.FORWARD,
            request_id=123,
            step_id=0,
            payload_size=20,  # 10 fp16 elements = 20 bytes
            dtype=DType.FP16,
            num_dims=1,
            dims=(10, 0, 0, 0),
        )
        
        # Create tensor data: 10 fp16 values
        tensor_data = [float(i) for i in range(10)]  # [0.0, 1.0, ..., 9.0]
        
        # Serialize header and tensor
        header_bytes = header.pack()
        tensor_bytes = tensor_to_bytes(tensor_data, DType.FP16)
        
        # Mock socket that returns header then tensor bytes
        sock = MagicMock()
        sock.recv.side_effect = [header_bytes, tensor_bytes]
        
        # Read the message
        read_header, read_tensor = read_message(sock)
        
        # Verify header matches
        assert read_header == header
        
        # Verify tensor data matches (within fp16 precision)
        assert len(read_tensor) == len(tensor_data)
        for orig, read in zip(tensor_data, read_tensor):
            if abs(orig) > 1e-5:
                assert abs(orig - read) / abs(orig) < 1e-3
            else:
                assert abs(orig - read) < 1e-5

    def test_read_message_int8_2d(self):
        """Read a 2D int8 message."""
        # Create a valid header
        header = Header(
            message_type=MessageType.RESULT,
            request_id=456,
            step_id=1,
            payload_size=32,  # 4*8 int8 elements = 32 bytes
            dtype=DType.INT8,
            num_dims=2,
            dims=(4, 8, 0, 0),
        )
        
        # Create tensor data: 32 int8 values
        tensor_data = [i % 128 for i in range(32)]  # Values 0-127
        
        # Serialize header and tensor
        header_bytes = header.pack()
        tensor_bytes = tensor_to_bytes(tensor_data, DType.INT8)
        
        # Mock socket that returns header then tensor bytes
        sock = MagicMock()
        sock.recv.side_effect = [header_bytes, tensor_bytes]
        
        # Read the message
        read_header, read_tensor = read_message(sock)
        
        # Verify header matches
        assert read_header == header
        
        # Verify tensor data matches exactly (int8)
        assert read_tensor == tensor_data

    def test_read_message_with_partial_reads(self):
        """Test read_message with partial reads (multiple recv calls)."""
        header = Header(
            message_type=MessageType.FORWARD,
            request_id=789,
            step_id=2,
            payload_size=8,  # 4 fp16 elements = 8 bytes
            dtype=DType.FP16,
            num_dims=1,
            dims=(4, 0, 0, 0),
        )
        
        tensor_data = [1.0, 2.0, 3.0, 4.0]
        
        header_bytes = header.pack()
        tensor_bytes = tensor_to_bytes(tensor_data, DType.FP16)
        
        # Split both header and tensor into multiple chunks
        sock = MagicMock()
        sock.recv.side_effect = [
            header_bytes[:16],  # First half of header
            header_bytes[16:],  # Second half of header
            tensor_bytes[:3],   # First 3 bytes of tensor
            tensor_bytes[3:6],  # Next 3 bytes
            tensor_bytes[6:],    # Last 2 bytes
        ]
        
        # Read the message
        read_header, read_tensor = read_message(sock)
        
        # Verify header matches
        assert read_header == header
        
        # Verify tensor data matches (within fp16 precision)
        assert len(read_tensor) == len(tensor_data)
        for orig, read in zip(tensor_data, read_tensor):
            if abs(orig) > 1e-5:
                assert abs(orig - read) / abs(orig) < 1e-3
            else:
                assert abs(orig - read) < 1e-5

    def test_read_message_rejects_invalid_header(self):
        """read_message should reject invalid headers (via Header.unpack)."""
        # Create an invalid header (bad message_type)
        bad_header = Header(
            message_type=0,  # Invalid
            request_id=1,
            step_id=0,
            payload_size=20,
            dtype=DType.FP16,
            num_dims=1,
            dims=(10, 0, 0, 0),
        )
        
        bad_header_bytes = bad_header.pack()
        
        sock = MagicMock()
        sock.recv.return_value = bad_header_bytes
        
        # Should raise ValueError from Header.unpack()
        with pytest.raises(ValueError, match="message_type"):
            read_message(sock)

    def test_read_message_eof_during_header(self):
        """Connection closed while reading header."""
        sock = MagicMock()
        sock.recv.return_value = b""  # EOF immediately
        
        with pytest.raises(ConnectionError, match="Connection closed"):
            read_message(sock)

    def test_read_message_eof_during_payload(self):
        """Connection closed while reading payload."""
        header = _valid_header()
        header_bytes = header.pack()
        
        sock = MagicMock()
        sock.recv.side_effect = [
            header_bytes,  # Header read successfully
            b"",           # EOF during payload read
        ]
        
        with pytest.raises(ConnectionError, match="Connection closed"):
            read_message(sock)

    def test_read_message_timeout_during_header(self):
        """Timeout while reading header."""
        sock = MagicMock()
        sock.recv.side_effect = socket.timeout("timed out")
        
        with pytest.raises(socket.timeout):
            read_message(sock)

    def test_read_message_timeout_during_payload(self):
        """Timeout while reading payload."""
        header = _valid_header()
        header_bytes = header.pack()
        
        sock = MagicMock()
        sock.recv.side_effect = [
            header_bytes,  # Header read successfully
            socket.timeout("timed out"),  # Timeout during payload
        ]
        
        with pytest.raises(socket.timeout):
            read_message(sock)

    def test_read_message_4d_tensor(self):
        """Read a 4D tensor message."""
        header = Header(
            message_type=MessageType.HEARTBEAT_DATA,
            request_id=999,
            step_id=3,
            payload_size=48,  # 1*2*3*4 fp16 elements = 48 bytes
            dtype=DType.FP16,
            num_dims=4,
            dims=(1, 2, 3, 4),
        )
        
        # Create tensor data: 24 fp16 values (1*2*3*4 = 24)
        tensor_data = [float(i) for i in range(24)]
        
        header_bytes = header.pack()
        tensor_bytes = tensor_to_bytes(tensor_data, DType.FP16)
        
        sock = MagicMock()
        sock.recv.side_effect = [header_bytes, tensor_bytes]
        
        read_header, read_tensor = read_message(sock)
        
        assert read_header == header
        assert len(read_tensor) == len(tensor_data)
        for orig, read in zip(tensor_data, read_tensor):
            if abs(orig) > 1e-5:
                assert abs(orig - read) / abs(orig) < 1e-3
            else:
                assert abs(orig - read) < 1e-5


class TestReadMessagePropertyBased:
    """Property-based tests for read_message using Hypothesis."""
    
    from hypothesis import given, settings, assume
    from hypothesis import strategies as st
    
    @given(
        header=valid_headers(),
        elements=st.lists(st.floats(allow_infinity=False, allow_nan=False, min_value=-65504.0, max_value=65504.0), min_size=1, max_size=100)
    )
    @settings(max_examples=50)
    def test_fp16_roundtrip_property(self, header, elements):
        """**Validates: Requirements 1.3, 1.4**
        
        Property: For any valid fp16 header and tensor data, we can:
        1. Serialize header and tensor
        2. Simulate reading from a socket
        3. Use read_message to reconstruct both
        4. Verify header matches and tensor data is within fp16 precision
        """
        # Skip if header dtype is not FP16
        assume(header.dtype == DType.FP16)
        
        # Adjust elements count to match header dimensions
        product = 1
        for i in range(header.num_dims):
            product *= header.dims[i]
        
        # Create tensor data with correct number of elements
        tensor_data = elements[:product]
        # Pad if needed (unlikely with hypothesis generation)
        if len(tensor_data) < product:
            tensor_data.extend([0.0] * (product - len(tensor_data)))
        else:
            tensor_data = tensor_data[:product]
        
        # Update header payload_size to match actual tensor data
        header = Header(
            message_type=header.message_type,
            request_id=header.request_id,
            step_id=header.step_id,
            payload_size=product * DTYPE_SIZE[DType.FP16],
            dtype=header.dtype,
            num_dims=header.num_dims,
            dims=header.dims,
        )
        
        # Serialize
        header_bytes = header.pack()
        tensor_bytes = tensor_to_bytes(tensor_data, DType.FP16)
        
        # Mock socket
        sock = MagicMock()
        sock.recv.side_effect = [header_bytes, tensor_bytes]
        
        # Read message
        read_header, read_tensor = read_message(sock)
        
        # Verify header matches
        assert read_header == header
        
        # Verify tensor data matches (within fp16 precision)
        assert len(read_tensor) == len(tensor_data)
        for orig, read in zip(tensor_data, read_tensor):
            if abs(orig) > 1e-5:
                assert abs(orig - read) / abs(orig) < 1e-3
            else:
                assert abs(orig - read) < 1e-5
    
    @given(
        header=valid_headers(),
        elements=st.lists(st.integers(min_value=-128, max_value=127), min_size=1, max_size=100)
    )
    @settings(max_examples=50)
    def test_int8_roundtrip_property(self, header, elements):
        """**Validates: Requirements 1.3, 1.4**
        
        Property: For any valid int8 header and tensor data, we can:
        1. Serialize header and tensor
        2. Simulate reading from a socket
        3. Use read_message to reconstruct both
        4. Verify header matches and tensor data is exactly equal
        """
        # Skip if header dtype is not INT8
        assume(header.dtype == DType.INT8)
        
        # Adjust elements count to match header dimensions
        product = 1
        for i in range(header.num_dims):
            product *= header.dims[i]
        
        # Create tensor data with correct number of elements
        tensor_data = elements[:product]
        # Pad if needed
        if len(tensor_data) < product:
            tensor_data.extend([0] * (product - len(tensor_data)))
        else:
            tensor_data = tensor_data[:product]
        
        # Update header payload_size to match actual tensor data
        header = Header(
            message_type=header.message_type,
            request_id=header.request_id,
            step_id=header.step_id,
            payload_size=product * DTYPE_SIZE[DType.INT8],
            dtype=header.dtype,
            num_dims=header.num_dims,
            dims=header.dims,
        )
        
        # Serialize
        header_bytes = header.pack()
        tensor_bytes = tensor_to_bytes(tensor_data, DType.INT8)
        
        # Mock socket
        sock = MagicMock()
        sock.recv.side_effect = [header_bytes, tensor_bytes]
        
        # Read message
        read_header, read_tensor = read_message(sock)
        
        # Verify header matches
        assert read_header == header
        
        # Verify tensor data matches exactly (int8)
        assert read_tensor == tensor_data
# ═══════════════════════════════════════════════════════════════════════════════
# Task 2.6 — write_message unit tests
# ═══════════════════════════════════════════════════════════════════════════════

from meshrun.worker.protocol import write_message, Header, DType, MessageType, tensor_to_bytes
from unittest.mock import MagicMock, patch
import socket


class TestWriteMessage:
    """**Validates: Requirements 1.4, 1.5**
    
    Tests for the full message write flow:
    1. Validate tensor_data length matches header.dims
    2. Serialize tensor_data to bytes using tensor_to_bytes() with dtype from header
    3. Ensure header.payload_size matches serialized tensor byte count
    4. Pack header using header.pack()
    5. Write header + payload as single contiguous write using write_all()
    """

    def test_write_message_fp16_1d(self):
        """Write a simple 1D fp16 message."""
        # Create tensor data: 10 fp16 values
        tensor_data = [float(i) for i in range(10)]  # [0.0, 1.0, ..., 9.0]
        
        # Create a valid header matching the tensor data
        header = Header(
            message_type=MessageType.FORWARD,
            request_id=123,
            step_id=0,
            payload_size=20,  # 10 fp16 elements = 20 bytes
            dtype=DType.FP16,
            num_dims=1,
            dims=(10, 0, 0, 0),
        )
        
        # Mock socket - send returns all data at once
        sock = MagicMock()
        sock.send.return_value = 100  # Large enough to accept all data
        
        # Call write_message
        write_message(sock, header, tensor_data)
        
        # Verify write_all was called once with the concatenated header + payload
        assert sock.send.call_count == 1
        
        # Get the actual data that was written
        call_args = sock.send.call_args[0][0]
        all_data = call_args
        
        # Verify total length = header (32 bytes) + payload (20 bytes) = 52 bytes
        assert len(all_data) == 32 + 20
        
        # Verify first 32 bytes are the packed header
        header_bytes = header.pack()
        assert all_data[:32] == header_bytes
        
        # Verify last 20 bytes are the serialized tensor
        tensor_bytes = tensor_to_bytes(tensor_data, DType.FP16)
        assert all_data[32:] == tensor_bytes

    def test_write_message_int8_2d(self):
        """Write a 2D int8 message."""
        # Create tensor data: 32 int8 values
        tensor_data = [i % 128 for i in range(32)]  # Values 0-127
        
        # Create a valid header matching the tensor data
        header = Header(
            message_type=MessageType.RESULT,
            request_id=456,
            step_id=1,
            payload_size=32,  # 4*8 int8 elements = 32 bytes
            dtype=DType.INT8,
            num_dims=2,
            dims=(4, 8, 0, 0),
        )
        
        # Mock socket - send returns all data at once
        sock = MagicMock()
        sock.send.return_value = 100  # Large enough to accept all data
        
        # Call write_message
        write_message(sock, header, tensor_data)
        
        # Get all data written
        call_args = sock.send.call_args[0][0]
        all_data = call_args
        
        # Verify total length = header (32 bytes) + payload (32 bytes) = 64 bytes
        assert len(all_data) == 32 + 32
        
        # Verify header
        header_bytes = header.pack()
        assert all_data[:32] == header_bytes
        
        # Verify tensor
        tensor_bytes = tensor_to_bytes(tensor_data, DType.INT8)
        assert all_data[32:] == tensor_bytes

    def test_write_message_rejects_tensor_length_mismatch(self):
        """write_message should reject tensor data with wrong length."""
        header = Header(
            message_type=MessageType.FORWARD,
            request_id=1,
            step_id=0,
            payload_size=20,  # Expects 10 fp16 elements
            dtype=DType.FP16,
            num_dims=1,
            dims=(10, 0, 0, 0),
        )
        
        # Tensor data with wrong length (9 elements instead of 10)
        tensor_data = [float(i) for i in range(9)]
        
        sock = MagicMock()
        
        with pytest.raises(ValueError, match="Tensor data length"):
            write_message(sock, header, tensor_data)

    def test_write_message_rejects_payload_size_mismatch(self):
        """write_message should reject header with wrong payload_size."""
        # Create tensor data: 10 fp16 values
        tensor_data = [float(i) for i in range(10)]
        
        # Header with wrong payload_size (should be 20 bytes for 10 fp16 elements)
        header = Header(
            message_type=MessageType.FORWARD,
            request_id=1,
            step_id=0,
            payload_size=999,  # Wrong!
            dtype=DType.FP16,
            num_dims=1,
            dims=(10, 0, 0, 0),
        )
        
        sock = MagicMock()
        
        with pytest.raises(ValueError, match="payload_size"):
            write_message(sock, header, tensor_data)

    def test_write_message_with_partial_writes(self):
        """Test write_message with partial writes (multiple send calls)."""
        tensor_data = [1.0, 2.0, 3.0, 4.0]
        
        header = Header(
            message_type=MessageType.FORWARD,
            request_id=789,
            step_id=2,
            payload_size=8,  # 4 fp16 elements = 8 bytes
            dtype=DType.FP16,
            num_dims=1,
            dims=(4, 0, 0, 0),
        )
        
        # Mock socket that accepts partial writes
        sock = MagicMock()
        sock.send.side_effect = [5, 3, 32]  # Partial writes: 5 bytes, 3 bytes, then 32 bytes
        
        # Call write_message
        write_message(sock, header, tensor_data)
        
        # Should have called send multiple times due to partial writes
        assert sock.send.call_count == 3

    def test_write_message_connection_broken(self):
        """Connection broken during write."""
        tensor_data = [float(i) for i in range(5)]
        
        header = Header(
            message_type=MessageType.FORWARD,
            request_id=1,
            step_id=0,
            payload_size=10,  # 5 fp16 elements = 10 bytes
            dtype=DType.FP16,
            num_dims=1,
            dims=(5, 0, 0, 0),
        )
        
        sock = MagicMock()
        sock.send.return_value = 0  # Connection broken
        
        with pytest.raises(ConnectionError, match="Connection closed"):
            write_message(sock, header, tensor_data)

    def test_write_message_timeout(self):
        """Timeout during write."""
        tensor_data = [float(i) for i in range(5)]
        
        header = Header(
            message_type=MessageType.FORWARD,
            request_id=1,
            step_id=0,
            payload_size=10,
            dtype=DType.FP16,
            num_dims=1,
            dims=(5, 0, 0, 0),
        )
        
        sock = MagicMock()
        sock.send.side_effect = socket.timeout("timed out")
        
        with pytest.raises(socket.timeout):
            write_message(sock, header, tensor_data)

    def test_write_message_4d_tensor(self):
        """Write a 4D tensor message."""
        # Create tensor data: 24 fp16 values (1*2*3*4 = 24)
        tensor_data = [float(i) for i in range(24)]
        
        header = Header(
            message_type=MessageType.HEARTBEAT_DATA,
            request_id=999,
            step_id=3,
            payload_size=48,  # 1*2*3*4 fp16 elements = 48 bytes
            dtype=DType.FP16,
            num_dims=4,
            dims=(1, 2, 3, 4),
        )
        
        sock = MagicMock()
        sock.send.return_value = 100  # Large enough to accept all data
        
        # Call write_message
        write_message(sock, header, tensor_data)
        
        # Get all data written
        call_args = sock.send.call_args[0][0]
        all_data = call_args
        
        # Verify total length = header (32 bytes) + payload (48 bytes) = 80 bytes
        assert len(all_data) == 32 + 48
        
        # Verify header
        header_bytes = header.pack()
        assert all_data[:32] == header_bytes
        
        # Verify tensor
        tensor_bytes = tensor_to_bytes(tensor_data, DType.FP16)
        assert all_data[32:] == tensor_bytes


class TestWriteMessageRoundTrip:
    """Round-trip tests: write_message → read_message."""
    
    def test_roundtrip_fp16_1d(self):
        """Round-trip: write then read should reconstruct same data."""
        # Create tensor data
        tensor_data = [float(i) for i in range(10)]
        
        # Create header
        header = Header(
            message_type=MessageType.FORWARD,
            request_id=123,
            step_id=0,
            payload_size=20,
            dtype=DType.FP16,
            num_dims=1,
            dims=(10, 0, 0, 0),
        )
        
        # Create a mock socket that captures written data
        written_data = bytearray()
        
        def mock_send(data):
            written_data.extend(data)
            return len(data)
        
        sock_write = MagicMock()
        sock_write.send.side_effect = mock_send
        
        # Write the message
        write_message(sock_write, header, tensor_data)
        
        # Now create a mock socket for reading that returns the written data
        sock_read = MagicMock()
        
        # Split data into header and payload for read_message
        header_bytes = bytes(written_data[:32])
        payload_bytes = bytes(written_data[32:])
        
        sock_read.recv.side_effect = [header_bytes, payload_bytes]
        
        # Read the message back
        from meshrun.worker.protocol import read_message
        read_header, read_tensor = read_message(sock_read)
        
        # Verify header matches
        assert read_header == header
        
        # Verify tensor data matches (within fp16 precision)
        assert len(read_tensor) == len(tensor_data)
        for orig, read in zip(tensor_data, read_tensor):
            if abs(orig) > 1e-5:
                assert abs(orig - read) / abs(orig) < 1e-3
            else:
                assert abs(orig - read) < 1e-5

    def test_roundtrip_int8_2d(self):
        """Round-trip: write then read should reconstruct same int8 data."""
        # Create tensor data
        tensor_data = [i % 128 for i in range(32)]
        
        # Create header
        header = Header(
            message_type=MessageType.RESULT,
            request_id=456,
            step_id=1,
            payload_size=32,
            dtype=DType.INT8,
            num_dims=2,
            dims=(4, 8, 0, 0),
        )
        
        # Create a mock socket that captures written data
        written_data = bytearray()
        
        def mock_send(data):
            written_data.extend(data)
            return len(data)
        
        sock_write = MagicMock()
        sock_write.send.side_effect = mock_send
        
        # Write the message
        write_message(sock_write, header, tensor_data)
        
        # Now create a mock socket for reading
        sock_read = MagicMock()
        
        header_bytes = bytes(written_data[:32])
        payload_bytes = bytes(written_data[32:])
        
        sock_read.recv.side_effect = [header_bytes, payload_bytes]
        
        # Read the message back
        from meshrun.worker.protocol import read_message
        read_header, read_tensor = read_message(sock_read)
        
        # Verify header matches
        assert read_header == header
        
        # Verify tensor data matches exactly (int8)
        assert read_tensor == tensor_data


class TestWriteMessagePropertyBased:
    """Property-based tests for write_message using Hypothesis."""
    
    from hypothesis import given, settings, assume
    from hypothesis import strategies as st
    
    @given(
        header=valid_headers(),
        elements=st.lists(st.floats(allow_infinity=False, allow_nan=False, min_value=-65504.0, max_value=65504.0), min_size=1, max_size=100)
    )
    @settings(max_examples=50)
    def test_fp16_roundtrip_property(self, header, elements):
        """**Validates: Requirements 1.4, 1.5**
        
        Property: For any valid fp16 header and tensor data, we can:
        1. Adjust header to match tensor data dimensions
        2. Write message to mock socket
        3. Read message back from captured data
        4. Verify round-trip reconstruction
        """
        # Skip if header dtype is not FP16
        assume(header.dtype == DType.FP16)
        
        # Adjust elements count to match header dimensions
        product = 1
        for i in range(header.num_dims):
            product *= header.dims[i]
        
        # Create tensor data with correct number of elements
        tensor_data = elements[:product]
        # Pad if needed
        if len(tensor_data) < product:
            tensor_data.extend([0.0] * (product - len(tensor_data)))
        else:
            tensor_data = tensor_data[:product]
        
        # Update header payload_size to match actual tensor data
        header = Header(
            message_type=header.message_type,
            request_id=header.request_id,
            step_id=header.step_id,
            payload_size=product * DTYPE_SIZE[DType.FP16],
            dtype=header.dtype,
            num_dims=header.num_dims,
            dims=header.dims,
        )
        
        # Create a mock socket that captures written data
        written_data = bytearray()
        
        def mock_send(data):
            written_data.extend(data)
            return len(data)
        
        sock_write = MagicMock()
        sock_write.send.side_effect = mock_send
        
        # Write the message
        write_message(sock_write, header, tensor_data)
        
        # Now create a mock socket for reading
        sock_read = MagicMock()
        
        # Split data into header and payload
        header_bytes = bytes(written_data[:32])
        payload_bytes = bytes(written_data[32:])
        
        sock_read.recv.side_effect = [header_bytes, payload_bytes]
        
        # Read the message back
        from meshrun.worker.protocol import read_message
        read_header, read_tensor = read_message(sock_read)
        
        # Verify header matches
        assert read_header == header
        
        # Verify tensor data matches (within fp16 precision)
        assert len(read_tensor) == len(tensor_data)
        for orig, read in zip(tensor_data, read_tensor):
            if abs(orig) > 1e-5:
                assert abs(orig - read) / abs(orig) < 1e-3
            else:
                assert abs(orig - read) < 1e-5
    
    @given(
        header=valid_headers(),
        elements=st.lists(st.integers(min_value=-128, max_value=127), min_size=1, max_size=100)
    )
    @settings(max_examples=50)
    def test_int8_roundtrip_property(self, header, elements):
        """**Validates: Requirements 1.4, 1.5**
        
        Property: For any valid int8 header and tensor data, we can:
        1. Adjust header to match tensor data dimensions
        2. Write message to mock socket
        3. Read message back from captured data
        4. Verify round-trip reconstruction (exact match for int8)
        """
        # Skip if header dtype is not INT8
        assume(header.dtype == DType.INT8)
        
        # Adjust elements count to match header dimensions
        product = 1
        for i in range(header.num_dims):
            product *= header.dims[i]
        
        # Create tensor data with correct number of elements
        tensor_data = elements[:product]
        # Pad if needed
        if len(tensor_data) < product:
            tensor_data.extend([0] * (product - len(tensor_data)))
        else:
            tensor_data = tensor_data[:product]
        
        # Update header payload_size to match actual tensor data
        header = Header(
            message_type=header.message_type,
            request_id=header.request_id,
            step_id=header.step_id,
            payload_size=product * DTYPE_SIZE[DType.INT8],
            dtype=header.dtype,
            num_dims=header.num_dims,
            dims=header.dims,
        )
        
        # Create a mock socket that captures written data
        written_data = bytearray()
        
        def mock_send(data):
            written_data.extend(data)
            return len(data)
        
        sock_write = MagicMock()
        sock_write.send.side_effect = mock_send
        
        # Write the message
        write_message(sock_write, header, tensor_data)
        
        # Now create a mock socket for reading
        sock_read = MagicMock()
        
        # Split data into header and payload
        header_bytes = bytes(written_data[:32])
        payload_bytes = bytes(written_data[32:])
        
        sock_read.recv.side_effect = [header_bytes, payload_bytes]
        
        # Read the message back
        from meshrun.worker.protocol import read_message
        read_header, read_tensor = read_message(sock_read)
        
        # Verify header matches
        assert read_header == header
        
        # Verify tensor data matches exactly (int8)
        assert read_tensor == tensor_data