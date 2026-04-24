"""Tests for the Secure Inference Pipeline security layer."""

import sys
import os
import pytest
from unittest.mock import patch
from cryptography.fernet import InvalidToken, Fernet
from hypothesis import given, settings
import hypothesis.strategies as st

# Ensure imports resolve from the security directory
sys.path.insert(0, os.path.dirname(__file__))

import crypto
import model
from node import Node
import pipeline

# Capture the real run_model before any patching so tests can call it without delay
_real_run_model = model.run_model
_fast_run_model = lambda p: _real_run_model(p, simulate_delay=False)


# ---------------------------------------------------------------------------
# Task 7: Unit Tests
# ---------------------------------------------------------------------------

class TestCryptoModule:
    def test_generate_key_returns_bytes(self):
        key = crypto.generate_key()
        assert isinstance(key, bytes)

    def test_generate_key_length(self):
        """Fernet keys are 32 random bytes, URL-safe base64-encoded → 44 chars."""
        key = crypto.generate_key()
        assert len(key) == 44

    def test_generate_key_is_valid_fernet_key(self):
        key = crypto.generate_key()
        Fernet(key)  # Should not raise

    def test_encrypt_decrypt_roundtrip(self):
        key = crypto.generate_key()
        plaintext = b"hello secure world"
        assert crypto.decrypt(crypto.encrypt(plaintext, key), key) == plaintext

    def test_encrypt_produces_different_output(self):
        key = crypto.generate_key()
        plaintext = b"hello"
        assert crypto.encrypt(plaintext, key) != plaintext

    def test_wrong_key_raises_invalid_token(self):
        k1 = crypto.generate_key()
        k2 = crypto.generate_key()
        encrypted = crypto.encrypt(b"secret", k1)
        with pytest.raises(InvalidToken):
            crypto.decrypt(encrypted, k2)

    def test_corrupted_ciphertext_raises_invalid_token(self):
        key = crypto.generate_key()
        encrypted = crypto.encrypt(b"secret", key)
        corrupted = encrypted[:-4] + b"XXXX"
        with pytest.raises(InvalidToken):
            crypto.decrypt(corrupted, key)


class TestModelModule:
    def test_output_format(self):
        result = _real_run_model("Explain AI", simulate_delay=False)
        assert result == "[AI RESPONSE]: Explain AI (processed)"

    def test_output_starts_with_prefix(self):
        result = _real_run_model("test", simulate_delay=False)
        assert result.startswith("[AI RESPONSE]: ")

    def test_output_ends_with_suffix(self):
        result = _real_run_model("test", simulate_delay=False)
        assert result.endswith(" (processed)")

    def test_output_contains_prompt(self):
        prompt = "my unique prompt"
        result = _real_run_model(prompt, simulate_delay=False)
        assert prompt in result

    def test_deterministic(self):
        prompt = "same prompt"
        assert _real_run_model(prompt, simulate_delay=False) == _real_run_model(prompt, simulate_delay=False)


class TestNodeProcessing:
    def test_first_node_decrypts_input(self, capsys):
        key = crypto.generate_key()
        node = Node("NodeA")
        encrypted = crypto.encrypt(b"hello", key)
        with patch("model.run_model", side_effect=_fast_run_model):
            output = node.process(encrypted, key, is_first=True, is_last=False)
        captured = capsys.readouterr()
        assert "[NodeA] Decrypting..." in captured.out
        assert b"[AI RESPONSE]" in output

    def test_last_node_encrypts_output(self, capsys):
        key = crypto.generate_key()
        node = Node("NodeC")
        with patch("model.run_model", side_effect=_fast_run_model):
            output = node.process(b"hello", key, is_first=False, is_last=True)
        captured = capsys.readouterr()
        assert "[NodeC] Encrypting output..." in captured.out
        decrypted = crypto.decrypt(output, key)
        assert b"[AI RESPONSE]" in decrypted

    def test_middle_node_plaintext_passthrough(self):
        key = crypto.generate_key()
        node = Node("NodeB")
        with patch("model.run_model", side_effect=_fast_run_model):
            output = node.process(b"hello", key, is_first=False, is_last=False)
        assert isinstance(output, bytes)
        assert "[AI RESPONSE]" in output.decode("utf-8")

    def test_inference_always_runs(self, capsys):
        key = crypto.generate_key()
        for is_first, is_last, data in [
            (False, False, b"plain"),
            (False, True, b"plain"),
        ]:
            node = Node("TestNode")
            with patch("model.run_model", side_effect=_fast_run_model):
                node.process(data, key, is_first=is_first, is_last=is_last)
        captured = capsys.readouterr()
        assert captured.out.count("[TestNode] Running inference...") == 2

    def test_node_log_prefix(self, capsys):
        key = crypto.generate_key()
        node = Node("MyNode")
        with patch("model.run_model", side_effect=_fast_run_model):
            node.process(b"data", key, is_first=False, is_last=False)
        captured = capsys.readouterr()
        assert "[MyNode]" in captured.out


class TestPipeline:
    def test_pipeline_returns_bytes(self):
        key = crypto.generate_key()
        encrypted_prompt = crypto.encrypt(b"test prompt", key)
        with patch("model.run_model", side_effect=_fast_run_model):
            result = pipeline.run(encrypted_prompt, key)
        assert isinstance(result, bytes)

    def test_pipeline_returns_decryptable_fernet_token(self):
        key = crypto.generate_key()
        encrypted_prompt = crypto.encrypt(b"test prompt", key)
        with patch("model.run_model", side_effect=_fast_run_model):
            result = pipeline.run(encrypted_prompt, key)
        decrypted = crypto.decrypt(result, key)
        assert isinstance(decrypted, bytes)

    def test_pipeline_preserves_original_prompt(self):
        key = crypto.generate_key()
        original = b"Explain AI to me"
        encrypted_prompt = crypto.encrypt(original, key)
        with patch("model.run_model", side_effect=_fast_run_model):
            result = pipeline.run(encrypted_prompt, key)
        final = crypto.decrypt(result, key)
        assert b"Explain AI to me" in final

    def test_wrong_key_raises_on_pipeline_output(self):
        k1 = crypto.generate_key()
        k2 = crypto.generate_key()
        encrypted_prompt = crypto.encrypt(b"hello", k1)
        with patch("model.run_model", side_effect=_fast_run_model):
            result = pipeline.run(encrypted_prompt, k1)
        with pytest.raises(InvalidToken):
            crypto.decrypt(result, k2)


# ---------------------------------------------------------------------------
# Task 8: Property-Based Tests
# ---------------------------------------------------------------------------

class TestPropertyBased:
    @given(st.binary(min_size=1, max_size=1024))
    @settings(max_examples=50, deadline=None)
    def test_encrypt_decrypt_roundtrip_arbitrary_bytes(self, data):
        key = crypto.generate_key()
        assert crypto.decrypt(crypto.encrypt(data, key), key) == data

    @given(st.text(min_size=1, max_size=200))
    @settings(max_examples=50, deadline=None)
    def test_run_model_output_format_arbitrary_prompts(self, prompt):
        result = _real_run_model(prompt, simulate_delay=False)
        assert result.startswith("[AI RESPONSE]: ")
        assert " (processed)" in result
        assert prompt in result
        assert len(result) > len(prompt)

    @given(st.text(min_size=1, max_size=100,
                   alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs"))))
    @settings(max_examples=30, deadline=None)
    def test_pipeline_preserves_prompt_arbitrary_inputs(self, prompt):
        key = crypto.generate_key()
        prompt_bytes = prompt.encode("utf-8")
        encrypted = crypto.encrypt(prompt_bytes, key)
        with patch("model.run_model", side_effect=_fast_run_model):
            result = pipeline.run(encrypted, key)
        final = crypto.decrypt(result, key).decode("utf-8")
        assert prompt in final


# ---------------------------------------------------------------------------
# Task 9: Integration Test
# ---------------------------------------------------------------------------

class TestClientIntegration:
    def test_full_client_flow_log_order(self, capsys):
        import client

        with patch("builtins.input", return_value="Explain AI"), \
             patch("model.run_model", side_effect=_fast_run_model):
            client.main()

        out = capsys.readouterr().out
        assert "[Client] Prompt received:" in out
        assert "[Client] Generated Session Key:" in out
        assert "[Client] Encrypted Prompt:" in out
        assert "[NodeA] Decrypting..." in out
        assert "[NodeA] Running inference..." in out
        assert "[NodeB] Running inference..." in out
        assert "[NodeC] Running inference..." in out
        assert "[NodeC] Encrypting output..." in out
        assert "[Client] Encrypted Response:" in out
        assert "[Client] Decrypted Final Output:" in out

    def test_full_client_flow_output_contains_prompt(self, capsys):
        import client

        with patch("builtins.input", return_value="hello world"), \
             patch("model.run_model", side_effect=_fast_run_model):
            client.main()

        assert "hello world" in capsys.readouterr().out

    def test_empty_input_rejected(self, capsys):
        import client

        with patch("builtins.input", return_value=""):
            client.main()

        out = capsys.readouterr().out
        assert "Error" in out or "empty" in out.lower()

    def test_log_order_is_correct(self, capsys):
        import client

        with patch("builtins.input", return_value="order test"), \
             patch("model.run_model", side_effect=_fast_run_model):
            client.main()

        lines = capsys.readouterr().out.splitlines()

        def idx(marker):
            for i, line in enumerate(lines):
                if marker in line:
                    return i
            return -1

        assert idx("[Client] Generated Session Key:") < idx("[NodeA] Decrypting...")
        assert idx("[NodeA] Running inference...") < idx("[NodeB] Running inference...")
        assert idx("[NodeB] Running inference...") < idx("[NodeC] Running inference...")
        assert idx("[NodeC] Encrypting output...") < idx("[Client] Encrypted Response:")
        assert idx("[Client] Encrypted Response:") < idx("[Client] Decrypted Final Output:")
