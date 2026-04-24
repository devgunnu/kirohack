# Tasks: Secure Inference Pipeline

## Task 1: Set Up Project Structure and Dependencies
- [x] 1.1 Create project directory structure with `crypto.py`, `model.py`, `node.py`, `pipeline.py`, `client.py`
- [x] 1.2 Create `requirements.txt` with `cryptography>=41.0.0` and `hypothesis` (for testing)

## Task 2: Implement Crypto Module (`crypto.py`)
- [x] 2.1 Implement `generate_key()` using `cryptography.fernet.Fernet.generate_key()`
- [x] 2.2 Implement `encrypt(data: bytes, key: bytes) -> bytes` using Fernet
- [x] 2.3 Implement `decrypt(data: bytes, key: bytes) -> bytes` using Fernet

## Task 3: Implement Model Module (`model.py`)
- [x] 3.1 Implement `run_model(prompt: str) -> str` returning `f"[AI RESPONSE]: {prompt} (processed)"`
- [x] 3.2 Add optional `time.sleep()` delay to simulate processing

## Task 4: Implement Node Class (`node.py`)
- [x] 4.1 Create `Node` class with `name` attribute and `__init__(self, name: str)`
- [x] 4.2 Implement `process(self, data, key, is_first=False, is_last=False)` with conditional decrypt/encrypt logic
- [x] 4.3 Add CLI logging for each operation step with `[NodeName]` prefix

## Task 5: Implement Pipeline Module (`pipeline.py`)
- [x] 5.1 Implement `run(encrypted_prompt: bytes, key: bytes) -> bytes`
- [x] 5.2 Instantiate NodeA, NodeB, NodeC and pass data sequentially with correct `is_first`/`is_last` flags

## Task 6: Implement Client CLI (`client.py`)
- [x] 6.1 Implement `main()` function with user input collection
- [x] 6.2 Add session key generation and base64 display
- [x] 6.3 Add prompt encryption, pipeline invocation, and response decryption with full CLI logging
- [x] 6.4 Add empty input validation
- [x] 6.5 Add `if __name__ == "__main__"` entry point

## Task 7: Write Unit Tests
- [x] 7.1 Test `generate_key()` returns valid 44-byte Fernet key
- [x] 7.2 Test `encrypt()`/`decrypt()` round-trip for sample inputs
- [x] 7.3 Test `run_model()` output format matches expected pattern
- [x] 7.4 Test `Node.process()` with `is_first=True` (decrypts input)
- [x] 7.5 Test `Node.process()` with `is_last=True` (encrypts output)
- [x] 7.6 Test `Node.process()` with both flags `False` (plaintext pass-through)
- [x] 7.7 Test `pipeline.run()` returns decryptable Fernet token
- [x] 7.8 Test wrong key raises `InvalidToken`
- [x] 7.9 Test corrupted ciphertext raises `InvalidToken`

## Task 8: Write Property-Based Tests
- [x] 8.1 Property test: encrypt/decrypt round-trip for arbitrary byte strings (hypothesis)
- [x] 8.2 Property test: `run_model()` output format for arbitrary string prompts (hypothesis)
- [x] 8.3 Property test: full pipeline preserves original prompt in decrypted output (hypothesis)

## Task 9: Integration Test
- [x] 9.1 Test full `client.py` flow with mocked `input()`, capturing stdout and verifying all expected log lines appear in correct order
