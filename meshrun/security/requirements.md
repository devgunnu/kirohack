# Requirements: Secure Inference Pipeline

## Requirement 1: Crypto Module

### 1.1 Key Generation
GIVEN the crypto module is invoked
WHEN `generate_key()` is called
THEN it returns a valid Fernet-compatible key as `bytes` of length 44 (URL-safe base64)

### 1.2 Encryption Round-Trip
GIVEN a valid Fernet key and arbitrary plaintext bytes
WHEN the data is encrypted then decrypted with the same key
THEN the decrypted output equals the original plaintext exactly

### 1.3 Encryption Produces Different Output
GIVEN a valid Fernet key and non-empty plaintext bytes
WHEN the data is encrypted
THEN the encrypted output is different from the original plaintext

### 1.4 Wrong Key Rejection
GIVEN data encrypted with key `k1`
WHEN decryption is attempted with a different key `k2`
THEN `cryptography.fernet.InvalidToken` is raised

### 1.5 Corrupted Ciphertext Rejection
GIVEN a valid Fernet token that has been tampered with
WHEN decryption is attempted
THEN `cryptography.fernet.InvalidToken` is raised

## Requirement 2: Model Module

### 2.1 Inference Output Format
GIVEN any non-empty string prompt `s`
WHEN `run_model(s)` is called
THEN the return value equals `"[AI RESPONSE]: {s} (processed)"` exactly

### 2.2 Deterministic Output
GIVEN the same prompt string
WHEN `run_model()` is called multiple times
THEN it returns the same result every time

## Requirement 3: Node Processing

### 3.1 First Node Decrypts Input
GIVEN a Node with `is_first=True`
WHEN `process()` is called with encrypted data and a valid key
THEN the node decrypts the data before running inference
AND logs `"[{name}] Decrypting..."` to stdout

### 3.2 Last Node Encrypts Output
GIVEN a Node with `is_last=True`
WHEN `process()` is called
THEN the node encrypts the inference result before returning
AND logs `"[{name}] Encrypting output..."` to stdout
AND the returned bytes are a valid Fernet token

### 3.3 Middle Node Passes Plaintext Through
GIVEN a Node with `is_first=False` and `is_last=False`
WHEN `process()` is called with plaintext bytes
THEN the node runs inference only (no decrypt/encrypt)
AND returns plaintext bytes

### 3.4 Inference Always Runs
GIVEN any Node regardless of `is_first`/`is_last` flags
WHEN `process()` is called
THEN inference via `run_model()` is always executed
AND `"[{name}] Running inference..."` is logged to stdout

## Requirement 4: Pipeline Orchestration

### 4.1 Three-Node Sequential Processing
GIVEN an encrypted prompt and a valid session key
WHEN `pipeline.run()` is called
THEN data passes through exactly 3 nodes in order: NodeA → NodeB → NodeC

### 4.2 Correct Flag Assignment
GIVEN the pipeline is running
WHEN nodes are invoked
THEN NodeA receives `is_first=True, is_last=False`
AND NodeB receives `is_first=False, is_last=False`
AND NodeC receives `is_first=False, is_last=True`

### 4.3 Pipeline Returns Encrypted Response
GIVEN a completed pipeline run
WHEN the result is returned to the caller
THEN the result is a valid Fernet token that can be decrypted with the session key

### 4.4 Pipeline Preserves Original Prompt
GIVEN any user prompt
WHEN the prompt is encrypted, sent through the pipeline, and the response is decrypted
THEN the final decrypted output contains the original prompt verbatim

## Requirement 5: Client CLI

### 5.1 User Input Collection
GIVEN the client is started
WHEN the program runs
THEN it prompts the user for text input

### 5.2 Session Key Generation and Display
GIVEN the user has entered a prompt
WHEN the session key is generated
THEN the key is displayed in base64 format with prefix `"[Client] Generated Session Key:"`

### 5.3 Encrypted Prompt Display
GIVEN the prompt has been encrypted
WHEN the encrypted prompt is ready
THEN it is displayed with prefix `"[Client] Encrypted Prompt:"`

### 5.4 Encrypted Response Display
GIVEN the pipeline has returned an encrypted response
WHEN the response is received
THEN it is displayed with prefix `"[Client] Encrypted Response:"`

### 5.5 Final Decrypted Output Display
GIVEN the encrypted response has been decrypted
WHEN the final output is ready
THEN it is displayed with prefix `"[Client] Decrypted Final Output:"`

### 5.6 Empty Input Handling
GIVEN the user enters an empty string
WHEN input validation occurs
THEN the client rejects the input and either re-prompts or exits with a message

## Requirement 6: CLI Log Transparency

### 6.1 Full Lifecycle Visibility
GIVEN a complete pipeline run
WHEN the CLI output is reviewed
THEN all of the following are visible in order:
- Original prompt acknowledgment
- Session key (base64)
- Encrypted prompt
- NodeA decrypt + inference logs
- NodeB inference log
- NodeC inference + encrypt logs
- Encrypted response
- Final decrypted output

### 6.2 Node Log Prefixes
GIVEN any node operation
WHEN a log line is printed
THEN it is prefixed with the node name in brackets (e.g., `[NodeA]`, `[NodeB]`, `[NodeC]`)

## Requirement 7: Portability for Team Integration

### 7.1 Standalone Crypto Module
GIVEN the `crypto.py` module
WHEN it is imported into another project
THEN it works independently with no dependencies on pipeline, node, or client modules

### 7.2 Configurable Node Security Logic
GIVEN the `Node.process()` method
WHEN `is_first` and `is_last` flags are configured
THEN the security behavior (decrypt/encrypt) adapts accordingly, supporting different pipeline topologies
