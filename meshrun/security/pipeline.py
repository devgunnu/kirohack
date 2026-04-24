"""Pipeline module: Simulates encrypted hop-to-hop transport across 3 nodes."""

from node import Node


def run(encrypted_prompt: bytes, key: bytes) -> bytes:
    """
    Run the 3-node pipeline. Data is encrypted at every hop,
    simulating real network transport where each node:
      1. Receives encrypted bytes
      2. Decrypts → processes → re-encrypts
      3. Forwards encrypted bytes to next node

    Args:
        encrypted_prompt: AES-256-GCM encrypted prompt from client
        key: 32-byte shared AES session key

    Returns:
        AES-256-GCM encrypted response from the last node
    """
    node_a = Node("NodeA")
    node_b = Node("NodeB")
    node_c = Node("NodeC")

    print("─" * 50)
    print("  HOP 1: Client → NodeA")
    print("─" * 50)
    encrypted_a = node_a.process(encrypted_prompt, key)

    print("─" * 50)
    print("  HOP 2: NodeA → NodeB")
    print("─" * 50)
    encrypted_b = node_b.process(encrypted_a, key)

    print("─" * 50)
    print("  HOP 3: NodeB → NodeC")
    print("─" * 50)
    encrypted_c = node_c.process(encrypted_b, key)

    return encrypted_c
