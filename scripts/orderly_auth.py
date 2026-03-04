#!/usr/bin/env python3
"""
orderly_auth.py — Orderly Network authentication for Solana wallets.

Handles:
  1. Account registration (one-time)
  2. Orderly key generation (ed25519 trading keys)
  3. Request signing for REST API calls
  4. Account ID derivation

Based on Orderly JS SDK's Solana adapter implementation.
"""

import json
import time
import struct
import hashlib
import base64
import os
from pathlib import Path

import base58
import requests
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey
)
from cryptography.hazmat.primitives import serialization

# Suppress urllib3 warnings on older Python
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ────────────────────────────────────────────────────────────────

ORDERLY_API_BASE = "https://api-evm.orderly.org"  # unified API for all chains
BROKER_ID = "raydium"
SOLANA_CHAIN_ID = 900900900
CONFIG_DIR = Path(__file__).parent.parent / "config"


# ── Solana Key Utilities ──────────────────────────────────────────────────

def load_solana_keypair(privkey_b58: str) -> tuple:
    """Return (secret_bytes_32, public_bytes_32, address_b58)."""
    raw = base58.b58decode(privkey_b58)
    assert len(raw) == 64, f"Expected 64-byte Solana keypair, got {len(raw)}"
    secret = raw[:32]
    pubkey = raw[32:]
    address = base58.b58encode(pubkey).decode()
    return secret, pubkey, address


def solana_sign(secret_32: bytes, message: bytes) -> bytes:
    """Sign a message with Solana ed25519 key (first 32 bytes of keypair)."""
    key = Ed25519PrivateKey.from_private_bytes(secret_32)
    return key.sign(message)


# ── ABI Encoding (minimal, matches ethers.AbiCoder) ──────────────────────

def _pad32(b: bytes) -> bytes:
    """Left-pad bytes to 32 bytes."""
    return b'\x00' * (32 - len(b)) + b


def _uint256(n: int) -> bytes:
    return n.to_bytes(32, 'big')


def _bytes32(b: bytes) -> bytes:
    assert len(b) == 32
    return b


def keccak256(data: bytes) -> bytes:
    """Keccak-256 hash."""
    from hashlib import sha3_256
    # Python's hashlib sha3_256 is NOT keccak-256. We need pysha3 or do it manually.
    # Let's use a pure approach via pyethash or just import from a lib we have
    try:
        import sha3
        k = sha3.keccak_256()
        k.update(data)
        return k.digest()
    except ImportError:
        pass
    # Fallback: use eth_hash or eth_utils
    try:
        from Crypto.Hash import keccak as _keccak
        k = _keccak.new(digest_bits=256)
        k.update(data)
        return k.digest()
    except ImportError:
        pass
    # Last resort: use web3
    try:
        from web3 import Web3
        return Web3.keccak(data)
    except ImportError:
        pass
    raise ImportError("Need pysha3, pycryptodome, or web3 for keccak256")


def solidity_packed_keccak256_string(s: str) -> bytes:
    """keccak256(abi.encodePacked(string)) — same as ethers solidityPackedKeccak256(['string'], [s])."""
    return keccak256(s.encode('utf-8'))


def abi_encode(types: list, values: list) -> bytes:
    """Minimal ABI encoder for bytes32, uint256, uint64."""
    parts = []
    for t, v in zip(types, values):
        if t == 'bytes32':
            if isinstance(v, bytes):
                parts.append(_bytes32(v))
            else:
                raise ValueError(f"bytes32 expects bytes, got {type(v)}")
        elif t == 'uint256':
            parts.append(_uint256(int(v)))
        elif t == 'uint64':
            parts.append(_uint256(int(v)))  # ABI encodes uint64 as 32 bytes too
        elif t == 'string':
            raise ValueError("Use solidity_packed_keccak256_string for strings")
        else:
            raise ValueError(f"Unsupported ABI type: {t}")
    return b''.join(parts)


# ── Message Signing (Solana-style: keccak hash → text-encode hex → ed25519 sign) ─

def sign_registration_message(secret_32: bytes, broker_id: str, chain_id: int,
                               timestamp: int, registration_nonce: str) -> tuple:
    """
    Sign a registration message for Orderly (Solana flow).
    Returns (message_dict, signature_hex).
    """
    message = {
        "brokerId": broker_id,
        "chainId": chain_id,
        "timestamp": timestamp,
        "registrationNonce": registration_nonce,
    }

    broker_hash = solidity_packed_keccak256_string(broker_id)

    encoded = abi_encode(
        ['bytes32', 'uint256', 'uint256', 'uint256'],
        [broker_hash, chain_id, timestamp, int(registration_nonce)]
    )
    msg_hash = keccak256(encoded)
    msg_hex = msg_hash.hex()
    msg_bytes = msg_hex.encode('utf-8')  # text-encode the hex string

    sig = solana_sign(secret_32, msg_bytes)
    sig_hex = "0x" + sig.hex()

    return message, sig_hex


def sign_add_key_message(secret_32: bytes, broker_id: str, chain_id: int,
                          orderly_public_key: str, scope: str = "read,trading",
                          expiration_days: int = 365) -> tuple:
    """
    Sign an AddOrderlyKey message for Orderly (Solana flow).
    Returns (message_dict, signature_hex).
    """
    timestamp = int(time.time() * 1000)
    expiration = timestamp + 1000 * 60 * 60 * 24 * expiration_days

    message = {
        "brokerId": broker_id,
        "chainType": "SOL",
        "orderlyKey": orderly_public_key,
        "scope": scope,
        "chainId": chain_id,
        "timestamp": timestamp,
        "expiration": expiration,
    }

    broker_hash = solidity_packed_keccak256_string(broker_id)
    key_hash = solidity_packed_keccak256_string(orderly_public_key)
    scope_hash = solidity_packed_keccak256_string(scope)

    encoded = abi_encode(
        ['bytes32', 'bytes32', 'bytes32', 'uint256', 'uint256', 'uint256'],
        [broker_hash, key_hash, scope_hash, chain_id, timestamp, expiration]
    )
    msg_hash = keccak256(encoded)
    msg_hex = msg_hash.hex()
    msg_bytes = msg_hex.encode('utf-8')

    sig = solana_sign(secret_32, msg_bytes)
    sig_hex = "0x" + sig.hex()

    return message, sig_hex


# ── Orderly Key Generation ────────────────────────────────────────────────

def generate_orderly_keypair() -> tuple:
    """
    Generate a new ed25519 keypair for Orderly API auth.
    Returns (private_key_b58, public_key_str) where public_key_str = "ed25519:<b58pubkey>".
    """
    privkey = Ed25519PrivateKey.generate()
    privkey_bytes = privkey.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption()
    )
    pubkey_bytes = privkey.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw
    )
    privkey_b58 = base58.b58encode(privkey_bytes).decode()
    pubkey_str = f"ed25519:{base58.b58encode(pubkey_bytes).decode()}"
    return privkey_b58, pubkey_str


# ── Orderly API Request Signing ───────────────────────────────────────────

def sign_request(orderly_secret_b58: str, method: str, path: str,
                  body: dict = None) -> tuple:
    """
    Sign an Orderly API request using the Orderly trading key.
    Returns (headers dict, body_json_str) tuple.
    The body_json_str should be used with requests.post(..., data=body_json_str)
    to ensure the signed payload matches what's sent.
    """
    privkey_bytes = base58.b58decode(orderly_secret_b58)
    privkey = Ed25519PrivateKey.from_private_bytes(privkey_bytes[:32])
    pubkey_bytes = privkey.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw
    )
    orderly_key = f"ed25519:{base58.b58encode(pubkey_bytes).decode()}"

    timestamp = int(time.time() * 1000)
    message = f"{timestamp}{method.upper()}{path}"
    body_json = None
    if body:
        body_json = json.dumps(body, separators=(',', ':'))
        message += body_json

    signature = privkey.sign(message.encode('utf-8'))
    sig_b64 = base64.urlsafe_b64encode(signature).decode('utf-8')

    headers = {
        "orderly-timestamp": str(timestamp),
        "orderly-key": orderly_key,
        "orderly-signature": sig_b64,
    }
    return headers, body_json


# ── Account ID derivation ─────────────────────────────────────────────────

def derive_account_id(address: str, broker_id: str) -> str:
    """Derive Orderly account ID from address + broker."""
    # For Solana, we need to pad the address to bytes32
    addr_bytes = base58.b58decode(address)
    # Pad to 32 bytes (left-pad with zeros, like an EVM address)
    addr_padded = b'\x00' * (32 - len(addr_bytes)) + addr_bytes

    broker_hash = solidity_packed_keccak256_string(broker_id)

    # ABI encode (address as bytes32, broker_hash as bytes32)
    encoded = abi_encode(['bytes32', 'bytes32'], [addr_padded, broker_hash])
    account_id = keccak256(encoded)
    return "0x" + account_id.hex()


# ── Full Setup Flow ───────────────────────────────────────────────────────

class OrderlyClient:
    """
    Complete Orderly API client for Solana wallets.
    Handles registration, key management, and authenticated requests.
    """

    def __init__(self, solana_privkey_b58: str, broker_id: str = BROKER_ID,
                 base_url: str = ORDERLY_API_BASE):
        self.secret_32, self.pubkey_32, self.address = load_solana_keypair(solana_privkey_b58)
        self.broker_id = broker_id
        self.base_url = base_url
        self.session = requests.Session()

        # Orderly trading credentials (set after setup)
        self.account_id = None
        self.orderly_secret_b58 = None
        self.orderly_key = None  # "ed25519:..."

        # Try to load saved credentials
        self._load_config()

    def _config_path(self) -> Path:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return CONFIG_DIR / "orderly_credentials.json"

    def _save_config(self):
        data = {
            "account_id": self.account_id,
            "orderly_secret": self.orderly_secret_b58,
            "orderly_key": self.orderly_key,
            "address": self.address,
            "broker_id": self.broker_id,
        }
        self._config_path().write_text(json.dumps(data, indent=2))

    def _load_config(self):
        p = self._config_path()
        if p.exists():
            data = json.loads(p.read_text())
            if data.get("address") == self.address and data.get("broker_id") == self.broker_id:
                self.account_id = data.get("account_id")
                self.orderly_secret_b58 = data.get("orderly_secret")
                self.orderly_key = data.get("orderly_key")

    @property
    def is_ready(self) -> bool:
        return all([self.account_id, self.orderly_secret_b58, self.orderly_key])

    # ── Registration ──────────────────────────────────────────────────────

    def check_registration(self) -> bool:
        """Check if this wallet is already registered with the broker."""
        # For Solana, /v1/get_account doesn't accept base58 addresses.
        # Use /v1/public/account with derived account_id instead.
        derived_id = derive_account_id(self.address, self.broker_id)
        r = self.session.get(
            f"{self.base_url}/v1/public/account",
            params={"account_id": derived_id}
        )
        data = r.json()
        if data.get("success") and data.get("data", {}).get("address") == self.address:
            self.account_id = derived_id
            return True
        return False

    def register_account(self) -> str:
        """Register a new account. Returns account_id."""
        # Get nonce
        r = self.session.get(f"{self.base_url}/v1/registration_nonce")
        nonce_data = r.json()
        if not nonce_data.get("success"):
            raise RuntimeError(f"Failed to get nonce: {nonce_data}")
        nonce = nonce_data["data"]["registration_nonce"]

        timestamp = int(time.time() * 1000)
        message, signature = sign_registration_message(
            self.secret_32, self.broker_id, SOLANA_CHAIN_ID, timestamp, nonce
        )

        # Add chainType for Solana
        message["chainType"] = "SOL"

        r = self.session.post(
            f"{self.base_url}/v1/register_account",
            json={
                "message": message,
                "signature": signature,
                "userAddress": self.address,
            }
        )
        result = r.json()
        if result.get("success"):
            self.account_id = result["data"]["account_id"]
            print(f"✅ Registered! Account ID: {self.account_id}")
        elif result.get("code") == -1604:
            # Already registered — derive the account ID
            self.account_id = derive_account_id(self.address, self.broker_id)
            print(f"✅ Already registered. Account ID: {self.account_id[:20]}...")
        else:
            raise RuntimeError(f"Registration failed: {result}")

        return self.account_id

    def add_orderly_key(self) -> tuple:
        """Generate and register a new Orderly trading key. Returns (key, secret)."""
        secret_b58, pub_key_str = generate_orderly_keypair()

        message, signature = sign_add_key_message(
            self.secret_32, self.broker_id, SOLANA_CHAIN_ID,
            pub_key_str, scope="read,trading", expiration_days=365
        )

        r = self.session.post(
            f"{self.base_url}/v1/orderly_key",
            json={
                "message": message,
                "signature": signature,
                "userAddress": self.address,
            }
        )
        result = r.json()
        if not result.get("success"):
            raise RuntimeError(f"Add key failed: {result}")

        self.orderly_secret_b58 = secret_b58
        self.orderly_key = pub_key_str
        print(f"✅ Orderly key added: {pub_key_str}")
        return pub_key_str, secret_b58

    def setup(self) -> bool:
        """Full setup: register if needed, add trading key if needed."""
        if self.is_ready:
            print(f"✅ Already configured. Account: {self.account_id[:16]}...")
            return True

        # Step 1: Check/register account
        if not self.account_id:
            print("Checking registration...")
            if not self.check_registration():
                print("Registering new account...")
                self.register_account()
            else:
                print(f"✅ Already registered. Account: {self.account_id[:16]}...")

        # Step 2: Add orderly key if needed
        if not self.orderly_secret_b58:
            print("Generating trading key...")
            self.add_orderly_key()

        # Save
        self._save_config()
        print(f"✅ Setup complete! Credentials saved to {self._config_path()}")
        return True

    # ── Authenticated API calls ───────────────────────────────────────────

    def _auth_headers(self, method: str, path: str, body: dict = None) -> tuple:
        """
        Returns (headers dict, body_json_str or None).
        body_json_str should be used with data= param in requests.
        """
        if not self.is_ready:
            raise RuntimeError("Call setup() first")
        headers, body_json = sign_request(self.orderly_secret_b58, method, path, body)
        headers["orderly-account-id"] = self.account_id
        if method.upper() in ("GET", "DELETE"):
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            headers["Content-Type"] = "application/json"
        return headers, body_json

    def get(self, path: str, params: dict = None) -> dict:
        full_path = path
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            full_path = f"{path}?{qs}"
        headers, _ = self._auth_headers("GET", full_path)
        r = self.session.get(f"{self.base_url}{full_path}", headers=headers)
        return r.json()

    def post(self, path: str, body: dict = None) -> dict:
        headers, body_json = self._auth_headers("POST", path, body)
        r = self.session.post(f"{self.base_url}{path}", headers=headers, data=body_json)
        return r.json()

    def put(self, path: str, body: dict = None) -> dict:
        headers, body_json = self._auth_headers("PUT", path, body)
        r = self.session.put(f"{self.base_url}{path}", headers=headers, data=body_json)
        return r.json()

    def delete(self, path: str, params: dict = None) -> dict:
        full_path = path
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            full_path = f"{path}?{qs}"
        headers, _ = self._auth_headers("DELETE", full_path)
        r = self.session.delete(f"{self.base_url}{full_path}", headers=headers)
        return r.json()

    # ── Trading Convenience Methods ───────────────────────────────────────

    def get_account_info(self) -> dict:
        return self.get("/v1/client/info")

    def get_positions(self) -> dict:
        return self.get("/v1/positions")

    def get_position(self, symbol: str) -> dict:
        return self.get("/v1/position", {"symbol": symbol})

    def create_order(self, symbol: str, side: str, order_type: str,
                      order_quantity: float, order_price: float = None,
                      reduce_only: bool = False,
                      client_order_id: str = None) -> dict:
        body = {
            "symbol": symbol,
            "side": side.upper(),
            "order_type": order_type.upper(),
            "order_quantity": order_quantity,
            "reduce_only": reduce_only,
        }
        if order_price is not None:
            body["order_price"] = order_price
        if client_order_id:
            body["client_order_id"] = client_order_id
        return self.post("/v1/order", body)

    def cancel_order(self, order_id: int = None, client_order_id: str = None,
                      symbol: str = None) -> dict:
        body = {}
        if order_id:
            body["order_id"] = order_id
        if client_order_id:
            body["client_order_id"] = client_order_id
        if symbol:
            body["symbol"] = symbol
        return self.delete("/v1/order", body)

    def get_orders(self, symbol: str = None, status: str = None) -> dict:
        params = {}
        if symbol:
            params["symbol"] = symbol
        if status:
            params["status"] = status
        return self.get("/v1/orders", params)

    def get_trades(self, symbol: str = None) -> dict:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self.get("/v1/trades", params)

    def get_balance(self) -> dict:
        return self.get("/v1/client/holding")

    def get_leverage(self, symbol: str) -> dict:
        return self.get("/v1/client/futures_leverage_setting")

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        return self.post("/v1/client/leverage", {
            "symbol": symbol,
            "leverage": leverage,
        })

    def test_micro_trade(self, symbol: str = "PERP_SOL_USDC", min_notional: float = 10.0) -> bool:
        """
        Test order execution with a micro trade.
        Opens and immediately closes a minimum-size position to verify trading works.
        Returns True if successful, raises RuntimeError if failed.
        """
        print(f"🧪 Testing micro trade on {symbol}...")

        # Get current price
        r = requests.get(f"{self.base_url}/v1/public/market_trades",
                        params={"symbol": symbol, "limit": 1}, timeout=5)
        data = r.json()
        if not data.get("success") or not data["data"]["rows"]:
            raise RuntimeError(f"Failed to get price: {data}")
        current_price = float(data["data"]["rows"][0]["executed_price"])

        # Calculate minimum quantity (min_notional / price, rounded up to 0.01)
        import math
        min_qty = math.ceil(min_notional / current_price * 100) / 100
        print(f"   Current price: ${current_price:.3f}")
        print(f"   Micro trade quantity: {min_qty} SOL (notional: ${min_qty * current_price:.2f})")

        # Open position
        print(f"   Opening BUY {min_qty} {symbol}...")
        open_result = self.create_order(
            symbol=symbol,
            side="BUY",
            order_type="MARKET",
            order_quantity=min_qty,
        )

        if not open_result.get("success"):
            raise RuntimeError(f"Failed to open position: {open_result}")

        order_id = open_result["data"]["order_id"]
        print(f"   Opened: order_id={order_id}")

        # Wait a moment for order to fill
        import time
        time.sleep(1)

        # Close position
        print(f"   Closing SELL {min_qty} {symbol}...")
        close_result = self.create_order(
            symbol=symbol,
            side="SELL",
            order_type="MARKET",
            order_quantity=min_qty,
            reduce_only=True,
        )

        if not close_result.get("success"):
            # Try to clean up the open position
            print(f"   ⚠️ Failed to close position: {close_result}")
            print(f"   ⚠️ You may have an open position. Close manually in Raydium Perps UI.")
            raise RuntimeError(f"Failed to close position: {close_result}")

        print(f"   Closed: order_id={close_result['data']['order_id']}")
        print(f"✅ MICRO TRADE TEST PASSED")
        return True


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Load key from credentials
    cred_path = Path(__file__).parent.parent.parent / "memory" / "credentials.md"
    if not cred_path.exists():
        print("❌ No credentials file found")
        sys.exit(1)

    # Extract Solana private key from credentials.md
    sol_key = None
    for line in cred_path.read_text().splitlines():
        if "Private Key:" in line and "5XY4" in line:
            # Format: **Private Key:** 5XY4...
            sol_key = line.split("**")[-1].strip()
            if sol_key.startswith(":"):
                sol_key = sol_key[1:].strip()
            break

    if not sol_key:
        print("❌ Could not find Solana private key in credentials")
        sys.exit(1)

    client = OrderlyClient(sol_key)

    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        client.setup()
    elif len(sys.argv) > 1 and sys.argv[1] == "info":
        if not client.is_ready:
            client.setup()
        print(json.dumps(client.get_account_info(), indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "balance":
        if not client.is_ready:
            client.setup()
        print(json.dumps(client.get_balance(), indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run micro trade test
        if not client.is_ready:
            client.setup()
        success = client.test_micro_trade()
        sys.exit(0 if success else 1)
    else:
        print(f"Solana address: {client.address}")
        print(f"Configured: {client.is_ready}")
        if client.is_ready:
            print(f"Account ID: {client.account_id[:20]}...")
            print(f"Orderly Key: {client.orderly_key}")
        print(f"\nUsage: python3 {sys.argv[0]} [setup|info|balance]")
