"""Base62 share codes for Gradio share URLs.

Gradio labels a share tunnel with a hexadecimal subdomain (for example
``https://07ff8706ab.gradio.live``). We re-encode that label in base 62 so the
resulting code mixes digits and both letter cases and is easy to read out.

This module is deliberately free of the ``gradio`` import, so the pure maths can
be unit-tested without the heavy stack.

The base62 alphabet MUST match the frontend's decoder
(``frontend/src/lib/base62.ts``): digits, then A-Z, then a-z. Any other order
produces a code the web client cannot decode.
"""

from urllib.parse import urlparse

_BASE16_ALPHABET = "0123456789abcdef"
_BASE36_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
_BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _rebase(text: str, from_alphabet: str, to_alphabet: str) -> str:
    """Re-encode `text` from one positional alphabet into another."""
    value = 0
    for char in text:
        value = value * len(from_alphabet) + from_alphabet.index(char)
    if value == 0:
        return to_alphabet[0]
    digits = []
    base = len(to_alphabet)
    while value:
        value, remainder = divmod(value, base)
        digits.append(to_alphabet[remainder])
    return "".join(reversed(digits))


def share_code(share_url: str) -> str:
    """Derive a base-62 code from the label of a Gradio share URL.

    The label is treated as a hexadecimal (base-16) number -- the base
    Gradio's share URLs use -- and re-encoded as base 62 (0-9, A-Z, a-z).
    Labels that are not valid hex are retried as base 36, and anything that
    fits neither alphabet is returned unchanged.
    """
    # Parse without requiring a scheme so bare "host/path" inputs still work.
    parsed = urlparse(share_url if "://" in share_url else "//" + share_url)
    host = parsed.hostname or parsed.netloc or parsed.path
    label = host.split(".")[0].lower() if host else ""
    if not label:
        return label
    if all(char in _BASE16_ALPHABET for char in label):
        return _rebase(label, _BASE16_ALPHABET, _BASE62_ALPHABET)
    if all(char in _BASE36_ALPHABET for char in label):
        return _rebase(label, _BASE36_ALPHABET, _BASE62_ALPHABET)
    return label
