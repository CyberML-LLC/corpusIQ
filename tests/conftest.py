# Copyright 2026 CyberML (https://cyberml.ai)
# SPDX-License-Identifier: Apache-2.0
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "eval"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")  # generators are faked in tests
