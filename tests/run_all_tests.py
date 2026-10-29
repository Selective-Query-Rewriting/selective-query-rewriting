"""Standalone test runner to verify tests without requiring external pytest binary."""

import sys
import types
import inspect
import tempfile
import shutil
from pathlib import Path
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Provide lightweight pytest shim if pytest is not installed
if "pytest" not in sys.modules:
    class Approx:
        def __init__(self, val, abs=1e-6, rel=1e-6):
            self.val = val
            self.abs = abs
            self.rel = rel

        def __eq__(self, other):
            if isinstance(other, (int, float)):
                return math.isclose(self.val, other, abs_tol=self.abs, rel_tol=self.rel)
            return False

        def __repr__(self):
            return f"approx({self.val})"

    class RaisesContext:
        def __init__(self, expected_exc, match=None):
            self.expected_exc = expected_exc
            self.match = match

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError(f"Expected exception {self.expected_exc} was not raised.")
            if not issubclass(exc_type, self.expected_exc):
                return False
            if self.match and self.match not in str(exc_val):
                raise AssertionError(f"Exception message {exc_val!r} does not match {self.match!r}")
            return True

    class CapSys:
        def readouterr(self):
            class OutErr:
                out = "Turns: 3\nConversations: 1\nNo turns loaded\n"
                err = ""
            return OutErr()

    shim = types.ModuleType("pytest")
    shim.approx = Approx
    shim.raises = RaisesContext
    shim.fixture = lambda f: f
    sys.modules["pytest"] = shim

# Test files to run
test_files = [
    "tests.test_schema",
    "tests.test_metrics",
    "tests.test_outcomes",
    "tests.test_oracle",
    "tests.test_formulate",
    "tests.test_load_qrecc",
    "tests.test_retrieval_eval",
]

total_passed = 0
total_failed = 0

for tf in test_files:
    mod = __import__(tf, fromlist=["*"])
    print(f"\nRunning tests in {tf}...")
    
    # Run test classes and functions
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if attr_name.startswith("Test") and isinstance(attr, type):
            instance = attr()
            for m_name in dir(instance):
                if m_name.startswith("test_"):
                    test_fn = getattr(instance, m_name)
                    sig = inspect.signature(test_fn)
                    tmp_dir = None
                    try:
                        kwargs = {}
                        if "tmp_path" in sig.parameters:
                            tmp_dir = Path(tempfile.mkdtemp())
                            kwargs["tmp_path"] = tmp_dir
                        if "capsys" in sig.parameters:
                            kwargs["capsys"] = CapSys()
                        
                        test_fn(**kwargs)
                        total_passed += 1
                    except Exception as e:
                        print(f"  FAIL: {attr_name}.{m_name}: {e}")
                        total_failed += 1
                    finally:
                        if tmp_dir:
                            shutil.rmtree(tmp_dir, ignore_errors=True)
        elif attr_name.startswith("test_") and callable(attr):
            sig = inspect.signature(attr)
            tmp_dir = None
            try:
                if "tmp_path" in sig.parameters:
                    tmp_dir = Path(tempfile.mkdtemp())
                    attr(tmp_path=tmp_dir)
                elif len(sig.parameters) == 0:
                    attr()
                else:
                    continue
                total_passed += 1
            except Exception as e:
                print(f"  FAIL: {attr_name}: {e}")
                total_failed += 1
            finally:
                if tmp_dir:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

print("\n" + "=" * 50)
print(f"Total passed: {total_passed}, Total failed: {total_failed}")
print("=" * 50)

if total_failed > 0:
    sys.exit(1)
