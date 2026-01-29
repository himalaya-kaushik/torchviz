# """
# Example usage of DimViz for debugging PyTorch models.
# """

# import torch
# import torch.nn as nn
# from dimviz import DimViz, visualize, export_log


# # Example 1: Basic CNN Model
# print("=" * 80)
# print("Example 1: Basic CNN Model")
# print("=" * 80)

# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(32 * 8 * 8, 128)
#         self.fc2 = nn.Linear(128, 10)
    
#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# model = SimpleCNN()
# x = torch.randn(4, 3, 32, 32)

# with DimViz(verbose=True):
#     output = model(x)

# print(f"\nFinal output shape: {output.shape}")


# # Example 2: Only Shape Changes
# print("\n" + "=" * 80)
# print("Example 2: Track Only Shape Changes")
# print("=" * 80)

# with DimViz(verbose=False):
#     output = model(x)


# # Example 3: Memory Tracking
# print("\n" + "=" * 80)
# print("Example 3: Memory Tracking")
# print("=" * 80)

# with DimViz(track_memory=True, verbose=False):
#     output = model(x)


# # Example 4: Filter Specific Operations
# print("\n" + "=" * 80)
# print("Example 4: Filter Specific Operations (conv2d and linear only)")
# print("=" * 80)

# with DimViz(filter_ops=['conv2d', 'linear']):
#     output = model(x)


# # Example 5: Using Decorator
# print("\n" + "=" * 80)
# print("Example 5: Using @visualize Decorator")
# print("=" * 80)

# @visualize(verbose=False)
# def inference(model, x):
#     """Run inference with shape tracking."""
#     return model(x)

# result = inference(model, x)


# # Example 6: Export Logs
# print("\n" + "=" * 80)
# print("Example 6: Export Logs to Files")
# print("=" * 80)

# with DimViz(show_summary=False) as viz:
#     output = model(x)

# # Export to different formats
# export_log(viz.get_log(), 'cnn_shapes.json')
# export_log(viz.get_log(), 'cnn_shapes.csv')
# export_log(viz.get_log(), 'cnn_shapes.txt')

# print("\n✅ Logs exported to:")
# print("  - cnn_shapes.json")
# print("  - cnn_shapes.csv")
# print("  - cnn_shapes.txt")


# # Example 7: Transformer-like Model
# print("\n" + "=" * 80)
# print("Example 7: Transformer-like Attention Mechanism")
# print("=" * 80)

# class SimpleAttention(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.key = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)
#         self.out = nn.Linear(embed_dim, embed_dim)
    
#     def forward(self, x):
#         # x shape: (batch, seq_len, embed_dim)
#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)
        
#         # Attention scores
#         scores = torch.matmul(Q, K.transpose(-2, -1))
#         scores = scores / (x.size(-1) ** 0.5)
#         attn_weights = torch.softmax(scores, dim=-1)
        
#         # Apply attention
#         out = torch.matmul(attn_weights, V)
#         out = self.out(out)
#         return out


# attention_model = SimpleAttention(embed_dim=64)
# x = torch.randn(2, 10, 64)  # (batch=2, seq_len=10, embed_dim=64)

# with DimViz(verbose=False):
#     output = attention_model(x)

# print(f"\nInput shape: {x.shape}")
# print(f"Output shape: {output.shape}")


# # Example 8: Compare Two Models
# print("\n" + "=" * 80)
# print("Example 8: Compare Two Model Versions")
# print("=" * 80)

# from dimviz.exporter import compare_logs

# # Model v1 - Simple
# class ModelV1(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(10, 20)
#         self.fc2 = nn.Linear(20, 5)
    
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         return self.fc2(x)


# # Model v2 - With extra layer
# class ModelV2(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(10, 20)
#         self.fc2 = nn.Linear(20, 15)  # Extra layer
#         self.fc3 = nn.Linear(15, 5)
    
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)


# x = torch.randn(4, 10)

# # Track both models
# with DimViz(show_summary=False) as viz1:
#     output1 = ModelV1()(x)

# with DimViz(show_summary=False) as viz2:
#     output2 = ModelV2()(x)

# # Compare
# comparison = compare_logs(viz1.get_log(), viz2.get_log(), "ModelV1", "ModelV2")

# print(f"\n📊 Comparison Results:")
# print(f"  Length difference: {comparison['length_diff']} operations")
# if 'extra_ops' in comparison:
#     print(f"  Extra operations in {comparison['extra_ops']['in']}: {comparison['extra_ops']['count']}")
# if comparison['differences']:
#     print(f"  Found {len(comparison['differences'])} differences")


# # Example 9: Debugging Shape Mismatch
# print("\n" + "=" * 80)
# print("Example 9: Debugging a Shape Mismatch (Intentional Error)")
# print("=" * 80)

# class BuggyModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Conv2d(3, 16, 3)
#         self.fc = nn.Linear(16 * 30 * 30, 10)  # Wrong size!
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)  # This will crash!
#         return x


# buggy_model = BuggyModel()
# x = torch.randn(2, 3, 32, 32)

# try:
#     with DimViz(verbose=False):
#         output = buggy_model(x)
# except RuntimeError as e:
#     print(f"\n⚠️  Caught shape mismatch!")
#     print(f"Error: {str(e)[:100]}...")
#     print("\n💡 DimViz helps you see where the mismatch occurred!")


# print("\n" + "=" * 80)
# print("All examples completed! 🎉")
# print("=" * 80)


# # test.py
# # """
# # Hard/adversarial tests for the EXACT DimViz code you pasted.

# # What we test:
# # 1) Basic CNN correctness + log non-empty
# # 2) verbose=False logs fewer rows than verbose=True (shape changes only)
# # 3) track_memory=True produces 6 columns (Mem In/Out present)
# # 4) filter_ops works with YOUR clean_name values (e.g., 'convolution', 'linear_proj', 'view', etc.)
# # 5) Tuple/list/dict output functions don't crash
# # 6) Transformer-ish pipeline doesn't crash and output shape matches
# # 7) Autograd backward still works
# # 8) Intentional mismatch raises and still prints crash message (log may be partial)

# # Run:
# #   python test.py
# # """

# # import traceback
# # import torch
# # import torch.nn as nn
# # from dimviz import DimViz


# # def assert_(cond, msg):
# #     if not cond:
# #         raise AssertionError(msg)

# # def section(name):
# #     print("\n" + "=" * 100)
# #     print(name)
# #     print("=" * 100)


# # # ------------------ Models ------------------

# # class SimpleCNN(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
# #         self.pool = nn.MaxPool2d(2, 2)
# #         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
# #         self.fc1 = nn.Linear(32 * 8 * 8, 128)
# #         self.fc2 = nn.Linear(128, 10)

# #     def forward(self, x):
# #         x = self.pool(torch.relu(self.conv1(x)))
# #         x = self.pool(torch.relu(self.conv2(x)))
# #         x = x.view(x.size(0), -1)
# #         x = torch.relu(self.fc1(x))
# #         x = self.fc2(x)
# #         return x

# # class TupleOut(nn.Module):
# #     def forward(self, x):
# #         return x + 1, x.transpose(-1, -2)

# # class ListOut(nn.Module):
# #     def forward(self, x):
# #         return [x, x[:, :2], x.sum(dim=-1)]

# # class DictOut(nn.Module):
# #     def forward(self, x):
# #         return {"a": x * 2, "b": x.mean(dim=-1)}

# # class Transformerish(nn.Module):
# #     def __init__(self, d_model=32, n_heads=4):
# #         super().__init__()
# #         assert d_model % n_heads == 0
# #         self.h = n_heads
# #         self.dh = d_model // n_heads
# #         self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
# #         self.out = nn.Linear(d_model, d_model, bias=False)

# #     def forward(self, x):
# #         B, T, D = x.shape
# #         qkv = self.qkv(x)
# #         q, k, v = qkv.chunk(3, dim=-1)
# #         q = q.view(B, T, self.h, self.dh).transpose(1, 2)
# #         k = k.view(B, T, self.h, self.dh).transpose(1, 2)
# #         v = v.view(B, T, self.h, self.dh).transpose(1, 2)
# #         scores = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
# #         probs = torch.softmax(scores, dim=-1)
# #         ctx = probs @ v
# #         ctx = ctx.transpose(1, 2).contiguous().view(B, T, D)
# #         return self.out(ctx)

# # class BackwardTest(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.fc = nn.Linear(16, 16)
# #     def forward(self, x):
# #         return (self.fc(x).tanh()).sum()

# # class BuggyModel(nn.Module):
# #     # wrong in_features on purpose
# #     def __init__(self):
# #         super().__init__()
# #         self.conv = nn.Conv2d(3, 16, 3)         # -> (B,16,30,30)
# #         self.fc = nn.Linear(16 * 28 * 28, 10)   # WRONG
# #     def forward(self, x):
# #         x = self.conv(x)
# #         x = x.view(x.size(0), -1)
# #         return self.fc(x)


# # # ------------------ Tests ------------------

# # def test1_basic_cnn():
# #     section("TEST 1: CNN forward correctness + log non-empty")
# #     m = SimpleCNN()
# #     x = torch.randn(4, 3, 32, 32)

# #     with DimViz(verbose=True, show_summary=False) as viz:
# #         y = m(x)

# #     assert_(tuple(y.shape) == (4, 10), f"Expected (4,10), got {tuple(y.shape)}")
# #     log = viz.get_log()
# #     assert_(isinstance(log, list), "get_log() must return list")
# #     assert_(len(log) > 0, "Expected non-empty log")
# #     assert_(len(log[0]) == 4, f"Expected 4 columns (no memory). Got {len(log[0])}")
# #     print("[PASS] rows:", len(log))

# # def test2_verbose_false_logs_less():
# #     section("TEST 2: verbose=False should log fewer rows than verbose=True")
# #     m = SimpleCNN()
# #     x = torch.randn(4, 3, 32, 32)

# #     with DimViz(verbose=True, show_summary=False) as v_all:
# #         _ = m(x)
# #     with DimViz(verbose=False, show_summary=False) as v_less:
# #         _ = m(x)

# #     a = len(v_all.get_log())
# #     b = len(v_less.get_log())

# #     assert_(a >= b, f"Expected verbose=True rows >= verbose=False rows, got {a} < {b}")
# #     assert_(b > 0, "Expected some rows in verbose=False mode")
# #     print(f"[PASS] verbose=True={a}, verbose=False={b}")

# # def test3_memory_tracking_columns():
# #     section("TEST 3: track_memory=True should add Mem In/Out columns")
# #     m = SimpleCNN()
# #     x = torch.randn(2, 3, 32, 32)

# #     with DimViz(verbose=False, track_memory=True, show_summary=False) as viz:
# #         y = m(x)

# #     assert_(tuple(y.shape) == (2, 10), f"Expected (2,10), got {tuple(y.shape)}")
# #     log = viz.get_log()
# #     assert_(len(log) > 0, "Expected non-empty log")
# #     assert_(len(log[0]) == 6, f"Expected 6 columns with memory. Got {len(log[0])}")
# #     print("[PASS] memory rows:", len(log))

# # def test4_filter_ops():
# #     section("TEST 4: filter_ops works with YOUR op names")
# #     m = SimpleCNN()
# #     x = torch.randn(2, 3, 32, 32)

# #     # In your printed output, ops were named:
# #     # convolution, max_pool2d_with_indices, view, transpose, linear_proj, relu
# #     with DimViz(filter_ops=["convolution", "linear_proj"], show_summary=False) as viz:
# #         _ = m(x)

# #     log = viz.get_log()
# #     if len(log) == 0:
# #         raise AssertionError(
# #             "filter_ops logged 0 rows. Likely mismatch in expected op names.\n"
# #             "Try printing a full verbose log and copy the exact 'Op' values, then use those."
# #         )

# #     ops = set(row[1] for row in log)
# #     assert_(ops.issubset({"convolution", "linear_proj"}), f"filter_ops leaked other ops: {ops}")
# #     print("[PASS] filter_ops rows:", len(log), "ops seen:", ops)

# # def test5_tuple_list_dict_outputs_no_crash():
# #     section("TEST 5: tuple/list/dict outputs should not crash")
# #     x = torch.randn(2, 3, 4)

# #     for mod, name in [(TupleOut(), "TupleOut"), (ListOut(), "ListOut"), (DictOut(), "DictOut")]:
# #         with DimViz(verbose=False, show_summary=False) as viz:
# #             out = mod(x)

# #         # validate output shapes
# #         if name == "TupleOut":
# #             assert_(tuple(out[0].shape) == (2,3,4), "TupleOut[0] mismatch")
# #             assert_(tuple(out[1].shape) == (2,4,3), "TupleOut[1] mismatch")
# #         elif name == "ListOut":
# #             assert_(tuple(out[0].shape) == (2,3,4), "ListOut[0] mismatch")
# #             assert_(tuple(out[1].shape) == (2,2,4), "ListOut[1] mismatch")
# #             assert_(tuple(out[2].shape) == (2,3), "ListOut[2] mismatch")
# #         else:
# #             assert_(tuple(out["a"].shape) == (2,3,4), "DictOut[a] mismatch")
# #             assert_(tuple(out["b"].shape) == (2,3), "DictOut[b] mismatch")

# #         log = viz.get_log()
# #         assert_(len(log) > 0, f"{name}: expected some logged ops")
# #         print(f"[PASS] {name}: rows={len(log)}")

# # def test6_transformerish():
# #     section("TEST 6: transformer-ish attention pipeline")
# #     m = Transformerish(d_model=32, n_heads=4)
# #     x = torch.randn(2, 7, 32)

# #     with DimViz(verbose=False, show_summary=False) as viz:
# #         y = m(x)

# #     assert_(tuple(y.shape) == (2,7,32), f"Expected (2,7,32), got {tuple(y.shape)}")
# #     log = viz.get_log()
# #     assert_(len(log) > 0, "Expected some logged ops")
# #     print("[PASS] transformer-ish rows:", len(log))

# # def test7_backward_autograd():
# #     section("TEST 7: backward works under DimViz")
# #     m = BackwardTest()
# #     x = torch.randn(4, 16, requires_grad=True)

# #     with DimViz(verbose=False, show_summary=False) as viz:
# #         loss = m(x)

# #     loss.backward()
# #     assert_(x.grad is not None, "grad should exist")
# #     assert_(torch.isfinite(x.grad).all().item(), "grad should be finite")
# #     print("[PASS] grad norm:", float(x.grad.norm().detach()))

# # def test8_intentional_mismatch_raises():
# #     section("TEST 8: intentional mismatch should raise")
# #     m = BuggyModel()
# #     x = torch.randn(2, 3, 32, 32)

# #     try:
# #         with DimViz(verbose=False, show_summary=False) as viz:
# #             _ = m(x)
# #         raise AssertionError("Expected RuntimeError due to mismatch, but forward succeeded.")
# #     except RuntimeError as e:
# #         print("[PASS] caught expected RuntimeError:", str(e).split("\n")[0][:120], "...")


# # # ------------------ Runner ------------------

# # def main():
# #     print("Device: cpu")
# #     print("PyTorch:", torch.__version__)

# #     tests = [
# #         test1_basic_cnn,
# #         test2_verbose_false_logs_less,
# #         test3_memory_tracking_columns,
# #         test4_filter_ops,
# #         test5_tuple_list_dict_outputs_no_crash,
# #         test6_transformerish,
# #         test7_backward_autograd,
# #         test8_intentional_mismatch_raises,
# #     ]

# #     failed = 0
# #     for t in tests:
# #         try:
# #             t()
# #         except Exception as e:
# #             failed += 1
# #             print("\n[FAIL]", t.__name__)
# #             print(" ", repr(e))
# #             traceback.print_exc()

# #     section("RESULT")
# #     if failed == 0:
# #         print("✅ ALL TESTS PASSED")
# #         return 0
# #     else:
# #         print(f"❌ {failed} TEST(S) FAILED")
# #         return 1


# # if __name__ == "__main__":
# #     raise SystemExit(main())



# test_all.py
"""
Hard tests for DimViz + exporter utilities, matching your EXACT current API.

Run:
  python test_all.py

Assumptions:
- Your package exposes:
    from dimviz import DimViz, visualize
    from dimviz.exporter import export_log, compare_logs
- Or if everything is in one module for now, adjust imports accordingly.
"""

import os
import re
import json
import csv
import traceback
from pathlib import Path

import torch
import torch.nn as nn

# ---- Adjust these imports if your structure differs ----
from dimviz import DimViz, visualize
from dimviz.exporter import export_log, compare_logs


def assert_(cond, msg):
    if not cond:
        raise AssertionError(msg)

def section(name):
    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def safe_rm(p: Path):
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


# ------------------ Models ------------------

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TupleOut(nn.Module):
    def forward(self, x):
        return x + 1, x.transpose(-1, -2)

class ListOut(nn.Module):
    def forward(self, x):
        return [x, x[:, :2], x.sum(dim=-1)]

class DictOut(nn.Module):
    def forward(self, x):
        return {"a": x * 2, "b": x.mean(dim=-1)}

class Transformerish(nn.Module):
    def __init__(self, d_model=32, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
        probs = torch.softmax(scores, dim=-1)
        ctx = probs @ v

        ctx = ctx.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(ctx)

class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)         # -> (B,16,30,30)
        self.fc = nn.Linear(16 * 28 * 28, 10)   # WRONG on purpose
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ------------------ Tests ------------------

def test_1_basic_e2e():
    section("TEST 1: Basic end-to-end trace + output shape + log non-empty")
    m = SimpleCNN()
    x = torch.randn(4, 3, 32, 32)

    with DimViz(verbose=True, show_summary=False) as dv:
        y = m(x)

    assert_(tuple(y.shape) == (4, 10), f"Expected (4,10), got {tuple(y.shape)}")
    log = dv.get_log()
    assert_(len(log) > 0, "Expected non-empty log")
    assert_(len(log[0]) == 4, "Expected 4 columns without memory")
    print("[PASS] rows:", len(log))


def test_2_verbose_false_logs_less():
    section("TEST 2: verbose=False should log fewer rows than verbose=True")
    m = SimpleCNN()
    x = torch.randn(4, 3, 32, 32)

    with DimViz(verbose=True, show_summary=False) as a:
        _ = m(x)
    with DimViz(verbose=False, show_summary=False) as b:
        _ = m(x)

    la, lb = len(a.get_log()), len(b.get_log())
    assert_(la >= lb, f"Expected verbose=True >= verbose=False, got {la} < {lb}")
    assert_(lb > 0, "Expected some logs in verbose=False")
    print(f"[PASS] verbose=True={la}, verbose=False={lb}")


def test_3_track_memory_columns_and_format():
    section("TEST 3: track_memory=True adds Mem columns and strings end with 'MB'")
    m = SimpleCNN()
    x = torch.randn(2, 3, 32, 32)

    with DimViz(verbose=False, track_memory=True, show_summary=False) as dv:
        y = m(x)

    assert_(tuple(y.shape) == (2, 10), f"Expected (2,10), got {tuple(y.shape)}")
    log = dv.get_log()
    assert_(len(log) > 0, "Expected non-empty log")
    assert_(len(log[0]) == 6, f"Expected 6 cols; got {len(log[0])}")

    for row in log:
        mem_in, mem_out = row[4], row[5]
        assert_(isinstance(mem_in, str) and mem_in.endswith("MB"), f"Bad mem_in: {mem_in}")
        assert_(isinstance(mem_out, str) and mem_out.endswith("MB"), f"Bad mem_out: {mem_out}")

    print("[PASS] memory rows:", len(log))


def test_4_filter_ops():
    section("TEST 4: filter_ops keeps only requested ops")
    m = SimpleCNN()
    x = torch.randn(2, 3, 32, 32)

    with DimViz(filter_ops=["convolution", "linear_proj"], show_summary=False) as dv:
        _ = m(x)

    log = dv.get_log()
    assert_(len(log) > 0, "Expected some logs with filter_ops")
    ops = set(r[1] for r in log)
    assert_(ops.issubset({"convolution", "linear_proj"}), f"Filter leaked ops: {ops}")
    print("[PASS] ops:", ops, "rows:", len(log))


def test_5_max_entries_cap():
    section("TEST 5: max_entries caps log length")
    m = SimpleCNN()
    x = torch.randn(4, 3, 32, 32)

    with DimViz(verbose=True, max_entries=5, show_summary=False) as dv:
        _ = m(x)

    log = dv.get_log()
    assert_(len(log) == 5, f"Expected 5 rows due to cap; got {len(log)}")
    print("[PASS] capped rows:", len(log))


def test_6_tuple_list_dict_outputs_no_crash():
    section("TEST 6: tuple/list/dict outputs no crash + shapes correct")
    x = torch.randn(2, 3, 4)

    with DimViz(verbose=False, show_summary=False) as dv1:
        out = TupleOut()(x)
    assert_(tuple(out[0].shape) == (2, 3, 4), "TupleOut[0] shape mismatch")
    assert_(tuple(out[1].shape) == (2, 4, 3), "TupleOut[1] shape mismatch")
    assert_(len(dv1.get_log()) > 0, "Expected logs for TupleOut")

    with DimViz(verbose=False, show_summary=False) as dv2:
        out = ListOut()(x)
    assert_(tuple(out[1].shape) == (2, 2, 4), "ListOut slice shape mismatch")
    assert_(tuple(out[2].shape) == (2, 3), "ListOut sum shape mismatch")
    assert_(len(dv2.get_log()) > 0, "Expected logs for ListOut")

    with DimViz(verbose=False, show_summary=False) as dv3:
        out = DictOut()(x)
    assert_(tuple(out["a"].shape) == (2, 3, 4), "DictOut[a] mismatch")
    assert_(tuple(out["b"].shape) == (2, 3), "DictOut[b] mismatch")
    assert_(len(dv3.get_log()) > 0, "Expected logs for DictOut")

    print("[PASS]")


def test_7_kwargs_tensor_discovery_where():
    section("TEST 7: tensor in kwargs (torch.where) gets traced")
    x = torch.randn(3, 4)
    cond = x > 0
    y = torch.randn(3, 4)

    with DimViz(verbose=False, show_summary=False) as dv:
        out = torch.where(cond, x, y)  # y is in args, but good kwargs test is below

    assert_(tuple(out.shape) == (3, 4), "where output mismatch")
    assert_(len(dv.get_log()) > 0, "Expected logs")

    # Now a kwargs tensor: use torch.where with kwargs
    with DimViz(verbose=False, show_summary=False) as dv2:
        out2 = torch.where(condition=cond, input=x, other=y)
    assert_(tuple(out2.shape) == (3, 4), "where kwargs output mismatch")
    assert_(len(dv2.get_log()) > 0, "Expected logs for kwargs version")
    print("[PASS]")


def test_8_transformerish_pipeline():
    section("TEST 8: transformer-ish pipeline correctness")
    m = Transformerish(d_model=32, n_heads=4)
    x = torch.randn(2, 7, 32)

    with DimViz(verbose=False, show_summary=False) as dv:
        y = m(x)

    assert_(tuple(y.shape) == (2, 7, 32), f"Expected (2,7,32), got {tuple(y.shape)}")
    assert_(len(dv.get_log()) > 0, "Expected logs for transformer-ish")
    print("[PASS] rows:", len(dv.get_log()))


def test_9_autograd_backward_works():
    section("TEST 9: autograd backward works under DimViz")
    lin = nn.Linear(16, 16)
    x = torch.randn(4, 16, requires_grad=True)

    with DimViz(verbose=False, show_summary=False) as dv:
        loss = (lin(x).tanh()).sum()

    loss.backward()
    assert_(x.grad is not None, "grad is None")
    assert_(torch.isfinite(x.grad).all().item(), "grad has NaN/Inf")
    print("[PASS] grad norm:", float(x.grad.norm().detach()))


def test_10_decorator_visualize():
    section("TEST 10: @visualize decorator works (no crash)")
    m = SimpleCNN()
    x = torch.randn(2, 3, 32, 32)

    @visualize(verbose=False, show_summary=False)
    def run(model, inp):
        return model(inp)

    y = run(m, x)
    assert_(tuple(y.shape) == (2, 10), "decorator output mismatch")
    print("[PASS]")


def test_11_export_log_json_csv_txt_and_validate():
    section("TEST 11: export_log JSON/CSV/TXT created + contents validated")
    out_dir = Path("dimviz_test_outputs")
    out_dir.mkdir(exist_ok=True)

    p_json = out_dir / "log.json"
    p_csv  = out_dir / "log.csv"
    p_txt  = out_dir / "log.txt"

    safe_rm(p_json); safe_rm(p_csv); safe_rm(p_txt)

    m = SimpleCNN()
    x = torch.randn(2, 3, 32, 32)

    with DimViz(verbose=False, show_summary=False) as dv:
        _ = m(x)

    log = dv.get_log()
    assert_(len(log) > 0, "Need log rows to export")

    export_log(log, str(p_json))
    export_log(log, str(p_csv))
    export_log(log, str(p_txt))

    assert_(p_json.exists(), "JSON not created")
    assert_(p_csv.exists(),  "CSV not created")
    assert_(p_txt.exists(),  "TXT not created")

    # Validate JSON structure
    data = json.loads(read_text(p_json))
    assert_("log" in data, "JSON missing 'log'")
    assert_(isinstance(data["log"], list) and len(data["log"]) > 0, "JSON log empty")
    assert_(set(data["log"][0].keys()).issuperset({"step","operation","input_shape","output_shape"}), "JSON keys missing")
    assert_("metadata" in data, "JSON missing metadata (expected by default)")

    # Validate CSV header
    with open(p_csv, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
    assert_(header[:4] == ["step","operation","input_shape","output_shape"], f"CSV header wrong: {header}")

    # Validate TXT contains header text
    txt = read_text(p_txt)
    assert_("DimViz Log Export" in txt, "TXT missing title")
    assert_("Step" in txt and "Operation" in txt, "TXT missing header columns")

    print("[PASS]")


def test_12_compare_logs_detects_differences():
    section("TEST 12: compare_logs detects op/shape differences")

    class ModelV1(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    class ModelV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 15)
            self.fc3 = nn.Linear(15, 5)
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    x = torch.randn(4, 10)

    with DimViz(verbose=False, show_summary=False) as dv1:
        _ = ModelV1()(x)
    with DimViz(verbose=False, show_summary=False) as dv2:
        _ = ModelV2()(x)

    diff = compare_logs(dv1.get_log(), dv2.get_log(), "v1", "v2")
    assert_("length_diff" in diff, "compare_logs missing length_diff")
    assert_("differences" in diff, "compare_logs missing differences")
    assert_(diff["length_diff"] != 0 or len(diff["differences"]) > 0, "Expected some differences")
    print("[PASS] length_diff:", diff["length_diff"], "differences:", len(diff["differences"]))


def test_13_crash_detection_mismatch_raises():
    section("TEST 13: intentional mismatch raises RuntimeError but DimViz doesn't swallow it")
    m = BuggyModel()
    x = torch.randn(2, 3, 32, 32)

    try:
        with DimViz(verbose=False, show_summary=False) as dv:
            _ = m(x)
        raise AssertionError("Expected RuntimeError, but forward succeeded.")
    except RuntimeError:
        print("[PASS] RuntimeError caught (expected)")


def test_14_inplace_ops_and_non_tensor_args_no_crash():
    section("TEST 14: inplace ops + non-tensor args don't crash")
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)

    with DimViz(verbose=False, show_summary=False) as dv:
        x.add_(y)          # inplace
        z = torch.clamp(x, min=0.0, max=1.0)  # non-tensor args
        _ = z + 3          # scalar arg

    assert_(tuple(z.shape) == (3, 4), "shape mismatch")
    print("[PASS]")


def test_15_nested_inputs_should_not_crash():
    section("TEST 15: weird nested inputs / kwargs (should not crash)")
    # Your tracker doesn't fully recurse, but should not crash.
    class Weird(nn.Module):
        def forward(self, x, pack):
            a = pack["a"][0]
            b = pack["b"]["inner"]
            return x + a + b

    x = torch.randn(2, 3)
    pack = {
        "a": [torch.randn(2, 3), "ignore", 123],
        "b": {"inner": torch.randn(2, 3), "other": [1,2,3]},
    }

    with DimViz(verbose=False, show_summary=False) as dv:
        y = Weird()(x, pack=pack)

    assert_(tuple(y.shape) == (2, 3), "nested output mismatch")
    print("[PASS]")


# ------------------ Runner ------------------

def main():
    print("Device: cpu")
    print("PyTorch:", torch.__version__)

    tests = [
        test_1_basic_e2e,
        test_2_verbose_false_logs_less,
        test_3_track_memory_columns_and_format,
        test_4_filter_ops,
        test_5_max_entries_cap,
        test_6_tuple_list_dict_outputs_no_crash,
        test_7_kwargs_tensor_discovery_where,
        test_8_transformerish_pipeline,
        test_9_autograd_backward_works,
        test_10_decorator_visualize,
        test_11_export_log_json_csv_txt_and_validate,
        test_12_compare_logs_detects_differences,
        test_13_crash_detection_mismatch_raises,
        test_14_inplace_ops_and_non_tensor_args_no_crash,
        test_15_nested_inputs_should_not_crash,
    ]

    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            print("\n[FAIL]", t.__name__)
            print(" ", repr(e))
            traceback.print_exc()

    section("RESULT")
    if failed == 0:
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print(f"❌ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
