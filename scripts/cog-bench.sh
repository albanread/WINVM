#!/bin/sh
# cog-bench.sh — run the micro-benchmark suite under Pharo/Cog and WINVM,
# same workloads, same x10-inner timing protocol, back to back on the same
# machine. The standing performance target is: FASTER THAN COG.
#
# Setup (once): download Pharo headless into E:/cog —
#   curl -L -o vm.zip    https://files.pharo.org/get-files/130/pharo-vm-Windows-x86_64-stable.zip
#   curl -L -o image.zip https://files.pharo.org/get-files/130/pharoImage-x86_64.zip
#   unzip vm.zip -d vm && unzip image.zip -d image
set -eu
cd "$(dirname "$0")/.."
COG=${COG_DIR:-E:/cog}
IMG=$(ls "$COG"/image/*.image 2>/dev/null | head -1)
[ -n "$IMG" ] || { echo "no Pharo at $COG — see setup comment"; exit 2; }
[ -x ./target/release/macvm.exe ] || { echo "build first: cargo build --release"; exit 2; }

# Richards + DeltaBlue are translated from world/41a on the fly, so the
# .mst stays the single source of truth. --assemble emits the complete
# Cog-side fileIn (harness + classes + macro drivers with checksums).
python scripts/mst2st.py "$COG/cog-all.st" --assemble >/dev/null

echo "=== COG ($(cat "$COG"/image/pharo.version 2>/dev/null || echo Pharo)) ==="
"$COG"/vm/PharoConsole.exe --headless "$IMG" st "$COG"/cog-all.st 2>&1 | grep -vE "sqMakeMemory|^\["

cat > /tmp/winvm-cog-bench.mst <<'MST'
Object subclass: Runner [
    Runner class >> show: nm block: b [
        | t |
        t := BenchmarkDashboard time: [ 10 timesRepeat: [ b value ] ] reps: 7.
        Transcript showCr: nm, ' cold=', (t at: 1) printString, ' warm=', (t at: 2) printString
    ]
]
Runner show: 'arith    ' block: [ BenchmarkDashboard benchArith ].
Runner show: 'fib      ' block: [ BenchmarkDashboard benchFib ].
Runner show: 'sieve    ' block: [ BenchmarkDashboard benchSieve ].
Runner show: 'dict     ' block: [ BenchmarkDashboard benchDict ].
Runner show: 'alloc    ' block: [ BenchmarkDashboard benchAlloc ].
Runner show: 'richards ' block: [ BenchmarkDashboard benchRichards ].
Runner show: 'deltablue' block: [ BenchmarkDashboard benchDeltaBlue ].
MST
echo "=== WINVM threshold=20 ==="
MACVM_JIT=threshold=20 ./target/release/macvm.exe run /tmp/winvm-cog-bench.mst --world world 2>&1 | tail -7
