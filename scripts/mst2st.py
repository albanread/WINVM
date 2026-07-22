"""Translate world/41a_bench_workloads.mst (the .mst class dialect) into
Pharo fileIn chunk format. The dialect surface, verified by inspection:

    Super subclass: Name [
        | ivar1 ivar2 |                 (optional)
        <classVars: A B C>              (optional)
        sel [ body ]                    (instance method)
        Name class >> sel [ body ]      (class method)
    ]

Method bodies are standard Smalltalk and carry no `!`, so they can be
emitted into chunks verbatim.
"""
import io
import re
import sys

SRC = "world/41a_bench_workloads.mst"
DST = sys.argv[1] if len(sys.argv) > 1 else "E:/cog/richards_deltablue.st"

text = io.open(SRC, encoding="utf-8").read()


def skip_atom(s, i):
    """Advance past a string, comment, or char literal starting at i."""
    c = s[i]
    if c == "'":
        i += 1
        while i < len(s):
            if s[i] == "'":
                if i + 1 < len(s) and s[i + 1] == "'":
                    i += 2
                    continue
                return i + 1
            i += 1
    elif c == '"':
        i += 1
        while i < len(s) and s[i] != '"':
            i += 1
        return i + 1
    elif c == "$":
        return i + 2
    return i + 1


def find_matching(s, i):
    """i points at '['; return index just past its matching ']'."""
    depth = 0
    while i < len(s):
        c = s[i]
        if c in "'\"$":
            i = skip_atom(s, i)
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced brackets")


classes = []  # (super, name, ivars, classvars, [(is_class_side, selector, body)])

for m in re.finditer(r"(\w+) subclass: (\w+) \[", text):
    sup, name = m.group(1), m.group(2)
    body_start = m.end() - 1
    body_end = find_matching(text, body_start)
    body = text[m.end():body_end - 1]

    ivars = []
    classvars = []
    iv = re.match(r"\s*\|([^|]*)\|", body)
    if iv:
        ivars = iv.group(1).split()
        body = body[iv.end():]
    cv = re.search(r"<classVars:([^>]*)>", body)
    if cv:
        classvars = cv.group(1).split()
        body = body[:cv.start()] + body[cv.end():]

    methods = []
    i = 0
    while True:
        b = body.find("[", i)
        if b == -1:
            break
        # Header is everything from the previous method's end to this '['.
        header = body[:b] if not methods else body[i:b]
        # Strip comments from the header text before parsing the selector.
        header = re.sub(r'"[^"]*"', " ", header).strip()
        if not header:
            i = b + 1
            continue
        end = find_matching(body, b)
        mbody = body[b + 1:end - 1]
        is_class = False
        if header.startswith(name + " class >>"):
            is_class = True
            header = header[len(name + " class >>"):].strip()
        methods.append((is_class, header, mbody.strip("\n")))
        i = end
    classes.append((sup, name, ivars, classvars, methods))

out = []
for sup, name, ivars, classvars, methods in classes:
    out.append(
        f"{sup} subclass: #{name}\n"
        f"\tinstanceVariableNames: '{' '.join(ivars)}'\n"
        f"\tclassVariableNames: '{' '.join(classvars)}'\n"
        f"\tpackage: 'CogBench'!\n"
    )
for sup, name, ivars, classvars, methods in classes:
    for side in (False, True):
        ms = [(sel, b) for is_c, sel, b in methods if is_c == side]
        if not ms:
            continue
        target = f"{name} class" if side else name
        out.append(f"!{target} methodsFor: 'bench'!")
        for sel, b in ms:
            out.append(f"{sel}\n{b}!")
        out.append(" !\n")

result = "\n".join(out) + "\n"

if "--assemble" in sys.argv:
    # Emit the COMPLETE Cog-side fileIn: the micro harness (its doits
    # stripped), the translated classes (Pharo's own `Variable` renamed
    # DBVariable -- a system class in the Slot hierarchy; redefining it
    # would corrupt the image), and the macro drivers asserting the same
    # checksums the WINVM dashboard asserts.
    bench = io.open("scripts/cog-bench.st", encoding="utf-8").read()
    bench = bench.replace("CogBench runAll.!", "").replace("Smalltalk exitSuccess.!", "")
    classes_txt = re.sub(r"\bVariable\b", "DBVariable", result)
    macro = """
Strength initStrengths.!
!CogBench class methodsFor: 'bench'!
benchRichards
	| r |
	RichardsBenchmark setUp.
	r := RichardsBenchmark runOne.
	(RichardsBenchmark checkResult: r) ifFalse: [ ^self error: 'richards: wrong result' ].
	^r!
benchDeltaBlue
	| r |
	Strength initStrengths.
	DeltaBlue setUp.
	r := DeltaBlue runOne.
	(DeltaBlue checkResult: r) ifFalse: [ ^self error: 'deltablue: wrong result' ].
	^r!
runEverything
	self runAll.
	self run: 'richards ' block: [ self benchRichards ] check: 2324609297.
	self run: 'deltablue' block: [ self benchDeltaBlue ] check: 224874.! !

CogBench runEverything.!
Smalltalk exitSuccess.!
"""
    result = bench + "\n" + classes_txt + macro

io.open(DST, "w", encoding="utf-8", newline="\n").write(result)
print(f"{len(classes)} classes ->", DST)
for c in classes:
    n_i = sum(1 for x in c[4] if not x[0])
    n_c = sum(1 for x in c[4] if x[0])
    print(f"  {c[1]:24} super={c[0]:16} iv={len(c[2])} cv={len(c[3])} m={n_i}+{n_c}")
