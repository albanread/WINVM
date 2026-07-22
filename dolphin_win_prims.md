# Dolphin Smalltalk — VM primitive inventory (for the WINVM port)

Every `<primitive: N>` the Dolphin image source declares, cross-referenced
against the VM's own authoritative dispatch table
(`Core/DolphinVM/PrimitivesTable.cpp`, a 256-entry `primitivesTable`). To run
Dolphin's MVP framework on WINVM we must implement and test each of these.

## Summary

- **215 distinct primitives** are actually declared as pragmas in the
  image source, across **424 call sites** (`.cls` files, whole fork).
- The VM defines 256 slots; the rest are `unusedPrimitive` (always fail) or are
  used only via generated code (FFI/COM — see below).
- Source of truth for *what each primitive does*: the C++ implementation named
  in `PrimitivesTable.cpp`; the canonical Smalltalk selector(s) are that table's
  own case comments, reproduced here.

### Things the porter must know before starting

- **Primitives 31 and 32 are declared but map to `unusedPrimitive` in the VM**
  (`LargeInteger>>\\` and `LargeInteger>>//`). They *always fail* and fall through
  to the Smalltalk fallback — so on WINVM they need only fail cleanly, not compute.
- **FFI/COM primitives (48, 80, 96) show 0 pragma sites** because they are not
  written as `<primitive: N>` in method source — the external-method compiler
  emits them from `<stdcall:…>` / `<cdecl:…>` / `<virtual:…>` descriptors. They are
  nonetheless essential (every Win32 call routes through 96, every COM/vtable call
  through 80). Listed in the appendix.
- **Primitive 157 (`primitiveNewInitializedObject`) is by far the hottest**
  (73 sites) — the `x:y:`-style bulk instance initialiser. MVP leans on it heavily.
- **Primitive 1 (`^self`) and 2 (`^false`) are trivial** but ubiquitous; they are
  return-shortcut primitives, not real VM work.

## Coverage worksheet by category

Tick `Impl` / `Test` as each lands on WINVM. `Uses` = pragma call sites in the image.

### Return / inst-var / activation  (5 primitives, 22 sites)

| # | VM impl (`Interpreter::…`) | Smalltalk selector(s) | Uses | Key classes | Impl | Test |
|--:|---|---|--:|---|:--:|:--:|
| 1 | `primitiveReturnSelf` | ^self | 10 | Refactory.Browser.Tests.ParseTreeRewriterTest×9, Refactory.Browser.Tests.RBSourceFormatterTest | ☐ | ☐ |
| 2 | `primitiveReturnConst&lt;2&gt;` | ^false | 9 | Refactory.Browser.Tests.ParseTreeRewriterTest×9 | ☐ | ☐ |
| 68 | `primitiveReturnFromInterrupt` | ProcessorScheduler&gt;&gt;iret:list: | 1 | Kernel.ProcessorScheduler | ☐ | ☐ |
| 78 | `primitiveReturn` | ProcessorScheduler&gt;&gt;#returnValue:toFrame: | 1 | Kernel.ProcessorScheduler | ☐ | ☐ |
| 104 | `primitiveReturnFromCallback` | ProcessorScheduler&gt;&gt;#primReturn:callback: | 1 | Kernel.ProcessorScheduler | ☐ | ☐ |

### SmallInteger arithmetic & bits  (23 primitives, 30 sites)

| # | VM impl (`Interpreter::…`) | Smalltalk selector(s) | Uses | Key classes | Impl | Test |
|--:|---|---|--:|---|:--:|:--:|
| 9 | `primitiveMultiply` | SmallInteger&gt;&gt;#* | 1 | Core.SmallInteger | ☐ | ☐ |
| 10 | `primitiveDivide` | SmallInteger&gt;&gt;#/ | 2 | Core.SmallInteger, Kernel.Tests.ParserErrorTest | ☐ | ☐ |
| 11 | `primitiveMod` | SmallInteger&gt;&gt;#'\\' | 1 | Core.SmallInteger | ☐ | ☐ |
| 12 | `primitiveDiv` | SmallInteger&gt;&gt;#// | 1 | Core.SmallInteger | ☐ | ☐ |
| 13 | `primitiveQuo` | SmallInteger&gt;&gt;#quo: | 1 | Core.SmallInteger | ☐ | ☐ |
| 14 | `primitiveSubtract` | SmallInteger&gt;&gt;#- | 1 | Core.SmallInteger | ☐ | ☐ |
| 15 | `primitiveAdd` | SmallInteger&gt;&gt;#+ | 1 | Core.SmallInteger | ☐ | ☐ |
| 16 | `primitiveEqual` | SmallInteger&gt;&gt;#= | 1 | Core.SmallInteger | ☐ | ☐ |
| 17 | `primitiveIntegerCmp&lt;std::greater_equal&lt;SmallInteger&gt;, true&gt;` | SmallInteger&gt;&gt;#&gt;= | 1 | Core.SmallInteger | ☐ | ☐ |
| 18 | `primitiveIntegerCmp&lt;std::less&lt;SmallInteger&gt;, false&gt;` | SmallInteger&gt;&gt;#&lt; | 1 | Core.SmallInteger | ☐ | ☐ |
| 19 | `primitiveIntegerCmp&lt;std::greater&lt;SmallInteger&gt;, true&gt;` | SmallInteger&gt;&gt;#&gt; | 1 | Core.SmallInteger | ☐ | ☐ |
| 20 | `primitiveIntegerCmp&lt;std::less_equal&lt;SmallInteger&gt;, false&gt;` | SmallInteger&gt;&gt;#&lt;= | 1 | Core.SmallInteger | ☐ | ☐ |
| 40 | `primitiveIntegerOp&lt;std::bit_and&lt;Oop&gt;&gt;` | SmallInteger&gt;&gt;#bitAnd: | 2 | Core.SmallInteger×2 | ☐ | ☐ |
| 41 | `primitiveIntegerOp&lt;std::bit_or&lt;Oop&gt;&gt;` | SmallInteger&gt;&gt;#bitOr: | 2 | Core.SmallInteger×2 | ☐ | ☐ |
| 42 | `primitiveIntegerOp&lt;bit_xor&gt;` | SmallInteger&gt;&gt;#bitXor: | 1 | Core.SmallInteger | ☐ | ☐ |
| 43 | `primitiveBitShift` | SmallInteger&gt;&gt;#bitShift: | 2 | Core.SmallInteger×2 | ☐ | ☐ |
| 44 | `primitiveSmallIntegerPrintString` | SmallInteger&gt;&gt;#printString | 1 | Core.SmallInteger | ☐ | ☐ |
| 54 | `primitiveHighBit` | SmallInteger&gt;&gt;#highBit | 1 | Core.SmallInteger | ☐ | ☐ |
| 103 | `primitiveAddressOf` | Object&gt;&gt;#yourAddress | 2 | Core.Object×2 | ☐ | ☐ |
| 109 | `primitiveHashMultiply` | Integer&gt;&gt;#hashMultiply, SmallInteger&gt;&gt;#identityHash, SmallInteger&gt;&gt;#hash | 3 | Core.SmallInteger×2, Core.Integer | ☐ | ☐ |
| 145 | `primitiveAnyMask` | SmallInteger&gt;&gt;#anyMask: | 1 | Core.SmallInteger | ☐ | ☐ |
| 146 | `primitiveAllMask` | SmallInteger&gt;&gt;#allMask: | 1 | Core.SmallInteger | ☐ | ☐ |
| 152 | `primitiveLowBit` | SmallInteger&gt;&gt;#lowBit | 1 | Core.SmallInteger | ☐ | ☐ |

### LargeInteger  (21 primitives, 24 sites)

| # | VM impl (`Interpreter::…`) | Smalltalk selector(s) | Uses | Key classes | Impl | Test |
|--:|---|---|--:|---|:--:|:--:|
| 21 | `primitiveLargeIntegerOpR&lt;Li::Add, Li::AddSingle&gt;` | LargeInteger&gt;&gt;#+ | 2 | Core.LargeInteger×2 | ☐ | ☐ |
| 22 | `primitiveLargeIntegerOpR&lt;Li::Sub, Li::SubSingle&gt;` | LargeInteger&gt;&gt;#- | 1 | Core.LargeInteger | ☐ | ☐ |
| 23 | `primitiveLargeIntegerCmp&lt;true,false&gt;` | LargeInteger&gt;&gt;#&lt; | 1 | Core.LargeInteger | ☐ | ☐ |
| 24 | `primitiveLargeIntegerCmp&lt;false,false&gt;` | LargeInteger&gt;&gt;#&gt; | 2 | Core.LargeInteger×2 | ☐ | ☐ |
| 25 | `primitiveLargeIntegerCmp&lt;true, true&gt;` | LargeInteger&gt;&gt;#&lt;= | 1 | Core.LargeInteger | ☐ | ☐ |
| 26 | `primitiveLargeIntegerCmp&lt;false,true&gt;` | LargeInteger&gt;&gt;#&gt;= | 1 | Core.LargeInteger | ☐ | ☐ |
| 27 | `primitiveLargeIntegerEqual` | LargeInteger&gt;&gt;#= | 1 | Core.LargeInteger | ☐ | ☐ |
| 28 | `primitiveLargeIntegerUnaryOp&lt;Li::Normalize&gt;` | LargeInteger&gt;&gt;normalize | 1 | Core.LargeInteger | ☐ | ☐ |
| 29 | `primitiveLargeIntegerOpZ&lt;Li::Mul, Li::MulSingle&gt;` | LargeInteger&gt;&gt;#* | 2 | Core.LargeInteger×2 | ☐ | ☐ |
| 30 | `primitiveLargeIntegerDivide` | LargeInteger&gt;&gt;#/ | 1 | Core.LargeInteger | ☐ | ☐ |
| 31 | `unusedPrimitive`  **⚠ unusedPrimitive (always fails)** | LargeInteger#\\ | 1 | Core.LargeInteger | ☐ | ☐ |
| 32 | `unusedPrimitive`  **⚠ unusedPrimitive (always fails)** | LargeInteger&gt;&gt;#// | 1 | Core.LargeInteger | ☐ | ☐ |
| 33 | `primitiveLargeIntegerQuo` | LargeInteger&gt;&gt;#quo: | 1 | Core.LargeInteger | ☐ | ☐ |
| 34 | `primitiveLargeIntegerOpZ&lt;Li::BitAnd, Li::BitAndSingle&gt;` | LargeInteger&gt;&gt;#bitAnd: | 1 | Core.LargeInteger | ☐ | ☐ |
| 35 | `primitiveLargeIntegerOpR&lt;Li::BitOr, Li::BitOrSingle&gt;` | LargeInteger&gt;&gt;#bitOr: | 1 | Core.LargeInteger | ☐ | ☐ |
| 36 | `primitiveLargeIntegerOpR&lt;Li::BitXor, Li::BitXorSingle&gt;` | LargeInteger&gt;&gt;#bitXor: | 1 | Core.LargeInteger | ☐ | ☐ |
| 37 | `primitiveLargeIntegerBitShift` | LargeInteger&gt;&gt;#bitShift | 1 | Core.LargeInteger | ☐ | ☐ |
| 38 | `primitiveLargeIntegerBitInvert` | LargeInteger&gt;&gt;#bitInvert | 1 | Core.LargeInteger | ☐ | ☐ |
| 39 | `primitiveLargeIntegerUnaryOp&lt;Li::Negate&gt;` | LargeInteger&gt;&gt;#negate | 1 | Core.LargeInteger | ☐ | ☐ |
| 53 | `primitiveLargeIntegerHighBit` | LargeInteger&gt;&gt;#highBit | 1 | Core.LargeInteger | ☐ | ☐ |
| 167 | `primitiveLargeIntegerAsFloat` | LargeInteger&gt;&gt;#asFloat | 1 | Core.LargeInteger | ☐ | ☐ |

### Float  (36 primitives, 36 sites)

| # | VM impl (`Interpreter::…`) | Smalltalk selector(s) | Uses | Key classes | Impl | Test |
|--:|---|---|--:|---|:--:|:--:|
| 45 | `primitiveFloatCompare&lt;std::greater&lt;double&gt;&gt;` | Float&gt;&gt;#&gt; | 1 | Core.Float | ☐ | ☐ |
| 46 | `primitiveFloatCompare&lt;std::greater_equal&lt;double&gt;&gt;` | Float&gt;&gt;#&gt;= | 1 | Core.Float | ☐ | ☐ |
| 47 | `primitiveFloatCompare&lt;std::equal_to&lt;double&gt;&gt;` | Float&gt;&gt;#= | 1 | Core.Float | ☐ | ☐ |
| 128 | `primitiveFloatAtOffset&lt;double&gt;` | ByteArray&gt;&gt;#doubleAtOffset: | 1 | Core.ByteArray | ☐ | ☐ |
| 129 | `primitiveFloatAtOffsetPut&lt;double&gt;` | ByteArray&gt;&gt;#doubleAtOffset:put: | 1 | Core.ByteArray | ☐ | ☐ |
| 130 | `primitiveFloatAtOffset&lt;float&gt;` | ByteArray&gt;&gt;#floatAtOffset: | 1 | Core.ByteArray | ☐ | ☐ |
| 131 | `primitiveFloatAtOffsetPut&lt;float&gt;` | ByteArray&gt;&gt;#floatAtOffset:put: | 1 | Core.ByteArray | ☐ | ☐ |
| 160 | `primitiveFloatBinaryOp&lt;std::plus&lt;double&gt;&gt;` | Float&gt;&gt;#+ | 1 | Core.Float | ☐ | ☐ |
| 161 | `primitiveFloatBinaryOp&lt;std::minus&lt;double&gt;&gt;` | Float&gt;&gt;#- | 1 | Core.Float | ☐ | ☐ |
| 162 | `primitiveFloatCompare&lt;std::less&lt;double&gt;&gt;` | Float&gt;&gt;#&lt; | 1 | Core.Float | ☐ | ☐ |
| 164 | `primitiveFloatBinaryOp&lt;std::multiplies&lt;double&gt;&gt;` | Float&gt;&gt;#* | 1 | Core.Float | ☐ | ☐ |
| 165 | `primitiveFloatBinaryOp&lt;std::divides&lt;double&gt;&gt;` | Float&gt;&gt;#/ | 1 | Core.Float | ☐ | ☐ |
| 166 | `primitiveFloatTruncationOp&lt;Truncate&gt;` | Float&gt;&gt;#truncated | 1 | Core.Float | ☐ | ☐ |
| 168 | `primitiveAsFloat` | SmallInteger&gt;&gt;#asFloat | 1 | Core.SmallInteger | ☐ | ☐ |
| 193 | `primitiveFloatUnaryOp&lt;Sin&gt;` | Float&gt;&gt;#sin | 1 | Core.Float | ☐ | ☐ |
| 194 | `primitiveFloatUnaryOp&lt;Tan&gt;` | Float&gt;&gt;#tan | 1 | Core.Float | ☐ | ☐ |
| 195 | `primitiveFloatUnaryOp&lt;Cos&gt;` | Float&gt;&gt;#cos | 1 | Core.Float | ☐ | ☐ |
| 196 | `primitiveFloatUnaryOp&lt;ArcSin&gt;` | Float&gt;&gt;#arcSin | 1 | Core.Float | ☐ | ☐ |
| 197 | `primitiveFloatUnaryOp&lt;ArcTan&gt;` | Float&gt;&gt;#arcTan | 1 | Core.Float | ☐ | ☐ |
| 198 | `primitiveFloatUnaryOp&lt;ArcCos&gt;` | Float&gt;&gt;#arcCos | 1 | Core.Float | ☐ | ☐ |
| 199 | `primitiveFloatBinaryOp&lt;Atan2&gt;` | Float&gt;&gt;#acTan: | 1 | Core.Float | ☐ | ☐ |
| 200 | `primitiveFloatUnaryOp&lt;Log&gt;` | Float&gt;&gt;#ln | 1 | Core.Float | ☐ | ☐ |
| 201 | `primitiveFloatUnaryOp&lt;Exp&gt;` | Float&gt;&gt;#exp | 1 | Core.Float | ☐ | ☐ |
| 202 | `primitiveFloatUnaryOp&lt;Sqrt&gt;` | Float&gt;&gt;#sqrt | 1 | Core.Float | ☐ | ☐ |
| 203 | `primitiveFloatUnaryOp&lt;Log10&gt;` | Float&gt;&gt;#log | 1 | Core.Float | ☐ | ☐ |
| 204 | `primitiveFloatTimesTwoPower` | Float&gt;&gt;#timesTwoPower: | 1 | Core.Float | ☐ | ☐ |
| 205 | `primitiveFloatUnaryOp&lt;Abs&gt;` | Float&gt;&gt;#abs | 1 | Core.Float | ☐ | ☐ |
| 206 | `primitiveFloatBinaryOp&lt;Pow&gt;` | Float&gt;&gt;#raisedTo: | 1 | Core.Float | ☐ | ☐ |
| 207 | `primitiveFloatTruncationOp&lt;Floor&gt;` | Float&gt;&gt;#floor | 1 | Core.Float | ☐ | ☐ |
| 208 | `primitiveFloatTruncationOp&lt;Ceiling&gt;` | Float&gt;&gt;#ceiling | 1 | Core.Float | ☐ | ☐ |
| 209 | `primitiveFloatExponent` | Float&gt;&gt;#exponent | 1 | Core.Float | ☐ | ☐ |
| 210 | `primitiveFloatUnaryOp&lt;Negated&gt;` | Float&gt;&gt;#negated | 1 | Core.Float | ☐ | ☐ |
| 211 | `primitiveFloatClassify` | Float&gt;&gt;#fpClass | 1 | Core.Float | ☐ | ☐ |
| 212 | `primitiveFloatUnaryOp&lt;FractionPart&gt;` | Float&gt;&gt;#fractionPart | 1 | Core.Float | ☐ | ☐ |
| 213 | `primitiveFloatUnaryOp&lt;IntegerPart&gt;` | Float&gt;&gt;#integerPart | 1 | Core.Float | ☐ | ☐ |
| 214 | `primitiveFloatCompare&lt;std::less_equal&lt;double&gt;&gt;` | Float&gt;&gt;#&lt;= | 1 | Core.Float | ☐ | ☐ |

### String / Character  (21 primitives, 37 sites)

| # | VM impl (`Interpreter::…`) | Smalltalk selector(s) | Uses | Key classes | Impl | Test |
|--:|---|---|--:|---|:--:|:--:|
| 51 | `primitiveStringComparison&lt;CmpA,CmpW&gt;` | String&gt;&gt;#&lt;==&gt; | 1 | Core.String | ☐ | ☐ |
| 52 | `primitiveStringNextIndexOfFromTo` | String&gt;&gt;#nextIdentityIndexOf:from:to: | 2 | Core.String×2 | ☐ | ☐ |
| 55 | `primitiveBytesEqual` | ByteArray&gt;&gt;#= | 1 | Core.ByteArray | ☐ | ☐ |
| 56 | `primitiveStringComparison&lt;CmpIA,CmpIW&gt;` | String&gt;&gt;#&lt;=&gt; | 1 | Core.String | ☐ | ☐ |
| 59 | `primitiveNextIndexOfFromTo` | Object&gt;&gt;#basicIdentityIndexOf:from:to:, ArrayedCollection&gt;&gt;#nextIdentityIndexOf:from:to: | 3 | Core.ArrayedCollection, Core.Object, Core.OrderedCollection | ☐ | ☐ |
| 63 | `primitiveStringAt` | String&gt;&gt;#at: | 5 | Core.String×3, Core.AnsiString, Core.Utf16String | ☐ | ☐ |
| 64 | `primitiveStringAtPut` | String&gt;&gt;#at:put: | 1 | Core.String | ☐ | ☐ |
| 106 | `primitiveHashBytes` | ByteArray&gt;&gt;#hash, String&gt;&gt;#hash, LargeInteger&gt;&gt;#hash | 4 | Core.ByteArray, Core.GUID, Core.LargeInteger, Core.String | ☐ | ☐ |
| 149 | `primitiveStringSearch` | String&gt;&gt;#findString:startingAt: | 3 | Core.String×2, Core.ByteArray | ☐ | ☐ |
| 215 | `primitiveStringAsUtf16String` | String&gt;&gt;#asUtf16String | 2 | Core.String×2 | ☐ | ☐ |
| 216 | `primitiveStringAsUtf8String` | String&gt;&gt;#asUtf8String | 3 | Core.String×2, Core.Utf16String | ☐ | ☐ |
| 217 | `primitiveStringAsByteString` | String&gt;&gt;#asAnsiString | 1 | Core.String | ☐ | ☐ |
| 218 | `primitiveStringConcatenate` | String&gt;&gt;#, | 1 | Core.String | ☐ | ☐ |
| 219 | `primitiveStringOrdinalEqual` | String&gt;&gt;#= | 1 | Core.String | ☐ | ☐ |
| 220 | `primitiveStringCompareOrdinal` | String&gt;&gt;#compareOrdinals:ignoringCase: | 1 | Core.String | ☐ | ☐ |
| 224 | `primitiveBeginsWith` | String&gt;&gt;#beginsWith: | 1 | Core.String | ☐ | ☐ |
| 225 | `primitiveStringDecodeAt` | Utf8String&gt;&gt;decodeAt: | 1 | Core.UtfEncodedString | ☐ | ☐ |
| 226 | `primitiveStringEncodedSizeAt` | Utf8String&gt;&gt;encodedSizeAt: | 2 | Core.Utf16String, Core.Utf8String | ☐ | ☐ |
| 227 | `primitiveOrdinalHashIgnoreCase` | String&gt;&gt;#hashOrdinalsIgnoringCase: | 1 | Core.String | ☐ | ☐ |
| 228 | `primitiveStringLessOrEqual` | String&gt;&gt;#&lt;= | 1 | Core.String | ☐ | ☐ |
| 229 | `primitiveCharacterEquals` | Character&gt;&gt;#= | 1 | Core.Character | ☐ | ☐ |

### ByteArray / ExternalAddress accessors  (33 primitives, 48 sites)

| # | VM impl (`Interpreter::…`) | Smalltalk selector(s) | Uses | Key classes | Impl | Test |
|--:|---|---|--:|---|:--:|:--:|
| 120 | `primitiveIntegerAtOffset&lt;uint32_t, StoreUnsigned32&gt;` | ByteArray&gt;&gt;#dwordAtOffset: | 4 | Core.ByteArray×3, Core.LargeInteger | ☐ | ☐ |
| 121 | `primitiveUint32AtPut` | ByteArray&gt;&gt;#dwordAtOffset:put: | 4 | Core.ByteArray×3, Core.LargeInteger | ☐ | ☐ |
| 122 | `primitiveIntegerAtOffset&lt;int32_t, StoreSigned32&gt;` | ByteArray&gt;&gt;#sdwordAtOffset: | 2 | Core.ByteArray, Core.LargeInteger | ☐ | ☐ |
| 123 | `primitiveInt32AtPut` | ByteArray&gt;&gt;#sdwordAtOffset:put: | 1 | Core.ByteArray | ☐ | ☐ |
| 124 | `primitiveIntegerAtOffset&lt;uint16_t, StoreSmallInteger&gt;` | ByteArray&gt;&gt;#wordAtOffset: | 2 | Core.ByteArray, Core.Utf16String | ☐ | ☐ |
| 125 | `primitiveAtOffsetPutInteger&lt;uint16_t, 0x0, 0xffff&gt;` | ByteArray&gt;&gt;#wordAtOffset:put: | 2 | Core.ByteArray, Core.Utf16String | ☐ | ☐ |
| 126 | `primitiveIntegerAtOffset&lt;int16_t, StoreSmallInteger&gt;` | ByteArray&gt;&gt;#swordAtoffset: | 1 | Core.ByteArray | ☐ | ☐ |
| 127 | `primitiveAtOffsetPutInteger&lt;uint16_t, -0x8000, 0x7fff&gt;` | ByteArray&gt;&gt;#swordAtOffset:put: | 1 | Core.ByteArray | ☐ | ☐ |
| 132 | `primitiveIndirectIntegerAtOffset&lt;uint8_t, StoreSmallInteger&gt;` | External.Address&gt;&gt;#byteAtOffset: | 1 | External.Address | ☐ | ☐ |
| 133 | `primitiveIndirectAtOffsetPutInteger&lt;uint8_t, 0, 255&gt;` | External.Address&gt;&gt;#byteAtOffset:put: | 1 | External.Address | ☐ | ☐ |
| 134 | `primitiveIndirectIntegerAtOffset&lt;uint32_t, StoreUnsigned32&gt;` | External.Address&gt;&gt;#dwordAtOffset: | 1 | External.Address | ☐ | ☐ |
| 135 | `primitiveIndirectUint32AtPut` | External.Address&gt;&gt;#dwordAtOffset:put: | 1 | External.Address | ☐ | ☐ |
| 136 | `primitiveIndirectIntegerAtOffset&lt;int32_t, StoreSigned32&gt;` | External.Address&gt;&gt;#sdwordAtOffset: | 1 | External.Address | ☐ | ☐ |
| 137 | `primitiveIndirectInt32AtPut` | External.Address&gt;&gt;#sdwordAtOffset:put: | 1 | External.Address | ☐ | ☐ |
| 138 | `primitiveIndirectIntegerAtOffset&lt;uint16_t, StoreSmallInteger&gt;` | External.Address&gt;&gt;#wordAtOffset: | 1 | External.Address | ☐ | ☐ |
| 139 | `primitiveIndirectAtOffsetPutInteger&lt;uint16_t, 0, 0xffff&gt;` | External.Address&gt;&gt;#wordAtOffset:put: | 1 | External.Address | ☐ | ☐ |
| 140 | `primitiveIndirectIntegerAtOffset&lt;int16_t, StoreSmallInteger&gt;` | External.Address&gt;&gt;#swordAtOffset: | 1 | External.Address | ☐ | ☐ |
| 141 | `primitiveIndirectAtOffsetPutInteger&lt;int16_t, -0x8000, 0x7fff&gt;` | External.Address&gt;&gt;#swordAtOffset:put: | 1 | External.Address | ☐ | ☐ |
| 142 | `primitiveReplaceBytes` | ByteArray\|String&gt;&gt;#replaceBytesOf:from:to:startingAt: | 4 | Core.ByteArray, Core.GUID, Core.LargeInteger, Core.String | ☐ | ☐ |
| 143 | `primitiveIndirectReplaceBytes` | ExternalAddress&gt;&gt;#replaceBytesOf:from:to:startingAt: | 1 | External.Address | ☐ | ☐ |
| 158 | `primitiveSmallIntegerAt` | SmallInteger&gt;&gt;#byteAt: | 1 | Core.SmallInteger | ☐ | ☐ |
| 159 | `primitiveLongDoubleAt` | ByteArray&gt;&gt;#longDoubleAtOffset: | 1 | Core.ByteArray | ☐ | ☐ |
| 176 | `primitiveStackAtPut` | Process&gt;&gt;#at:put:, Process&gt;&gt;#basicAt:put: | 2 | Core.Process×2 | ☐ | ☐ |
| 180 | `primitiveIntegerAtOffset&lt;uintptr_t, StoreUIntPtr&gt;` | ByteArray&gt;&gt;#uintPtrAtOffset: | 1 | Core.ByteArray | ☐ | ☐ |
| 181 | `primitiveUintPtrAtPut` | ByteArray&gt;&gt;#uintPtrAtOffset:put:   (was primitiveUint32AtPut — truncated on x64) | 1 | Core.ByteArray | ☐ | ☐ |
| 182 | `primitiveIntegerAtOffset&lt;intptr_t, StoreIntPtr&gt;` | ByteArray&gt;&gt;#intptrAtOffset: | 1 | Core.ByteArray | ☐ | ☐ |
| 183 | `primitiveIntPtrAtPut` | ByteArray&gt;&gt;#intPtrAtOffset:put:    (was primitiveInt32AtPut — truncated on x64) | 2 | Core.ByteArray, Core.LargeInteger | ☐ | ☐ |
| 184 | `primitiveIndirectIntegerAtOffset&lt;uintptr_t, StoreUIntPtr&gt;` | External.Address&gt;&gt;#uintPtrAtOffset: | 1 | External.Address | ☐ | ☐ |
| 185 | `primitiveIndirectUintPtrAtPut` | External.Address&gt;&gt;#uintPtrAtOffset:put: (was primitiveIndirectUint32AtPut — truncated on x64) | 1 | External.Address | ☐ | ☐ |
| 186 | `primitiveIndirectIntegerAtOffset&lt;intptr_t, StoreIntPtr&gt;` | External.Address&gt;&gt;#intptrAtOffset: | 1 | External.Address | ☐ | ☐ |
| 187 | `primitiveIndirectIntPtrAtPut` | External.Address&gt;&gt;#intPtrAtOffset:put:  (was primitiveIndirectInt32AtPut — truncated on x64) | 1 | External.Address | ☐ | ☐ |
| 191 | `primitiveIntegerAtOffset&lt;uint64_t, StoreUnsigned64&gt;` | ByteArray&gt;&gt;#qwordAtOffset: | 2 | Core.ByteArray, Core.Float | ☐ | ☐ |
| 192 | `primitiveIntegerAtOffset&lt;int64_t, StoreSigned64&gt;` | ByteArray&gt;&gt;#sqwordAtOffset: | 1 | Core.ByteArray | ☐ | ☐ |

### Object / Behavior (new, at:, class, become, copy, hash)  (29 primitives, 148 sites)

| # | VM impl (`Interpreter::…`) | Smalltalk selector(s) | Uses | Key classes | Impl | Test |
|--:|---|---|--:|---|:--:|:--:|
| 50 | `primitiveCopyFromTo` | ArrayedCollection&gt;&gt;#copyFrom:to: | 1 | Core.ArrayedCollection | ☐ | ☐ |
| 57 | `primitiveIsKindOf` | Object&gt;&gt;#isKindOf: | 2 | Core.Object, Kernel.ProtoObject | ☐ | ☐ |
| 58 | `primitiveAllSubinstances` | Behavior&gt;&gt;#primAllSubinstances | 1 | Core.Behavior | ☐ | ☐ |
| 60 | `primitiveBasicAt` | Object&gt;&gt;#at:, Object&gt;&gt;#basicAt:, String&gt;&gt;#byteAt:, ArrayedCollection&gt;&gt;#at:, ArrayedCollection&gt;&gt;#lookup: | 9 | Core.ArrayedCollection×4, Core.Object×2, Core.ByteArray, Core.LargeInteger, +1 more | ☐ | ☐ |
| 61 | `primitiveBasicAtPut` | Object&gt;&gt;#at:put:, Object&gt;&gt;#basicAt:put:, String&gt;&gt;#byteAt:put:, ArrayedCollection&gt;&gt;#at:put: | 5 | Core.Object×2, Core.ArrayedCollection, Core.ByteArray, Core.String | ☐ | ☐ |
| 62 | `primitiveSize` | Object&gt;&gt;#size, Object&gt;&gt;#basicSize, ArrayedCollection&gt;&gt;#size | 4 | Core.Object×2, Core.ArrayedCollection, Kernel.ProtoObject | ☐ | ☐ |
| 69 | `primitiveSetSpecialBehavior` | Object&gt;&gt;#setSpecialBehavior: | 1 | Core.Object | ☐ | ☐ |
| 70 | `primitiveNew` | Behavior&gt;&gt;#new, Behavior&gt;&gt;#basicNew | 3 | Core.Behavior×2, Kernel.StVariableNode class | ☐ | ☐ |
| 71 | `primitiveNewWithArg` | Behavior&gt;&gt;#new:, Behavior&gt;&gt;#basicNew:, String class&gt;&gt;#new: | 4 | Core.Behavior×2, Core.ArrayedCollection class, Core.String class | ☐ | ☐ |
| 72 | `primitiveBecome` | Object&gt;&gt;#become:, Object&gt;&gt;#swappingBecome: | 3 | Core.Object×2, Kernel.ProtoObject | ☐ | ☐ |
| 73 | `primitiveInstVarAt` | Object&gt;&gt;#instVarAt: | 2 | Core.Object, Kernel.ProtoObject | ☐ | ☐ |
| 74 | `primitiveInstVarAtPut` | Object&gt;&gt;#instVarAt:put: | 1 | Core.Object | ☐ | ☐ |
| 75 | `primitiveBasicIdentityHash` | Object&gt;&gt;#basicIdentityHash | 2 | Core.Object, Kernel.ProtoObject | ☐ | ☐ |
| 76 | `primitiveNewPinned` | Behavior&gt;&gt;#newFixed:, Behavior&gt;&gt;#basicNewFixed:, String class&gt;&gt;#newFixed: | 3 | Core.Behavior×2, Core.String class | ☐ | ☐ |
| 77 | `primitiveAllInstances` | Behavior&gt;&gt;#primAllInstances | 1 | Core.Behavior | ☐ | ☐ |
| 90 | `primitiveNewVirtual` | Behavior&gt;&gt;#new:max: | 1 | Core.Behavior | ☐ | ☐ |
| 101 | `primitiveResize` | Object&gt;&gt;resize:, Object&gt;&gt;#basicResize:, ArrayedCollection&gt;&gt;#resize: | 3 | Core.Object×2, Core.ArrayedCollection | ☐ | ☐ |
| 102 | `primitiveChangeBehavior` | Object&gt;&gt;#becomeA:, Object&gt;&gt;#becomeAn: | 1 | Core.Object | ☐ | ☐ |
| 110 | `primitiveIdentical` | Object&gt;#==, Object&gt;&gt;#= | 5 | Core.Object×2, Refactory.Browser.TestData.RefactoryTestDataApp, Core.Semaphore, Kernel.ProtoObject | ☐ | ☐ |
| 111 | `primitiveClass` | Object&gt;&gt;#class, Object&gt;&gt;#basicClass | 5 | Core.Object×3, Core.Collection, Kernel.ProtoObject | ☐ | ☐ |
| 147 | `primitiveIdentityHash` | Object&gt;&gt;#identityHash | 2 | Core.Object×2 | ☐ | ☐ |
| 148 | `primitiveLookupMethod` | Behavior&gt;&gt;#lookupMethod: | 1 | Core.Behavior | ☐ | ☐ |
| 151 | `primitiveExtraInstanceSpec` | Behavior&gt;&gt;#extraInstanceSpec | 2 | Core.Behavior, External.Structure class | ☐ | ☐ |
| 153 | `primitiveAllReferences` | Object&gt;&gt;#allReferences | 2 | Core.Object, Kernel.ProtoObject | ☐ | ☐ |
| 154 | `primitiveOneWayBecome` | Object&gt;&gt;#oneWayBecome: | 2 | Core.Object, Kernel.ProtoObject | ☐ | ☐ |
| 155 | `primitiveShallowCopy` | Object&gt;&gt;#shallowCopy, Object&gt;&gt;#basicShallowCopy | 4 | Core.Object×2, Core.Utf16String, Core.Utf8String | ☐ | ☐ |
| 157 | `primitiveNewInitializedObject` | e.g. Point class&gt;&gt;#x:y: | 73 | Core.Array class×5, UI.CreateWindow class×2, UI.STBViewContext class×2, Graphics.ImageFromFileInitializer class×2, +60 more | ☐ | ☐ |
| 188 | `primitiveReplacePointers` | Array&gt;&gt;#replaceElementsOf:from:to:startingAt: | 4 | Core.Array, Core.OrderedCollection, Kernel.BlockClosure, Kernel.CompiledCode | ☐ | ☐ |
| 190 | `primitiveNewFromStack` | Array class&gt;&gt;#newFromStack: | 1 | Core.Array class | ☐ | ☐ |

### Streams  (8 primitives, 22 sites)

| # | VM impl (`Interpreter::…`) | Smalltalk selector(s) | Uses | Key classes | Impl | Test |
|--:|---|---|--:|---|:--:|:--:|
| 65 | `primitiveNext` | ReadStream&gt;&gt;#next[Available], FileStream&gt;&gt;#next[Available], ReadWriteStream&gt;&gt;#next[Available] | 5 | Core.ReadWriteStream×2, Core.AbstractReadStream, Core.FileStream, Core.ReadStream | ☐ | ☐ |
| 66 | `primitiveNextPut` | WriteStream&gt;&gt;#nextPut:, FileStream&gt;&gt;#primitiveNextPut:, WriteStream&gt;&gt;#primitiveNextPut: | 3 | Core.AbstractWriteStream×2, Core.FileStream | ☐ | ☐ |
| 67 | `primitiveAtEnd` | PositionableStream&gt;&gt;#atEnd | 2 | Core.ExternalReadStream, Core.PositionableStream | ☐ | ☐ |
| 144 | `primitiveNextInt32` | PositionableStream&gt;&gt;#newSDWORD | 1 | Core.PositionableStream | ☐ | ☐ |
| 172 | `primitiveVariantValue` | Variant&gt;&gt;#value | 1 | OS.COM.VARIANT | ☐ | ☐ |
| 173 | `primitiveNextPutAll` | WriteStream&gt;&gt;#basicNextPutAll:, WriteStream&gt;&gt;#nextPutAll: | 3 | Core.AbstractWriteStream×3 | ☐ | ☐ |
| 221 | `primitiveBasicNext` | ReadStream\|FileStream\|ReadWriteStream&gt;&gt;#basicNext, ReadStream&gt;&gt;#basicNextAvailable | 4 | Core.AbstractReadStream, Core.FileStream, Core.ReadStream, Core.ReadWriteStream | ☐ | ☐ |
| 222 | `primitiveBasicNextPut` | WriteStream&gt;&gt;#basicNextPut:, FileStream\|WriteStream&gt;&gt;#primitiveBasicNextPut: | 3 | Core.AbstractWriteStream×2, Core.FileStream | ☐ | ☐ |

### BlockClosure / perform  (8 primitives, 20 sites)

| # | VM impl (`Interpreter::…`) | Smalltalk selector(s) | Uses | Key classes | Impl | Test |
|--:|---|---|--:|---|:--:|:--:|
| 79 | `primitiveValueOnUnwind` | BlockClosure&gt;&gt;#valueOnUnwind: | 1 | Kernel.BlockClosure | ☐ | ☐ |
| 81 | `primitiveValue` | BlockClosure&gt;&gt;#value, BlockClosure&gt;&gt;(value:)+ | 9 | Kernel.BlockClosure×9 | ☐ | ☐ |
| 82 | `primitiveValueWithArgsThunk` | BlockClosure&gt;&gt;#valueWithArguments: | 1 | Kernel.BlockClosure | ☐ | ☐ |
| 83 | `primitivePerformThunk` | Object&gt;&gt;#perform:(with:)* | 5 | Core.Object×5 | ☐ | ☐ |
| 84 | `primitivePerformWithArgsThunk` | Object&gt;&gt;#perform:withArguments: | 1 | Core.Object | ☐ | ☐ |
| 114 | `primitivePerformWithArgsAtThunk` | Object&gt;&gt;#perform:withArgumentsAt:descriptor: | 1 | Core.Object | ☐ | ☐ |
| 116 | `primitivePerformMethodThunk` | CompiledCode&gt;&gt;#value:withArguments: | 1 | Kernel.CompiledCode | ☐ | ☐ |
| 117 | `primitiveValueWithArgsAtThunk` | BlockClosure&gt;&gt;#valueWithArgumentsAt:descriptor: | 1 | Kernel.BlockClosure | ☐ | ☐ |

### Process / Semaphore / Scheduler  (18 primitives, 21 sites)

| # | VM impl (`Interpreter::…`) | Smalltalk selector(s) | Uses | Key classes | Impl | Test |
|--:|---|---|--:|---|:--:|:--:|
| 85 | `primitiveSignalThunk` | Semaphore&gt;&gt;#signal | 1 | Core.Semaphore | ☐ | ☐ |
| 86 | `primitiveWaitThunk` | Semaphore&gt;&gt;#wait:ret: | 1 | Core.Semaphore | ☐ | ☐ |
| 87 | `primitiveResumeThunk` | Process&gt;&gt;#resume[:] | 3 | Core.Process×3 | ☐ | ☐ |
| 88 | `primitiveSuspendThunk` | Process&gt;&gt;#suspend | 2 | Core.Process×2 | ☐ | ☐ |
| 91 | `primitiveTerminateProcessThunk` | Process&gt;&gt;#primTerminate | 1 | Core.Process | ☐ | ☐ |
| 92 | `primitiveProcessPriorityThunk` | Process&gt;&gt;#priority: | 1 | Core.Process | ☐ | ☐ |
| 93 | `primitiveInputSemaphore` | VMLibrary&gt;&gt;#primRegistryAt:put: | 1 | Kernel.VMLibrary | ☐ | ☐ |
| 94 | `primitiveSampleInterval` | InputState&gt;&gt;#primSampleInterval: | 1 | Kernel.InputState | ☐ | ☐ |
| 95 | `primitiveEnableInterrupts` | ProcessorScheduler&gt;&gt;#enableAsyncEvents: | 1 | Kernel.ProcessorScheduler | ☐ | ☐ |
| 98 | `primitiveQueueInterrupt` | Process&gt;queueInterrupt:with: | 1 | Core.Process | ☐ | ☐ |
| 99 | `primitiveSetSignalsThunk` | Semaphore&gt;&gt;#primSetSignals: | 1 | Core.Semaphore | ☐ | ☐ |
| 100 | `primitiveSignalAtTickThunk` | Delay class&gt;&gt;signalTimerAfter: | 1 | Core.Delay class | ☐ | ☐ |
| 107 | `primitiveUnwindCallback` | ProcessorScheduler&gt;&gt;#primUnwindCallback | 1 | Kernel.ProcessorScheduler | ☐ | ☐ |
| 112 | `primitiveCoreLeftThunk` | MemoryManager&gt;&gt;#primCollectGarbage: | 1 | Kernel.MemoryManager | ☐ | ☐ |
| 115 | `primitiveOopsLeftThunk` | MemoryManager&gt;&gt;#primCompact | 1 | Kernel.MemoryManager | ☐ | ☐ |
| 150 | `primitiveUnwindInterruptThunk` | ProcessorScheduler&gt;&gt;#primUnwindInterrupt | 1 | Kernel.ProcessorScheduler | ☐ | ☐ |
| 156 | `primitiveYieldThunk` | ProcessorScheduler&gt;&gt;#yield | 1 | Kernel.ProcessorScheduler | ☐ | ☐ |
| 175 | `primitiveIndexOfSP` | Process&gt;&gt;#indexOfSP: | 1 | Core.Process | ☐ | ☐ |

### Memory / GC / image / session  (13 primitives, 16 sites)

| # | VM impl (`Interpreter::…`) | Smalltalk selector(s) | Uses | Key classes | Impl | Test |
|--:|---|---|--:|---|:--:|:--:|
| 89 | `primitiveFlushCache` | Behavior&gt;&gt;#flushCache | 1 | Core.Behavior | ☐ | ☐ |
| 97 | `primitiveSnapshot` | SessionManager&gt;&gt;#primSnapshot:backup:type:maxObjects: | 1 | Core.SessionManager | ☐ | ☐ |
| 113 | `primitiveQuit` | SessionManager&gt;&gt;#primQuit: | 1 | Core.SessionManager | ☐ | ☐ |
| 118 | `primitiveDeQForFinalize` | MemoryManager&gt;&gt;#dequeueForFinalization | 1 | Kernel.MemoryManager | ☐ | ☐ |
| 119 | `primitiveDeQBereavement` | MemoryManager&gt;&gt;#dequeueBereavementInto: | 1 | Kernel.MemoryManager | ☐ | ☐ |
| 169 | `primitiveObjectCount` | MemoryManager&gt;&gt;#objectCount | 1 | Kernel.MemoryManager | ☐ | ☐ |
| 170 | `primitiveStructureIsNull` | External.Structure&gt;&gt;#isNull | 1 | External.Structure | ☐ | ☐ |
| 171 | `primitiveBytesIsNull` | External.IntegerBytes&gt;&gt;isNull | 1 | External.IntegerBytes | ☐ | ☐ |
| 174 | `primitiveMillisecondClockValue` | MemoryManager&gt;&gt;#millisecondClock, InputState&gt;&gt;#millisecondClodkValue, Delay class&gt;&gt;#millisecondClockValue | 3 | Core.Delay class, Kernel.InputState, Kernel.MemoryManager | ☐ | ☐ |
| 177 | `primitiveGetImmutable` | Object&gt;&gt;#isImmutable | 1 | Core.Object | ☐ | ☐ |
| 178 | `primitiveSetImmutable` | Object&gt;&gt;#isImmutable: | 1 | Core.Object | ☐ | ☐ |
| 179 | `primitiveInstanceCounts` | MemoryManager&gt;&gt;#primInstanceStats: | 1 | Kernel.MemoryManager | ☐ | ☐ |
| 189 | `primitiveMicrosecondClockValue` | Delay&gt;&gt;#microsecondClockValue, Delay class&gt;&gt;#microsecondClockValue | 2 | Core.Delay, Core.Delay class | ☐ | ☐ |

## Appendix A — VM-defined primitives with no source pragma

Implemented by the VM but never written as `<primitive: N>` in the image
(FFI/COM call primitives emitted by the compiler, plus reserved/aux slots).
Several are load-bearing regardless of pragma count.

| # | VM impl | Canonical use |
|--:|---|---|
| 0 | `primitiveActivateMethod` | All methods without specific primitive |
| 3 | `primitiveReturnConst&lt;3&gt;` | ^true |
| 4 | `primitiveReturnConst&lt;1&gt;` | ^nil |
| 5 | `primitiveReturnLiteralZero` | ^&lt;literal&gt; |
| 6 | `primitiveReturnInstVar` | ^&lt;inst-var&gt; |
| 7 | `primitiveSetInstVar` | &lt;inst-var&gt; := &lt;arg&gt; |
| 8 | `primitiveReturnStaticZero` | ^&lt;static-var&gt; |
| 48 | `primitiveAsyncDLL32CallThunk` | Overlapped FFI call primitive |
| 49 | `primitiveSetMutableInstVar` | &lt;mutable inst var&gt; := arg |
| 80 | `primitiveVirtualCall` | C++ virtual (and COM method) FFI calls |
| 96 | `primitiveDLL32Call` | FFI call primitive |
| 105 | `primitiveSingleStepThunk` | Process&gt;&gt;#primStep: |
| 223 | `primitiveStringAsUtf32String` | Currently unimplemented/unused |
| 230 | `primitiveSnapshot64` | 32-bit VM cross-builds 64-bit image |

## Appendix B — full numeric index (every used primitive)

| # | VM impl | Selector(s) | Uses |
|--:|---|---|--:|
| 1 | `primitiveReturnSelf` | ^self | 10 |
| 2 | `primitiveReturnConst&lt;2&gt;` | ^false | 9 |
| 9 | `primitiveMultiply` | SmallInteger&gt;&gt;#* | 1 |
| 10 | `primitiveDivide` | SmallInteger&gt;&gt;#/ | 2 |
| 11 | `primitiveMod` | SmallInteger&gt;&gt;#'\\' | 1 |
| 12 | `primitiveDiv` | SmallInteger&gt;&gt;#// | 1 |
| 13 | `primitiveQuo` | SmallInteger&gt;&gt;#quo: | 1 |
| 14 | `primitiveSubtract` | SmallInteger&gt;&gt;#- | 1 |
| 15 | `primitiveAdd` | SmallInteger&gt;&gt;#+ | 1 |
| 16 | `primitiveEqual` | SmallInteger&gt;&gt;#= | 1 |
| 17 | `primitiveIntegerCmp&lt;std::greater_equal&lt;SmallInteger&gt;, true&gt;` | SmallInteger&gt;&gt;#&gt;= | 1 |
| 18 | `primitiveIntegerCmp&lt;std::less&lt;SmallInteger&gt;, false&gt;` | SmallInteger&gt;&gt;#&lt; | 1 |
| 19 | `primitiveIntegerCmp&lt;std::greater&lt;SmallInteger&gt;, true&gt;` | SmallInteger&gt;&gt;#&gt; | 1 |
| 20 | `primitiveIntegerCmp&lt;std::less_equal&lt;SmallInteger&gt;, false&gt;` | SmallInteger&gt;&gt;#&lt;= | 1 |
| 21 | `primitiveLargeIntegerOpR&lt;Li::Add, Li::AddSingle&gt;` | LargeInteger&gt;&gt;#+ | 2 |
| 22 | `primitiveLargeIntegerOpR&lt;Li::Sub, Li::SubSingle&gt;` | LargeInteger&gt;&gt;#- | 1 |
| 23 | `primitiveLargeIntegerCmp&lt;true,false&gt;` | LargeInteger&gt;&gt;#&lt; | 1 |
| 24 | `primitiveLargeIntegerCmp&lt;false,false&gt;` | LargeInteger&gt;&gt;#&gt; | 2 |
| 25 | `primitiveLargeIntegerCmp&lt;true, true&gt;` | LargeInteger&gt;&gt;#&lt;= | 1 |
| 26 | `primitiveLargeIntegerCmp&lt;false,true&gt;` | LargeInteger&gt;&gt;#&gt;= | 1 |
| 27 | `primitiveLargeIntegerEqual` | LargeInteger&gt;&gt;#= | 1 |
| 28 | `primitiveLargeIntegerUnaryOp&lt;Li::Normalize&gt;` | LargeInteger&gt;&gt;normalize | 1 |
| 29 | `primitiveLargeIntegerOpZ&lt;Li::Mul, Li::MulSingle&gt;` | LargeInteger&gt;&gt;#* | 2 |
| 30 | `primitiveLargeIntegerDivide` | LargeInteger&gt;&gt;#/ | 1 |
| 31 | `unusedPrimitive` | LargeInteger#\\ | 1 |
| 32 | `unusedPrimitive` | LargeInteger&gt;&gt;#// | 1 |
| 33 | `primitiveLargeIntegerQuo` | LargeInteger&gt;&gt;#quo: | 1 |
| 34 | `primitiveLargeIntegerOpZ&lt;Li::BitAnd, Li::BitAndSingle&gt;` | LargeInteger&gt;&gt;#bitAnd: | 1 |
| 35 | `primitiveLargeIntegerOpR&lt;Li::BitOr, Li::BitOrSingle&gt;` | LargeInteger&gt;&gt;#bitOr: | 1 |
| 36 | `primitiveLargeIntegerOpR&lt;Li::BitXor, Li::BitXorSingle&gt;` | LargeInteger&gt;&gt;#bitXor: | 1 |
| 37 | `primitiveLargeIntegerBitShift` | LargeInteger&gt;&gt;#bitShift | 1 |
| 38 | `primitiveLargeIntegerBitInvert` | LargeInteger&gt;&gt;#bitInvert | 1 |
| 39 | `primitiveLargeIntegerUnaryOp&lt;Li::Negate&gt;` | LargeInteger&gt;&gt;#negate | 1 |
| 40 | `primitiveIntegerOp&lt;std::bit_and&lt;Oop&gt;&gt;` | SmallInteger&gt;&gt;#bitAnd: | 2 |
| 41 | `primitiveIntegerOp&lt;std::bit_or&lt;Oop&gt;&gt;` | SmallInteger&gt;&gt;#bitOr: | 2 |
| 42 | `primitiveIntegerOp&lt;bit_xor&gt;` | SmallInteger&gt;&gt;#bitXor: | 1 |
| 43 | `primitiveBitShift` | SmallInteger&gt;&gt;#bitShift: | 2 |
| 44 | `primitiveSmallIntegerPrintString` | SmallInteger&gt;&gt;#printString | 1 |
| 45 | `primitiveFloatCompare&lt;std::greater&lt;double&gt;&gt;` | Float&gt;&gt;#&gt; | 1 |
| 46 | `primitiveFloatCompare&lt;std::greater_equal&lt;double&gt;&gt;` | Float&gt;&gt;#&gt;= | 1 |
| 47 | `primitiveFloatCompare&lt;std::equal_to&lt;double&gt;&gt;` | Float&gt;&gt;#= | 1 |
| 50 | `primitiveCopyFromTo` | ArrayedCollection&gt;&gt;#copyFrom:to: | 1 |
| 51 | `primitiveStringComparison&lt;CmpA,CmpW&gt;` | String&gt;&gt;#&lt;==&gt; | 1 |
| 52 | `primitiveStringNextIndexOfFromTo` | String&gt;&gt;#nextIdentityIndexOf:from:to: | 2 |
| 53 | `primitiveLargeIntegerHighBit` | LargeInteger&gt;&gt;#highBit | 1 |
| 54 | `primitiveHighBit` | SmallInteger&gt;&gt;#highBit | 1 |
| 55 | `primitiveBytesEqual` | ByteArray&gt;&gt;#= | 1 |
| 56 | `primitiveStringComparison&lt;CmpIA,CmpIW&gt;` | String&gt;&gt;#&lt;=&gt; | 1 |
| 57 | `primitiveIsKindOf` | Object&gt;&gt;#isKindOf: | 2 |
| 58 | `primitiveAllSubinstances` | Behavior&gt;&gt;#primAllSubinstances | 1 |
| 59 | `primitiveNextIndexOfFromTo` | Object&gt;&gt;#basicIdentityIndexOf:from:to:, ArrayedCollection&gt;&gt;#nextIdentityIndexOf:from:to: | 3 |
| 60 | `primitiveBasicAt` | Object&gt;&gt;#at:, Object&gt;&gt;#basicAt:, String&gt;&gt;#byteAt:, ArrayedCollection&gt;&gt;#at:, ArrayedCollection&gt;&gt;#lookup: | 9 |
| 61 | `primitiveBasicAtPut` | Object&gt;&gt;#at:put:, Object&gt;&gt;#basicAt:put:, String&gt;&gt;#byteAt:put:, ArrayedCollection&gt;&gt;#at:put: | 5 |
| 62 | `primitiveSize` | Object&gt;&gt;#size, Object&gt;&gt;#basicSize, ArrayedCollection&gt;&gt;#size | 4 |
| 63 | `primitiveStringAt` | String&gt;&gt;#at: | 5 |
| 64 | `primitiveStringAtPut` | String&gt;&gt;#at:put: | 1 |
| 65 | `primitiveNext` | ReadStream&gt;&gt;#next[Available], FileStream&gt;&gt;#next[Available], ReadWriteStream&gt;&gt;#next[Available] | 5 |
| 66 | `primitiveNextPut` | WriteStream&gt;&gt;#nextPut:, FileStream&gt;&gt;#primitiveNextPut:, WriteStream&gt;&gt;#primitiveNextPut: | 3 |
| 67 | `primitiveAtEnd` | PositionableStream&gt;&gt;#atEnd | 2 |
| 68 | `primitiveReturnFromInterrupt` | ProcessorScheduler&gt;&gt;iret:list: | 1 |
| 69 | `primitiveSetSpecialBehavior` | Object&gt;&gt;#setSpecialBehavior: | 1 |
| 70 | `primitiveNew` | Behavior&gt;&gt;#new, Behavior&gt;&gt;#basicNew | 3 |
| 71 | `primitiveNewWithArg` | Behavior&gt;&gt;#new:, Behavior&gt;&gt;#basicNew:, String class&gt;&gt;#new: | 4 |
| 72 | `primitiveBecome` | Object&gt;&gt;#become:, Object&gt;&gt;#swappingBecome: | 3 |
| 73 | `primitiveInstVarAt` | Object&gt;&gt;#instVarAt: | 2 |
| 74 | `primitiveInstVarAtPut` | Object&gt;&gt;#instVarAt:put: | 1 |
| 75 | `primitiveBasicIdentityHash` | Object&gt;&gt;#basicIdentityHash | 2 |
| 76 | `primitiveNewPinned` | Behavior&gt;&gt;#newFixed:, Behavior&gt;&gt;#basicNewFixed:, String class&gt;&gt;#newFixed: | 3 |
| 77 | `primitiveAllInstances` | Behavior&gt;&gt;#primAllInstances | 1 |
| 78 | `primitiveReturn` | ProcessorScheduler&gt;&gt;#returnValue:toFrame: | 1 |
| 79 | `primitiveValueOnUnwind` | BlockClosure&gt;&gt;#valueOnUnwind: | 1 |
| 81 | `primitiveValue` | BlockClosure&gt;&gt;#value, BlockClosure&gt;&gt;(value:)+ | 9 |
| 82 | `primitiveValueWithArgsThunk` | BlockClosure&gt;&gt;#valueWithArguments: | 1 |
| 83 | `primitivePerformThunk` | Object&gt;&gt;#perform:(with:)* | 5 |
| 84 | `primitivePerformWithArgsThunk` | Object&gt;&gt;#perform:withArguments: | 1 |
| 85 | `primitiveSignalThunk` | Semaphore&gt;&gt;#signal | 1 |
| 86 | `primitiveWaitThunk` | Semaphore&gt;&gt;#wait:ret: | 1 |
| 87 | `primitiveResumeThunk` | Process&gt;&gt;#resume[:] | 3 |
| 88 | `primitiveSuspendThunk` | Process&gt;&gt;#suspend | 2 |
| 89 | `primitiveFlushCache` | Behavior&gt;&gt;#flushCache | 1 |
| 90 | `primitiveNewVirtual` | Behavior&gt;&gt;#new:max: | 1 |
| 91 | `primitiveTerminateProcessThunk` | Process&gt;&gt;#primTerminate | 1 |
| 92 | `primitiveProcessPriorityThunk` | Process&gt;&gt;#priority: | 1 |
| 93 | `primitiveInputSemaphore` | VMLibrary&gt;&gt;#primRegistryAt:put: | 1 |
| 94 | `primitiveSampleInterval` | InputState&gt;&gt;#primSampleInterval: | 1 |
| 95 | `primitiveEnableInterrupts` | ProcessorScheduler&gt;&gt;#enableAsyncEvents: | 1 |
| 97 | `primitiveSnapshot` | SessionManager&gt;&gt;#primSnapshot:backup:type:maxObjects: | 1 |
| 98 | `primitiveQueueInterrupt` | Process&gt;queueInterrupt:with: | 1 |
| 99 | `primitiveSetSignalsThunk` | Semaphore&gt;&gt;#primSetSignals: | 1 |
| 100 | `primitiveSignalAtTickThunk` | Delay class&gt;&gt;signalTimerAfter: | 1 |
| 101 | `primitiveResize` | Object&gt;&gt;resize:, Object&gt;&gt;#basicResize:, ArrayedCollection&gt;&gt;#resize: | 3 |
| 102 | `primitiveChangeBehavior` | Object&gt;&gt;#becomeA:, Object&gt;&gt;#becomeAn: | 1 |
| 103 | `primitiveAddressOf` | Object&gt;&gt;#yourAddress | 2 |
| 104 | `primitiveReturnFromCallback` | ProcessorScheduler&gt;&gt;#primReturn:callback: | 1 |
| 106 | `primitiveHashBytes` | ByteArray&gt;&gt;#hash, String&gt;&gt;#hash, LargeInteger&gt;&gt;#hash | 4 |
| 107 | `primitiveUnwindCallback` | ProcessorScheduler&gt;&gt;#primUnwindCallback | 1 |
| 109 | `primitiveHashMultiply` | Integer&gt;&gt;#hashMultiply, SmallInteger&gt;&gt;#identityHash, SmallInteger&gt;&gt;#hash | 3 |
| 110 | `primitiveIdentical` | Object&gt;#==, Object&gt;&gt;#= | 5 |
| 111 | `primitiveClass` | Object&gt;&gt;#class, Object&gt;&gt;#basicClass | 5 |
| 112 | `primitiveCoreLeftThunk` | MemoryManager&gt;&gt;#primCollectGarbage: | 1 |
| 113 | `primitiveQuit` | SessionManager&gt;&gt;#primQuit: | 1 |
| 114 | `primitivePerformWithArgsAtThunk` | Object&gt;&gt;#perform:withArgumentsAt:descriptor: | 1 |
| 115 | `primitiveOopsLeftThunk` | MemoryManager&gt;&gt;#primCompact | 1 |
| 116 | `primitivePerformMethodThunk` | CompiledCode&gt;&gt;#value:withArguments: | 1 |
| 117 | `primitiveValueWithArgsAtThunk` | BlockClosure&gt;&gt;#valueWithArgumentsAt:descriptor: | 1 |
| 118 | `primitiveDeQForFinalize` | MemoryManager&gt;&gt;#dequeueForFinalization | 1 |
| 119 | `primitiveDeQBereavement` | MemoryManager&gt;&gt;#dequeueBereavementInto: | 1 |
| 120 | `primitiveIntegerAtOffset&lt;uint32_t, StoreUnsigned32&gt;` | ByteArray&gt;&gt;#dwordAtOffset: | 4 |
| 121 | `primitiveUint32AtPut` | ByteArray&gt;&gt;#dwordAtOffset:put: | 4 |
| 122 | `primitiveIntegerAtOffset&lt;int32_t, StoreSigned32&gt;` | ByteArray&gt;&gt;#sdwordAtOffset: | 2 |
| 123 | `primitiveInt32AtPut` | ByteArray&gt;&gt;#sdwordAtOffset:put: | 1 |
| 124 | `primitiveIntegerAtOffset&lt;uint16_t, StoreSmallInteger&gt;` | ByteArray&gt;&gt;#wordAtOffset: | 2 |
| 125 | `primitiveAtOffsetPutInteger&lt;uint16_t, 0x0, 0xffff&gt;` | ByteArray&gt;&gt;#wordAtOffset:put: | 2 |
| 126 | `primitiveIntegerAtOffset&lt;int16_t, StoreSmallInteger&gt;` | ByteArray&gt;&gt;#swordAtoffset: | 1 |
| 127 | `primitiveAtOffsetPutInteger&lt;uint16_t, -0x8000, 0x7fff&gt;` | ByteArray&gt;&gt;#swordAtOffset:put: | 1 |
| 128 | `primitiveFloatAtOffset&lt;double&gt;` | ByteArray&gt;&gt;#doubleAtOffset: | 1 |
| 129 | `primitiveFloatAtOffsetPut&lt;double&gt;` | ByteArray&gt;&gt;#doubleAtOffset:put: | 1 |
| 130 | `primitiveFloatAtOffset&lt;float&gt;` | ByteArray&gt;&gt;#floatAtOffset: | 1 |
| 131 | `primitiveFloatAtOffsetPut&lt;float&gt;` | ByteArray&gt;&gt;#floatAtOffset:put: | 1 |
| 132 | `primitiveIndirectIntegerAtOffset&lt;uint8_t, StoreSmallInteger&gt;` | External.Address&gt;&gt;#byteAtOffset: | 1 |
| 133 | `primitiveIndirectAtOffsetPutInteger&lt;uint8_t, 0, 255&gt;` | External.Address&gt;&gt;#byteAtOffset:put: | 1 |
| 134 | `primitiveIndirectIntegerAtOffset&lt;uint32_t, StoreUnsigned32&gt;` | External.Address&gt;&gt;#dwordAtOffset: | 1 |
| 135 | `primitiveIndirectUint32AtPut` | External.Address&gt;&gt;#dwordAtOffset:put: | 1 |
| 136 | `primitiveIndirectIntegerAtOffset&lt;int32_t, StoreSigned32&gt;` | External.Address&gt;&gt;#sdwordAtOffset: | 1 |
| 137 | `primitiveIndirectInt32AtPut` | External.Address&gt;&gt;#sdwordAtOffset:put: | 1 |
| 138 | `primitiveIndirectIntegerAtOffset&lt;uint16_t, StoreSmallInteger&gt;` | External.Address&gt;&gt;#wordAtOffset: | 1 |
| 139 | `primitiveIndirectAtOffsetPutInteger&lt;uint16_t, 0, 0xffff&gt;` | External.Address&gt;&gt;#wordAtOffset:put: | 1 |
| 140 | `primitiveIndirectIntegerAtOffset&lt;int16_t, StoreSmallInteger&gt;` | External.Address&gt;&gt;#swordAtOffset: | 1 |
| 141 | `primitiveIndirectAtOffsetPutInteger&lt;int16_t, -0x8000, 0x7fff&gt;` | External.Address&gt;&gt;#swordAtOffset:put: | 1 |
| 142 | `primitiveReplaceBytes` | ByteArray\|String&gt;&gt;#replaceBytesOf:from:to:startingAt: | 4 |
| 143 | `primitiveIndirectReplaceBytes` | ExternalAddress&gt;&gt;#replaceBytesOf:from:to:startingAt: | 1 |
| 144 | `primitiveNextInt32` | PositionableStream&gt;&gt;#newSDWORD | 1 |
| 145 | `primitiveAnyMask` | SmallInteger&gt;&gt;#anyMask: | 1 |
| 146 | `primitiveAllMask` | SmallInteger&gt;&gt;#allMask: | 1 |
| 147 | `primitiveIdentityHash` | Object&gt;&gt;#identityHash | 2 |
| 148 | `primitiveLookupMethod` | Behavior&gt;&gt;#lookupMethod: | 1 |
| 149 | `primitiveStringSearch` | String&gt;&gt;#findString:startingAt: | 3 |
| 150 | `primitiveUnwindInterruptThunk` | ProcessorScheduler&gt;&gt;#primUnwindInterrupt | 1 |
| 151 | `primitiveExtraInstanceSpec` | Behavior&gt;&gt;#extraInstanceSpec | 2 |
| 152 | `primitiveLowBit` | SmallInteger&gt;&gt;#lowBit | 1 |
| 153 | `primitiveAllReferences` | Object&gt;&gt;#allReferences | 2 |
| 154 | `primitiveOneWayBecome` | Object&gt;&gt;#oneWayBecome: | 2 |
| 155 | `primitiveShallowCopy` | Object&gt;&gt;#shallowCopy, Object&gt;&gt;#basicShallowCopy | 4 |
| 156 | `primitiveYieldThunk` | ProcessorScheduler&gt;&gt;#yield | 1 |
| 157 | `primitiveNewInitializedObject` | e.g. Point class&gt;&gt;#x:y: | 73 |
| 158 | `primitiveSmallIntegerAt` | SmallInteger&gt;&gt;#byteAt: | 1 |
| 159 | `primitiveLongDoubleAt` | ByteArray&gt;&gt;#longDoubleAtOffset: | 1 |
| 160 | `primitiveFloatBinaryOp&lt;std::plus&lt;double&gt;&gt;` | Float&gt;&gt;#+ | 1 |
| 161 | `primitiveFloatBinaryOp&lt;std::minus&lt;double&gt;&gt;` | Float&gt;&gt;#- | 1 |
| 162 | `primitiveFloatCompare&lt;std::less&lt;double&gt;&gt;` | Float&gt;&gt;#&lt; | 1 |
| 164 | `primitiveFloatBinaryOp&lt;std::multiplies&lt;double&gt;&gt;` | Float&gt;&gt;#* | 1 |
| 165 | `primitiveFloatBinaryOp&lt;std::divides&lt;double&gt;&gt;` | Float&gt;&gt;#/ | 1 |
| 166 | `primitiveFloatTruncationOp&lt;Truncate&gt;` | Float&gt;&gt;#truncated | 1 |
| 167 | `primitiveLargeIntegerAsFloat` | LargeInteger&gt;&gt;#asFloat | 1 |
| 168 | `primitiveAsFloat` | SmallInteger&gt;&gt;#asFloat | 1 |
| 169 | `primitiveObjectCount` | MemoryManager&gt;&gt;#objectCount | 1 |
| 170 | `primitiveStructureIsNull` | External.Structure&gt;&gt;#isNull | 1 |
| 171 | `primitiveBytesIsNull` | External.IntegerBytes&gt;&gt;isNull | 1 |
| 172 | `primitiveVariantValue` | Variant&gt;&gt;#value | 1 |
| 173 | `primitiveNextPutAll` | WriteStream&gt;&gt;#basicNextPutAll:, WriteStream&gt;&gt;#nextPutAll: | 3 |
| 174 | `primitiveMillisecondClockValue` | MemoryManager&gt;&gt;#millisecondClock, InputState&gt;&gt;#millisecondClodkValue, Delay class&gt;&gt;#millisecondClockValue | 3 |
| 175 | `primitiveIndexOfSP` | Process&gt;&gt;#indexOfSP: | 1 |
| 176 | `primitiveStackAtPut` | Process&gt;&gt;#at:put:, Process&gt;&gt;#basicAt:put: | 2 |
| 177 | `primitiveGetImmutable` | Object&gt;&gt;#isImmutable | 1 |
| 178 | `primitiveSetImmutable` | Object&gt;&gt;#isImmutable: | 1 |
| 179 | `primitiveInstanceCounts` | MemoryManager&gt;&gt;#primInstanceStats: | 1 |
| 180 | `primitiveIntegerAtOffset&lt;uintptr_t, StoreUIntPtr&gt;` | ByteArray&gt;&gt;#uintPtrAtOffset: | 1 |
| 181 | `primitiveUintPtrAtPut` | ByteArray&gt;&gt;#uintPtrAtOffset:put:   (was primitiveUint32AtPut — truncated on x64) | 1 |
| 182 | `primitiveIntegerAtOffset&lt;intptr_t, StoreIntPtr&gt;` | ByteArray&gt;&gt;#intptrAtOffset: | 1 |
| 183 | `primitiveIntPtrAtPut` | ByteArray&gt;&gt;#intPtrAtOffset:put:    (was primitiveInt32AtPut — truncated on x64) | 2 |
| 184 | `primitiveIndirectIntegerAtOffset&lt;uintptr_t, StoreUIntPtr&gt;` | External.Address&gt;&gt;#uintPtrAtOffset: | 1 |
| 185 | `primitiveIndirectUintPtrAtPut` | External.Address&gt;&gt;#uintPtrAtOffset:put: (was primitiveIndirectUint32AtPut — truncated on x64) | 1 |
| 186 | `primitiveIndirectIntegerAtOffset&lt;intptr_t, StoreIntPtr&gt;` | External.Address&gt;&gt;#intptrAtOffset: | 1 |
| 187 | `primitiveIndirectIntPtrAtPut` | External.Address&gt;&gt;#intPtrAtOffset:put:  (was primitiveIndirectInt32AtPut — truncated on x64) | 1 |
| 188 | `primitiveReplacePointers` | Array&gt;&gt;#replaceElementsOf:from:to:startingAt: | 4 |
| 189 | `primitiveMicrosecondClockValue` | Delay&gt;&gt;#microsecondClockValue, Delay class&gt;&gt;#microsecondClockValue | 2 |
| 190 | `primitiveNewFromStack` | Array class&gt;&gt;#newFromStack: | 1 |
| 191 | `primitiveIntegerAtOffset&lt;uint64_t, StoreUnsigned64&gt;` | ByteArray&gt;&gt;#qwordAtOffset: | 2 |
| 192 | `primitiveIntegerAtOffset&lt;int64_t, StoreSigned64&gt;` | ByteArray&gt;&gt;#sqwordAtOffset: | 1 |
| 193 | `primitiveFloatUnaryOp&lt;Sin&gt;` | Float&gt;&gt;#sin | 1 |
| 194 | `primitiveFloatUnaryOp&lt;Tan&gt;` | Float&gt;&gt;#tan | 1 |
| 195 | `primitiveFloatUnaryOp&lt;Cos&gt;` | Float&gt;&gt;#cos | 1 |
| 196 | `primitiveFloatUnaryOp&lt;ArcSin&gt;` | Float&gt;&gt;#arcSin | 1 |
| 197 | `primitiveFloatUnaryOp&lt;ArcTan&gt;` | Float&gt;&gt;#arcTan | 1 |
| 198 | `primitiveFloatUnaryOp&lt;ArcCos&gt;` | Float&gt;&gt;#arcCos | 1 |
| 199 | `primitiveFloatBinaryOp&lt;Atan2&gt;` | Float&gt;&gt;#acTan: | 1 |
| 200 | `primitiveFloatUnaryOp&lt;Log&gt;` | Float&gt;&gt;#ln | 1 |
| 201 | `primitiveFloatUnaryOp&lt;Exp&gt;` | Float&gt;&gt;#exp | 1 |
| 202 | `primitiveFloatUnaryOp&lt;Sqrt&gt;` | Float&gt;&gt;#sqrt | 1 |
| 203 | `primitiveFloatUnaryOp&lt;Log10&gt;` | Float&gt;&gt;#log | 1 |
| 204 | `primitiveFloatTimesTwoPower` | Float&gt;&gt;#timesTwoPower: | 1 |
| 205 | `primitiveFloatUnaryOp&lt;Abs&gt;` | Float&gt;&gt;#abs | 1 |
| 206 | `primitiveFloatBinaryOp&lt;Pow&gt;` | Float&gt;&gt;#raisedTo: | 1 |
| 207 | `primitiveFloatTruncationOp&lt;Floor&gt;` | Float&gt;&gt;#floor | 1 |
| 208 | `primitiveFloatTruncationOp&lt;Ceiling&gt;` | Float&gt;&gt;#ceiling | 1 |
| 209 | `primitiveFloatExponent` | Float&gt;&gt;#exponent | 1 |
| 210 | `primitiveFloatUnaryOp&lt;Negated&gt;` | Float&gt;&gt;#negated | 1 |
| 211 | `primitiveFloatClassify` | Float&gt;&gt;#fpClass | 1 |
| 212 | `primitiveFloatUnaryOp&lt;FractionPart&gt;` | Float&gt;&gt;#fractionPart | 1 |
| 213 | `primitiveFloatUnaryOp&lt;IntegerPart&gt;` | Float&gt;&gt;#integerPart | 1 |
| 214 | `primitiveFloatCompare&lt;std::less_equal&lt;double&gt;&gt;` | Float&gt;&gt;#&lt;= | 1 |
| 215 | `primitiveStringAsUtf16String` | String&gt;&gt;#asUtf16String | 2 |
| 216 | `primitiveStringAsUtf8String` | String&gt;&gt;#asUtf8String | 3 |
| 217 | `primitiveStringAsByteString` | String&gt;&gt;#asAnsiString | 1 |
| 218 | `primitiveStringConcatenate` | String&gt;&gt;#, | 1 |
| 219 | `primitiveStringOrdinalEqual` | String&gt;&gt;#= | 1 |
| 220 | `primitiveStringCompareOrdinal` | String&gt;&gt;#compareOrdinals:ignoringCase: | 1 |
| 221 | `primitiveBasicNext` | ReadStream\|FileStream\|ReadWriteStream&gt;&gt;#basicNext, ReadStream&gt;&gt;#basicNextAvailable | 4 |
| 222 | `primitiveBasicNextPut` | WriteStream&gt;&gt;#basicNextPut:, FileStream\|WriteStream&gt;&gt;#primitiveBasicNextPut: | 3 |
| 224 | `primitiveBeginsWith` | String&gt;&gt;#beginsWith: | 1 |
| 225 | `primitiveStringDecodeAt` | Utf8String&gt;&gt;decodeAt: | 1 |
| 226 | `primitiveStringEncodedSizeAt` | Utf8String&gt;&gt;encodedSizeAt: | 2 |
| 227 | `primitiveOrdinalHashIgnoreCase` | String&gt;&gt;#hashOrdinalsIgnoringCase: | 1 |
| 228 | `primitiveStringLessOrEqual` | String&gt;&gt;#&lt;= | 1 |
| 229 | `primitiveCharacterEquals` | Character&gt;&gt;#= | 1 |

---

*Generated from `E:\dsfork` — `PrimitivesTable.cpp` (VM dispatch table) joined
with `<primitive: N>` pragma occurrences across all `.cls` image source. Excludes
mentions inside comments and test strings; excludes the false `<primitive: 256>`
in a parser-error test fixture.*
