Object subclass: #CogBench
	instanceVariableNames: ''
	classVariableNames: ''
	package: 'CogBench'!

!CogBench class methodsFor: 'bench'!
benchArith
	| s |
	s := 0.
	1 to: 1500000 do: [:i | s := s + (i * i) - (i * 3) ].
	^s!
fib: n
	n < 2 ifTrue: [ ^n ].
	^(self fib: n - 1) + (self fib: n - 2)!
benchFib
	^self fib: 32!
sieveOnce
	| flags count size |
	size := 8190.
	flags := Array new: size.
	1 to: size do: [:x | flags at: x put: true ].
	count := 0.
	1 to: size do: [:i |
		(flags at: i) ifTrue: [
			| prime k |
			prime := i + i + 1.
			k := i + prime.
			[ k <= size ] whileTrue: [ flags at: k put: false. k := k + prime ].
			count := count + 1 ] ].
	^count!
benchSieve
	| count |
	count := 0.
	4 timesRepeat: [ count := self sieveOnce ].
	^count!
benchDict
	| d sum |
	d := Dictionary new.
	1 to: 8000 do: [:i | d at: i put: i * i ].
	sum := 0.
	1 to: 8000 do: [:i | sum := sum + (d at: i) ].
	^sum!
benchAlloc
	| last |
	last := nil.
	1 to: 200000 do: [:i | last := Association key: i value: last ].
	^last key!
median: coll
	| s |
	s := coll asSortedCollection asArray.
	^s at: (s size + 1) // 2!
check: name got: got want: want
	got = want ifFalse: [
		Stdio stdout nextPutAll: name; nextPutAll: ' WRONG RESULT '; print: got; lf; flush.
		Smalltalk exitFailure ]!
run: name block: aBlock check: want
	| times cold t0 r |
	t0 := Time millisecondClockValue.
	10 timesRepeat: [ r := aBlock value ].
	self check: name got: r want: want.
	cold := Time millisecondClockValue - t0.
	times := OrderedCollection new.
	6 timesRepeat: [
		t0 := Time millisecondClockValue.
		10 timesRepeat: [ r := aBlock value ].
		times add: Time millisecondClockValue - t0 ].
	self check: name got: r want: want.
	Stdio stdout nextPutAll: name; nextPutAll: ' cold='; print: cold;
		nextPutAll: ' warm='; print: (self median: times); lf; flush!
runAll
	self run: 'arith    ' block: [ self benchArith ] check: 1124997749998000000.
	self run: 'fib      ' block: [ self benchFib ] check: 2178309.
	self run: 'sieve    ' block: [ self benchSieve ] check: 1899.
	self run: 'dict     ' block: [ self benchDict ] check: 170698668000.
	self run: 'alloc    ' block: [ self benchAlloc ] check: 200000.! !

CogBench runAll.!
Smalltalk exitSuccess.!
