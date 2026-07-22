# Granular sockets — design + investigation

**Status: DESIGN (investigated empirically 2026-07-21).** Ask: raw-socket
support giving granular control over our own protocols. **Answer: the full
BSD socket surface — UDP datagrams, arbitrary addresses, every socket
option, and hand-crafted ICMP — is reachable TODAY through the S20 FFI with
zero new Rust, proven live this session. True `SOCK_RAW` is privilege-gated
by the OS (EPERM without root), so it is exposed honestly as a value-
returning constructor; the unprivileged capability that actually delivers
"our own protocols" is `SOCK_DGRAM`+`IPPROTO_ICMP` packet crafting plus UDP.**

## What the investigation established (all run for real, loopback-only)

The probes (`sock_probe.mst`, `ping_probe.mst`, session scratchpad):

1. **UDP datagrams round-trip.** `socket(AF_INET, SOCK_DGRAM, 0)`, `bind` to
   an ephemeral loopback port, `sendto` a 3-byte payload from an unbound
   client, `recvfrom` on the server — exact bytes back, and the kernel-
   filled peer sockaddr yields the client's ephemeral port. `sendto`/
   `recvfrom` are 6 args each: FFI-reachable (≤8, no stack-spill needed).

2. **Socket options work.** `setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &1, 4)`
   returned 0. 5 args, reachable. `getsockopt` is the value-result twin.

3. **`SOCK_RAW` is privilege-gated.** `socket(AF_INET, SOCK_RAW, IPPROTO_ICMP)`
   → EPERM (errno 1) unprivileged, as expected on macOS. Exposed but never
   the load-bearing path.

4. **Unprivileged ICMP works** — the key finding. `socket(AF_INET,
   SOCK_DGRAM, IPPROTO_ICMP)` OPENS without root (macOS's ping path). A
   hand-built 8-byte ICMP echo request (type 8, one's-complement checksum
   computed in Smalltalk) `sendto` 127.0.0.1 returned 8; `recvfrom` got 28
   bytes = a **20-byte IP header** (first byte `0x45`: IPv4, IHL 5) followed
   by the 8-byte echo reply. The kernel rewrites the ICMP id (like an
   ephemeral port) for match-back. So crafting and parsing our own wire
   protocol, checksum and all, needs no privilege — a real ping.

**Two macOS specifics baked into the design:** the DGRAM-ICMP reply
**includes the IP header** (unlike Linux), so a parser skips `(firstByte &
0x0f) * 4` bytes to reach the ICMP header; and the kernel owns the ICMP id,
so callers don't match on it.

## Design (three world layers, zero new Rust)

Builds on world/61's existing `socket`/`bind`/`listen`/`accept`/`connect`/
`getsockname` and its `sockaddr_in` packing, and on the GC-stable
`NativeBuffer` rule (kernel buffers never live in the movable heap).

### A. `Posix` binding additions + arbitrary addressing (world/61c)

- Bindings: `sendto`, `recvfrom`, `send`, `recv`, `setsockopt`, `getsockopt`,
  `shutdown`, `inet_pton`/`inet_ntop` (or pack octets directly — chosen:
  direct, no variadics, no string marshaling on the hot path).
- `NativeBuffer sockaddrInAddr:a:b:c:d:port:` — the general form the
  loopback packer was a special case of (network-order octets + port).
- Socket constants as class-side methods (domains AF_INET/AF_INET6/AF_UNIX,
  types STREAM/DGRAM/RAW, protocols, option levels/names) — named, not
  magic numbers at call sites.

### B. Socket classes (world/61c)

- **`InetAddress`** — an IPv4 address: `InetAddress fromString: '127.0.0.1'`
  or `a:b:c:d:`, `loopback`, `any`; packs itself network-order into a
  sockaddr, formats back to a dotted string. Values, immutable.
- **`Socket`** — the general fd wrapper: `Socket domain:type:protocol:`
  (the escape hatch for any combination, incl. `SOCK_RAW` — returns a
  Socket wrapping a negative fd on EPERM, `isOpen` false, errno readable),
  plus `bindTo:port:`, `connectTo:port:`, `send:`, `receiveInto:`,
  `sendTo:port:data:` (sendto), `receive` (recvfrom → `{bytes. from. port}`),
  `option:level:name:` get/set, `shutdown:`, `close`, `fd` (for handing to
  an IoWorker). Every op answers a value; errors are the return code +
  `Posix errno`, never a raise — the world/61 discipline.
- **`UdpSocket`** — `SOCK_DGRAM` sugar over `Socket`: `UdpSocket bound` (an
  ephemeral loopback server), `sendData:toHost:port:`, `receive`.
- **`IcmpSocket`** — the unprivileged `SOCK_DGRAM`+`IPPROTO_ICMP` socket
  with the internet-checksum helper and echo-request builder; `Ping`
  (world/61d, the demo) is the worked example.

### C. IoWorker integration — the async path is first-class

The blocking `receive`/`receiveReply` above are a low-level ESCAPE HATCH
(fine for a one-shot; they work in bare `macvm run`, which has no workers).
The proper path rides the IoWorker, non-blocking, exactly like TCP does —
and the substrate gap that made the first cut block is closed here.

The IoWorker's original `drainData:` does a `read`, which drops the peer on
an unconnected datagram socket — so datagram servers can't answer their
sender through it. **Fixed, not worked around:** the IoWorker gains a
`recvfrom`-based **datagram drain** (`watchDatagramFd:`/`drainDatagram:`)
parallel to its accept drain, delivering `{#datagram. fd. bytes. octets.
port}` tuples, and a primary-side `watchDatagram:onPacket:` firing a
`[:bytes :octets :port]` continuation. The socket layer wraps it as
`aSocket via: iow onPacket: [:bytes :from :port | ...]` (`from` an
`InetAddress`), and `IcmpSocket via: iow onReply: [:type :code :seq | ...]`
parses the IP+ICMP reply. So a UDP/ICMP server multiplexes non-blocking
with full peer info, no VM ever sleeps in `recvfrom`, and pinging a real
host never freezes the GUI. (The IoWorker itself stays free of the
world/61c socket globals — it passes raw octets, the socket layer rebuilds
the address — since 61c loads after it.)

Requires the embedded worker context (GUI / `VmHandle`), like every
IoWorker path; bare `macvm run` has no workers, so there the blocking
escape hatch is what's available.

## The demo — `Ping` (world/61d)

A real unprivileged ICMP ping to 127.0.0.1 (loopback: on-machine, no
firewall prompt, no network egress): craft the echo request, compute the
one's-complement checksum in Smalltalk, `sendto`, `recvfrom`, skip the IP
header, confirm the echo-reply type, report round-trip time. "Granular
control over our own protocols" made concrete — every byte of the packet
authored in the language. A `Ping host: '127.0.0.1' count: 4` prints the
classic per-echo lines.

## What NOT to do

- **Don't pretend `SOCK_RAW` is unprivileged.** It EPERMs; the constructor
  returns that as a value and the docs say so. The ICMP-DGRAM path is what
  delivers the capability without root.
- **Don't route datagram servers through the IoWorker's read path** and
  silently lose the peer — that boundary is explicit above.
- **Don't leave loopback-only.** Unlike the TCP conveniences (deliberately
  127.0.0.1 to dodge the firewall prompt), the general `Socket`/`InetAddress`
  accept any address — the point is granular control. The TEST suite stays
  loopback-only (headless, no prompt, no egress); real addresses are the
  user's to use.

## Cross-references

- `docs/FFI.md` — fixed-arity contract; sendto/recvfrom/setsockopt all fit.
- `world/61_posix_io.mst` — the socket bindings + `sockaddr_in` packing +
  the errors-as-values / GC-stable-buffer conventions this extends.
- `docs/asyncio_design.md` — the IoWorker/kqueue readiness engine these
  sockets multiplex through.
