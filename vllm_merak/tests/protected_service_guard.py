"""Read-only snapshot and isolation checks for hybrid-padding service tests.

This helper never opens an HTTP connection and never sends a signal.  Its
only subprocesses are fixed read-only inspection commands (``ss``, ``ps``,
and ``nvidia-smi``).  It is intentionally kept under tests so production
runtime code cannot depend on deployment-host assumptions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from typing import Callable, Sequence


PROTECTED_PORTS = frozenset({18006, 18036})
PROTECTED_GPUS = frozenset({0, 1, 6})


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    pgid: int
    start_time: str


@dataclass(frozen=True)
class Listener:
    port: int
    process: ProcessIdentity


@dataclass(frozen=True)
class GPUProcess:
    gpu_index: int
    gpu_uuid: str
    pid: int
    pgid: int


@dataclass(frozen=True)
class HostSnapshot:
    listeners: tuple[Listener, ...]
    gpu_processes: tuple[GPUProcess, ...]


CommandRunner = Callable[[Sequence[str]], str]


def _run_read_only(command: Sequence[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def parse_ss_listeners(output: str) -> dict[int, int]:
    listeners: dict[int, int] = {}
    for line in output.splitlines():
        port_match = re.search(r"(?:\]|\d|\*):(\d+)\s", line)
        pid_match = re.search(r"pid=(\d+)", line)
        if port_match and pid_match:
            listeners[int(port_match.group(1))] = int(pid_match.group(1))
    return listeners


def parse_gpu_uuid_indexes(output: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        index, uuid = (item.strip() for item in line.split(",", maxsplit=1))
        result[uuid] = int(index)
    return result


def parse_compute_apps(
    output: str, uuid_indexes: dict[str, int]
) -> list[tuple[int, str, int]]:
    result: list[tuple[int, str, int]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        uuid, pid = (item.strip() for item in line.split(",", maxsplit=1))
        if uuid not in uuid_indexes:
            raise RuntimeError(f"compute app references unknown GPU UUID {uuid}")
        result.append((uuid_indexes[uuid], uuid, int(pid)))
    return result


def _process_identity(pid: int, run: CommandRunner) -> ProcessIdentity:
    output = run(["ps", "-p", str(pid), "-o", "pid=,pgid=,lstart="]).strip()
    fields = output.split(maxsplit=2)
    if len(fields) != 3 or int(fields[0]) != pid:
        raise RuntimeError(f"could not capture stable process identity for PID {pid}")
    return ProcessIdentity(pid=pid, pgid=int(fields[1]), start_time=fields[2])


def capture_snapshot(run: CommandRunner = _run_read_only) -> HostSnapshot:
    listener_pids = parse_ss_listeners(run(["ss", "-H", "-ltnp"]))
    listeners = tuple(
        Listener(port=port, process=_process_identity(pid, run))
        for port, pid in sorted(listener_pids.items())
    )

    uuid_indexes = parse_gpu_uuid_indexes(
        run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ]
        )
    )
    apps = parse_compute_apps(
        run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ]
        ),
        uuid_indexes,
    )
    gpu_processes = tuple(
        GPUProcess(
            gpu_index=index,
            gpu_uuid=uuid,
            pid=pid,
            pgid=_process_identity(pid, run).pgid,
        )
        for index, uuid, pid in apps
    )
    return HostSnapshot(listeners=listeners, gpu_processes=gpu_processes)


def assert_candidate_isolated(
    snapshot: HostSnapshot,
    *,
    port: int,
    gpu_indexes: Sequence[int],
) -> None:
    if port in PROTECTED_PORTS:
        raise ValueError(f"candidate port {port} is protected")
    protected = sorted(PROTECTED_GPUS.intersection(gpu_indexes))
    if protected:
        raise ValueError(f"candidate GPUs overlap protected GPUs: {protected}")
    listening_ports = {listener.port for listener in snapshot.listeners}
    if port in listening_ports:
        raise RuntimeError(f"candidate port {port} is already listening")
    busy_gpus = sorted(
        {process.gpu_index for process in snapshot.gpu_processes}.intersection(
            gpu_indexes
        )
    )
    if busy_gpus:
        raise RuntimeError(
            f"candidate GPUs already have compute processes: {busy_gpus}"
        )


def assert_protected_unchanged(before: HostSnapshot, after: HostSnapshot) -> None:
    def protected_listeners(snapshot: HostSnapshot) -> dict[int, ProcessIdentity]:
        return {
            listener.port: listener.process
            for listener in snapshot.listeners
            if listener.port in PROTECTED_PORTS
        }

    before_listeners = protected_listeners(before)
    after_listeners = protected_listeners(after)
    if before_listeners != after_listeners:
        raise RuntimeError(
            "protected listener identity changed: "
            f"before={before_listeners}, after={after_listeners}"
        )

    def protected_owners(snapshot: HostSnapshot) -> set[tuple[int, int, int]]:
        return {
            (process.gpu_index, process.pid, process.pgid)
            for process in snapshot.gpu_processes
            if process.gpu_index in PROTECTED_GPUS
        }

    before_owners = protected_owners(before)
    after_owners = protected_owners(after)
    if before_owners != after_owners:
        raise RuntimeError(
            "protected GPU ownership changed: "
            f"before={sorted(before_owners)}, after={sorted(after_owners)}"
        )


def snapshot_to_json(snapshot: HostSnapshot) -> str:
    return json.dumps(asdict(snapshot), indent=2, sort_keys=True)


def snapshot_from_json(payload: str) -> HostSnapshot:
    raw = json.loads(payload)
    listeners = tuple(
        Listener(
            port=int(listener["port"]),
            process=ProcessIdentity(
                pid=int(listener["process"]["pid"]),
                pgid=int(listener["process"]["pgid"]),
                start_time=str(listener["process"]["start_time"]),
            ),
        )
        for listener in raw["listeners"]
    )
    gpu_processes = tuple(
        GPUProcess(
            gpu_index=int(process["gpu_index"]),
            gpu_uuid=str(process["gpu_uuid"]),
            pid=int(process["pid"]),
            pgid=int(process["pgid"]),
        )
        for process in raw["gpu_processes"]
    )
    return HostSnapshot(listeners=listeners, gpu_processes=gpu_processes)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="optional JSON output path")
    parser.add_argument(
        "--compare",
        metavar="BEFORE_JSON",
        help="fail unless protected listeners and GPU owners match a prior snapshot",
    )
    parser.add_argument(
        "--candidate-port", type=int, help="fail unless this candidate port is isolated"
    )
    parser.add_argument(
        "--candidate-gpus",
        help="comma-separated GPU indexes paired with --candidate-port",
    )
    args = parser.parse_args()
    snapshot = capture_snapshot()
    if (args.candidate_port is None) != (args.candidate_gpus is None):
        parser.error("--candidate-port and --candidate-gpus must be supplied together")
    if args.candidate_port is not None:
        try:
            gpu_indexes = [int(item) for item in args.candidate_gpus.split(",")]
        except ValueError as exc:
            parser.error(f"invalid --candidate-gpus: {exc}")
        assert_candidate_isolated(
            snapshot,
            port=args.candidate_port,
            gpu_indexes=gpu_indexes,
        )
    if args.compare:
        with open(args.compare, encoding="utf-8") as before_file:
            before = snapshot_from_json(before_file.read())
        assert_protected_unchanged(before, snapshot)

    payload = snapshot_to_json(snapshot)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(payload + os.linesep)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
