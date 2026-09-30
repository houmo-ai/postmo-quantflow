import pytest

from protected_service_guard import (
    GPUProcess,
    HostSnapshot,
    Listener,
    ProcessIdentity,
    assert_candidate_isolated,
    assert_protected_unchanged,
    parse_compute_apps,
    parse_gpu_uuid_indexes,
    parse_ss_listeners,
    snapshot_from_json,
    snapshot_to_json,
)


def _identity(
    pid: int, pgid: int, start: str = "Fri Jul 10 21:30:34 2026"
) -> ProcessIdentity:
    return ProcessIdentity(pid=pid, pgid=pgid, start_time=start)


def _protected_snapshot() -> HostSnapshot:
    return HostSnapshot(
        listeners=(
            Listener(port=18036, process=_identity(1_958_132, 1_958_132)),
            Listener(port=18006, process=_identity(3_913_386, 3_913_386)),
        ),
        gpu_processes=(
            GPUProcess(0, "GPU-0", 1_960_404, 1_958_132),
            GPUProcess(1, "GPU-1", 1_960_404, 1_958_132),
            GPUProcess(6, "GPU-6", 3_914_832, 3_913_386),
        ),
    )


def test_snapshot_parsers_preserve_port_pid_and_gpu_uuid_mapping() -> None:
    ss_output = """
LISTEN 0 2048 127.0.0.1:18036 0.0.0.0:* users:((\"python\",pid=1958132,fd=59))
LISTEN 0 2048 [::1]:18006 [::]:* users:((\"python\",pid=3913386,fd=51))
"""
    uuid_output = "0, GPU-zero\n7, GPU-seven\n"
    apps_output = "GPU-zero, 100\nGPU-seven, 700\n"

    assert parse_ss_listeners(ss_output) == {18036: 1_958_132, 18006: 3_913_386}
    indexes = parse_gpu_uuid_indexes(uuid_output)
    assert indexes == {"GPU-zero": 0, "GPU-seven": 7}
    assert parse_compute_apps(apps_output, indexes) == [
        (0, "GPU-zero", 100),
        (7, "GPU-seven", 700),
    ]


def test_candidate_isolation_accepts_only_free_non_protected_resources() -> None:
    snapshot = _protected_snapshot()

    assert_candidate_isolated(snapshot, port=18136, gpu_indexes=[7])
    assert_candidate_isolated(snapshot, port=18137, gpu_indexes=[2, 3])


@pytest.mark.parametrize(
    ("port", "gpus", "message"),
    [
        (18036, [7], "port 18036 is protected"),
        (18136, [0, 7], "overlap protected GPUs"),
    ],
)
def test_candidate_isolation_rejects_protected_resources(port, gpus, message) -> None:
    with pytest.raises(ValueError, match=message):
        assert_candidate_isolated(_protected_snapshot(), port=port, gpu_indexes=gpus)


def test_candidate_isolation_rejects_busy_candidate_without_touching_it() -> None:
    base = _protected_snapshot()
    busy = HostSnapshot(
        listeners=base.listeners + (Listener(18136, _identity(999, 999)),),
        gpu_processes=base.gpu_processes + (GPUProcess(7, "GPU-7", 777, 777),),
    )

    with pytest.raises(RuntimeError, match="port 18136 is already listening"):
        assert_candidate_isolated(busy, port=18136, gpu_indexes=[7])

    no_port_conflict = HostSnapshot(
        listeners=base.listeners,
        gpu_processes=busy.gpu_processes,
    )
    with pytest.raises(
        RuntimeError, match=r"GPUs already have compute processes: \[7\]"
    ):
        assert_candidate_isolated(no_port_conflict, port=18136, gpu_indexes=[7])


def test_protected_before_after_snapshot_requires_stable_identity_and_gpu_ownership() -> (
    None
):
    before = _protected_snapshot()
    assert_protected_unchanged(before, before)

    restarted = HostSnapshot(
        listeners=(
            Listener(
                18036, _identity(1_958_132, 1_958_132, "Sat Jul 11 01:00:00 2026")
            ),
            before.listeners[1],
        ),
        gpu_processes=before.gpu_processes,
    )
    with pytest.raises(RuntimeError, match="protected listener identity changed"):
        assert_protected_unchanged(before, restarted)

    moved = HostSnapshot(
        listeners=before.listeners,
        gpu_processes=before.gpu_processes[:-1],
    )
    with pytest.raises(RuntimeError, match="protected GPU ownership changed"):
        assert_protected_unchanged(before, moved)


def test_snapshot_json_round_trip_preserves_process_start_identity() -> None:
    before = _protected_snapshot()

    after = snapshot_from_json(snapshot_to_json(before))

    assert after == before
    assert_protected_unchanged(before, after)
