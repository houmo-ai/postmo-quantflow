"""Owned staging directories, output locking, and no-replace atomic publication."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import stat as stat_module
import uuid
from pathlib import Path

from xhmodel_merak.errors import ErrorCode, ModelError

from .contracts import BuildPlan


def reject_artifact_symlinks(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Published Artifact root must be a regular directory",
            symlinks=["."] if root.is_symlink() else [],
            root=str(root),
        )
    symlinks: list[str] = []
    special_nodes: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        try:
            mode = path.lstat().st_mode
        except OSError:
            special_nodes.append(relative)
            continue
        if stat_module.S_ISLNK(mode):
            symlinks.append(relative)
        elif not (stat_module.S_ISREG(mode) or stat_module.S_ISDIR(mode)):
            special_nodes.append(relative)
    if symlinks:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Published Artifacts must not contain symlinks",
            symlinks=sorted(symlinks),
        )
    if special_nodes:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Published Artifacts must contain only regular files and directories",
            special_nodes=sorted(special_nodes),
        )


def path_has_symlink(root: Path, candidate: Path) -> bool:
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def publish(
    stage_path: Path,
    output_path: Path,
    stage_identity: tuple[int, int, int] | None = None,
) -> None:
    if stage_identity is not None and not owned_directory_matches(stage_path, stage_identity):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Build stage directory was replaced before publication",
            stage_dir=str(stage_path),
        )
    reject_artifact_symlinks(stage_path)
    if output_path.exists() or output_path.is_symlink():
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"output_dir appeared during build and will not be overwritten: {output_path}",
            output_dir=str(output_path),
        )
    rename_noreplace(stage_path, output_path)
    if stage_identity is not None and not owned_directory_matches(output_path, stage_identity):
        try:
            rename_noreplace(output_path, stage_path)
        except (ModelError, OSError):
            pass
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Build stage directory changed during publication",
            stage_dir=str(stage_path),
            output_dir=str(output_path),
        )


def rename_noreplace(source: Path, destination: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Atomic no-replace publication is unavailable on this platform",
            output_dir=str(destination),
        )
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"output_dir appeared during build and will not be overwritten: {destination}",
            output_dir=str(destination),
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def acquire_output_lock(output_path: Path) -> tuple[Path, int, tuple[int, int]]:
    if output_path.exists() or output_path.is_symlink():
        raise ModelError(
            ErrorCode.INVALID_PARAMETER,
            f"output_dir must not already exist: {output_path}",
            output_dir=str(output_path),
        )
    lock_path = output_path.parent / f".{output_path.name}.xhbuild.lock"
    descriptor: int | None = None
    locked = False
    try:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            descriptor = os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW)
        descriptor_stat = os.fstat(descriptor)
        if (
            not stat_module.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_nlink != 1
            or descriptor_stat.st_uid != os.getuid()
        ):
            raise OSError(errno.EPERM, "unsafe output lock file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        pathname_stat = lock_path.stat(follow_symlinks=False)
        identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
        if not stat_module.S_ISREG(pathname_stat.st_mode) or (pathname_stat.st_dev, pathname_stat.st_ino) != identity:
            raise OSError(errno.ESTALE, "output lock pathname changed while it was being acquired")
        if output_path.exists() or output_path.is_symlink():
            raise ModelError(
                ErrorCode.INVALID_PARAMETER,
                f"output_dir must not already exist: {output_path}",
                output_dir=str(output_path),
            )
        payload = f"pid={os.getpid()}\n".encode("ascii")
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.write(descriptor, payload) != len(payload):
            raise OSError(errno.EIO, "short write while recording output lock owner")
        return lock_path, descriptor, identity
    except BaseException as exc:
        if descriptor is not None:
            if locked:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except BaseException:
                    pass
            try:
                os.close(descriptor)
            except OSError:
                pass
        if isinstance(exc, ModelError):
            raise
        if isinstance(exc, OSError):
            raise ModelError(
                ErrorCode.INVALID_PARAMETER,
                f"Another build is already publishing output_dir: {output_path}",
                output_dir=str(output_path),
                lock_file=str(lock_path),
            ) from exc
        raise


def release_output_lock(lock: tuple[Path, int, tuple[int, int]]) -> None:
    _path, descriptor, _identity = lock
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def reserve_owned_directory(path: Path, plan: BuildPlan, field: str) -> tuple[int, int, int]:
    try:
        path.mkdir(mode=0o700, parents=True)
    except FileExistsError as exc:
        raise ModelError(
            ErrorCode.INVALID_PARAMETER,
            f"{field} already exists: {path}",
            model=plan.model_id,
            field=field,
            path=str(path),
        ) from exc
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    info = os.fstat(descriptor)
    if not stat_module.S_ISDIR(info.st_mode):
        os.close(descriptor)
        raise ModelError(
            ErrorCode.INVALID_PARAMETER,
            f"{field} could not be reserved as a regular directory: {path}",
            model=plan.model_id,
            field=field,
            path=str(path),
        )
    return info.st_dev, info.st_ino, descriptor


def owned_directory_matches(path: Path, identity: tuple[int, int, int] | None) -> bool:
    if identity is None:
        return False
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return (
        stat_module.S_ISDIR(info.st_mode)
        and not stat_module.S_ISLNK(info.st_mode)
        and (
            info.st_dev,
            info.st_ino,
        )
        == identity[:2]
    )


def require_owned_directory(
    path: Path,
    identity: tuple[int, int, int] | None,
    plan: BuildPlan,
    field: str,
) -> None:
    if not owned_directory_matches(path, identity):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"Reserved {field} was replaced during the build",
            model=plan.model_id,
            field=field,
            path=str(path),
        )


def remove_owned_directory(path: Path, identity: tuple[int, int, int] | None) -> None:
    try:
        if identity is None:
            return
        # Rename first and inspect the moved inode so replacement cannot race a check-then-rmtree sequence.
        quarantine = unused_sibling(path, "cleanup")
        try:
            rename_noreplace(path, quarantine)
        except (ModelError, OSError):
            return
        if not owned_directory_matches(quarantine, identity):
            try:
                rename_noreplace(quarantine, path)
            except (ModelError, OSError):
                pass
            return

        clear_owned_directory(identity[2])
        if owned_directory_matches(quarantine, identity):
            try:
                quarantine.rmdir()
            except OSError:
                pass
    finally:
        release_owned_directory(identity)


def clear_owned_directory(descriptor: int) -> None:
    try:
        names = os.listdir(descriptor)
    except OSError:
        return
    for name in names:
        try:
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError:
            continue
        if not stat_module.S_ISDIR(info.st_mode):
            try:
                os.unlink(name, dir_fd=descriptor)
            except OSError:
                pass
            continue

        try:
            child_descriptor = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
        except OSError:
            continue
        try:
            child_info = os.fstat(child_descriptor)
            if (child_info.st_dev, child_info.st_ino) != (info.st_dev, info.st_ino):
                continue
            clear_owned_directory(child_descriptor)
        finally:
            os.close(child_descriptor)

        try:
            current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat_module.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) == (
                info.st_dev,
                info.st_ino,
            ):
                os.rmdir(name, dir_fd=descriptor)
        except OSError:
            pass


def release_owned_directory(identity: tuple[int, int, int] | None) -> None:
    if identity is not None:
        os.close(identity[2])


def unused_sibling(output_path: Path, purpose: str) -> Path:
    while True:
        candidate = output_path.parent / f".{output_path.name}.{purpose}-{uuid.uuid4().hex}"
        if not candidate.exists():
            return candidate
