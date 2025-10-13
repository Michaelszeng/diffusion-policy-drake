import fcntl
import os
from contextlib import contextmanager


@contextmanager
def locked_open(file_path: str, mode: str = "wb"):
    """Context manager that opens *file_path* and obtains an exclusive advisory
    lock for the duration of the *with* block.

    Parameters
    ----------
    file_path : str
        Path to the file that will be opened.
    mode : str, default "wb"
        Mode string forwarded to :pyfunc:`open`.
    """

    # Ensure directory exists; the file may not yet exist.
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode) as fh:
        # Acquire exclusive lock
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield fh
            # Ensure contents are flushed to disk before releasing the lock
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            # Release lock regardless of success or failure
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _temp_path(original_path: str) -> str:
    """Return a temporary path in the same directory as *original_path*."""
    directory, filename = os.path.split(original_path)
    return os.path.join(directory, f".{filename}.tmp")


def write_atomic(file_path: str, data: str, mode: str = "w") -> None:
    """Atomically write *data* to *file_path*.

    The content is first written to a temporary file in the same directory
    (to guarantee that os.replace is atomic on POSIX) and then moved over the
    destination. This prevents readers from observing a partially-written or
    truncated file.
    """

    temp_path = _temp_path(file_path)
    # Write to temp file with locking
    with locked_open(temp_path, mode) as fh:
        fh.write(data)
    # Atomically replace target
    os.replace(temp_path, file_path)
