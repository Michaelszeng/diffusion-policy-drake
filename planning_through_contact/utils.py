import fcntl
import os
import tempfile
from contextlib import contextmanager

@contextmanager
def file_lock(path: str):
    """
    Context-manager that takes an exclusive advisory lock on *path*.

    All other processes that try to lock the same path (with this same helper
    or any `flock(LOCK_EX)`) will block until the lock is released.
    Works across different machines provided the underlying filesystem
    (NFS, Lustre, GPFS, …) supports POSIX locks—as SuperCloud's /home does.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT)   # O_CREAT lets us lock a fresh .lock file
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def write_atomic(file_path: str, data: str, mode: str = "w") -> None:
    """
    Atomically write *data* to *file_path*.

    The content is first written to a temporary file in the same directory
    (to guarantee that os.replace is atomic on POSIX) and then moved over the
    destination. This prevents readers from observing a partially-written or
    truncated file.
    """
    directory = os.path.dirname(file_path) or "."
    
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Create a unique temporary file in the same directory
    # The temp file gets a unique name per process/invocation to avoid races
    fd = None
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_", text=(mode == "w"))
        
        # Write data to the temp file
        with os.fdopen(fd, mode) as fh:
            fd = None  # fdopen takes ownership
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        
        # Atomically replace target
        os.replace(temp_path, file_path)
        temp_path = None  # Successfully moved
    finally:
        # Clean up temp file if something went wrong
        if fd is not None:
            os.close(fd)
        if temp_path is not None and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
