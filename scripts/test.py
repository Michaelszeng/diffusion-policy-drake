import os
import fcntl
import tempfile
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


# Create a temporary file that will act as our lock target
with tempfile.NamedTemporaryFile(delete=False) as tmp:
    path = tmp.name

try:
    # First: acquire lock using file_lock
    with file_lock(path):
        # While locked, try to grab the same lock in non-blocking mode
        fd = os.open(path, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # If we get here, the lock was re-acquired — locking FAILED
            print(False)
        except BlockingIOError:
            # Expected: lock is held by file_lock, second attempt blocked → locking WORKS
            print(True)
        finally:
            os.close(fd)
finally:
    # Clean up temp file
    os.unlink(path)
