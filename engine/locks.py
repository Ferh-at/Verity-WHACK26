"""Process locks for the shared journals, including Windows checkouts."""
from contextlib import contextmanager
import errno
import os

if os.name == "nt":
    import msvcrt
else:
    import fcntl


@contextmanager
def file_lock(path, *, wait=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as file:
        if os.name == "nt":
            if file.tell() == 0:
                file.write(b"\0")
                file.flush()
            file.seek(0)
            try:
                msvcrt.locking(file.fileno(), msvcrt.LK_LOCK if wait else msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                if not wait and exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise BlockingIOError(exc.errno, str(exc)) from exc
                raise
            try:
                yield
            finally:
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(file, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
            yield
