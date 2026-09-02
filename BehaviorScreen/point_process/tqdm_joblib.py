import contextlib
from typing import Iterator, Protocol

import joblib
import joblib.parallel

_ORIGINAL_BATCH_COMPLETION_CALLBACK = joblib.parallel.BatchCompletionCallBack

class _TqdmLike(Protocol):
    def update(self, n: int = ...) -> None: ...
    def close(self) -> None: ...


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: _TqdmLike) -> Iterator[_TqdmLike]:

    previous_callback = joblib.parallel.BatchCompletionCallBack  # for LIFO restore

    class _TqdmBatchCompletionCallback(_ORIGINAL_BATCH_COMPLETION_CALLBACK):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = _TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = previous_callback
        tqdm_object.close()