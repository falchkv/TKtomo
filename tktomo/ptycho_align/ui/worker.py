"""QThread wrapper around :class:`AlignmentEngine`.

All heavy computation -- reconstruction, reprojection, registration -- happens here,
never on the GUI thread. The engine itself stays Qt-free; this is the only place the
two meet.

Cancellation is cooperative: :meth:`AlignmentEngine.run` checks the event *between*
outer iterations, so Stop always finishes the iteration in flight and leaves a
complete, valid state rather than a half-updated one.
"""

from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, Qt, QThread, Signal

from tktomo.ptycho_align.core.engine import AlignmentEngine
from tktomo.ptycho_align.core.state import IterationResult


class AlignmentWorker(QObject):
    """Runs ``n`` outer iterations on a worker thread, reporting as it goes."""

    iteration_started = Signal(int)
    iteration_finished = Signal(object)  # IterationResult
    progress = Signal(float, str)  # fraction 0..1, message
    run_finished = Signal()
    failed = Signal(str)

    def __init__(self, engine: AlignmentEngine, n_iterations: int) -> None:
        super().__init__()
        self._engine = engine
        self._n = n_iterations
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Ask the run to stop after the current iteration completes."""
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(self) -> None:
        completed = 0
        try:
            for index in range(self._n):
                if self._cancel.is_set():
                    break

                self.iteration_started.emit(self._engine.iteration + 1)
                self.progress.emit(
                    index / self._n,
                    f"Iteration {self._engine.iteration + 1} of "
                    f"{self._engine.iteration + self._n - index}...",
                )

                result: IterationResult = self._engine.step()
                completed += 1

                self.iteration_finished.emit(result)
                self.progress.emit(
                    completed / self._n,
                    f"Iteration {result.iteration}: shift RMS {result.error:.4f} px, "
                    f"residual {result.residual:.4f}",
                )
        except Exception:  # a failed run must not take the app down with it
            self.failed.emit(traceback.format_exc())
        finally:
            self.run_finished.emit()


class AlignmentRun:
    """Owns the worker + its QThread and keeps both alive for the run's duration.

    Qt deletes a QThread that goes out of scope while running, so the caller must
    hold on to this object (the main window keeps it in ``self._run``).
    """

    def __init__(self, engine: AlignmentEngine, n_iterations: int) -> None:
        self.thread = QThread()
        self.worker = AlignmentWorker(engine, n_iterations)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        # DirectConnection matters: the QThread object lives in the *main* thread, so a
        # queued quit() would only run when the main thread next pumps events -- and
        # anyone calling wait() has blocked the main thread precisely so it cannot.
        # That deadlocks until the timeout (on window close, and in the tests). Calling
        # quit() directly from the worker thread is safe and returns immediately.
        self.worker.run_finished.connect(self.thread.quit, Qt.DirectConnection)

    def start(self) -> None:
        self.thread.start()

    def cancel(self) -> None:
        self.worker.cancel()

    def wait(self, timeout_ms: int = 30_000) -> bool:
        """Block until the thread exits. Used on window close and by the tests."""
        return self.thread.wait(timeout_ms)

    @property
    def running(self) -> bool:
        return self.thread.isRunning()
