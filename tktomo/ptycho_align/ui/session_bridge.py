"""The only place Qt meets a session.

Sessions emit plain :class:`~tktomo.ptycho_align.session.protocol.Event` objects because
the layer below the window has to import without PySide6 installed. The window wants Qt
signals. This adapter is the whole of the translation, and it deliberately re-emits the
*exact* five signatures the old ``AlignmentWorker`` had -- ``iteration_started(int)``,
``iteration_finished(object)``, ``progress(float, str)``, ``run_finished()`` and
``failed(str)`` -- so the slots already connected to them keep working untouched.

Thread safety: events arrive on the session's compute thread, which is a plain
``threading.Thread``. Emitting a Qt signal from there is safe -- Qt dispatches on the
*receiver's* thread affinity, and this object is created in the GUI thread, so every
connected slot runs on the GUI thread through a queued connection. What is not safe, and
is not done here, is touching a widget from the callback itself.

:meth:`SessionBridge.run_job` is the other half. Loading, preprocessing and COM are
places the window legitimately blocks behind a modal dialog; it spins a nested event loop
rather than pumping ``processEvents`` by hand, so the window stays responsive and Cancel
reaches the job through the same path a remote client would use.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEventLoop, QObject, Qt, Signal
from PySide6.QtWidgets import QProgressDialog, QWidget

from tktomo.ptycho_align.session.protocol import (
    EVENT_ITERATION,
    EVENT_ITERATION_STARTED,
    EVENT_JOB_FAILED,
    EVENT_JOB_FINISHED,
    EVENT_JOB_STARTED,
    EVENT_LOG,
    EVENT_PROGRESS,
    EVENT_RUN_FINISHED,
    EVENT_SUMMARY,
    EVENT_TELEMETRY,
    Event,
    JobFailed,
    JobHandle,
)

__all__ = ["JobCancelled", "SessionBridge"]


class JobCancelled(Exception):
    """The user cancelled a job from its progress dialog. Not an error, so no dialog."""


class SessionBridge(QObject):
    """Turns session events into Qt signals, and blocks on a job when asked to."""

    # Identical to the retired AlignmentWorker, so existing slots need no change.
    iteration_started = Signal(int)
    iteration_finished = Signal(object)  # IterationSummary
    progress = Signal(float, str)  # fraction 0..1, message
    run_finished = Signal()
    failed = Signal(str)

    # New, for what the window used to read straight off the engine.
    summary_changed = Signal(object)  # SessionSummary
    log_message = Signal(str)
    telemetry = Signal(object)  # ResourceSample
    job_started = Signal(str, str)  # job_id, kind
    job_finished = Signal(str, str, object)  # job_id, kind, result
    job_failed = Signal(str, str, str)  # job_id, kind, message

    # Private, and the reason everything above is safe. Events arrive on the compute
    # thread; a signal connected to a *plain function* is delivered directly, on the
    # emitting thread, so re-emitting there would run every slot -- and touch every
    # widget -- off the GUI thread. Bouncing through one queued connection to a bound
    # method of this QObject moves the whole dispatch onto the GUI thread, where Qt's
    # thread affinity rules apply again.
    _event_received = Signal(object)

    def __init__(self, session, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self._event_received.connect(self._dispatch, Qt.QueuedConnection)
        self._unsubscribe = session.subscribe(self._event_received.emit)

    def close(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    # -- event translation ---------------------------------------------------------

    def _dispatch(self, event: Event) -> None:
        """Runs on the GUI thread, one queued hop after the compute thread emitted."""
        kind, payload = event.kind, event.payload

        if kind == EVENT_PROGRESS:
            self.progress.emit(payload["fraction"], payload["message"])
        elif kind == EVENT_ITERATION_STARTED:
            self.iteration_started.emit(payload["iteration"])
        elif kind == EVENT_ITERATION:
            self.iteration_finished.emit(payload["result"])
            self.summary_changed.emit(payload["summary"])
        elif kind == EVENT_RUN_FINISHED:
            self.run_finished.emit()
        elif kind == EVENT_SUMMARY:
            self.summary_changed.emit(payload["summary"])
        elif kind == EVENT_LOG:
            self.log_message.emit(payload.get("message", ""))
        elif kind == EVENT_TELEMETRY:
            self.telemetry.emit(payload.get("sample"))
        elif kind == EVENT_JOB_STARTED:
            self.job_started.emit(event.job_id or "", payload.get("kind", ""))
        elif kind == EVENT_JOB_FINISHED:
            self.job_finished.emit(
                event.job_id or "", payload.get("kind", ""), payload.get("result")
            )
        elif kind == EVENT_JOB_FAILED:
            job_kind = payload.get("kind", "")
            message = payload.get("traceback") or payload.get("message", "")
            self.job_failed.emit(event.job_id or "", job_kind, payload.get("message", ""))
            if job_kind == "start_run":
                # A failed run keeps the old contract: the window shows the traceback.
                self.failed.emit(message)

    # -- blocking on a job ---------------------------------------------------------

    def run_job(
        self,
        handle: JobHandle,
        *,
        title: str,
        label: str,
        parent: QWidget | None = None,
        cancellable: bool = True,
        minimum_duration_ms: int = 300,
    ) -> Any:
        """Block on ``handle`` behind a modal progress dialog; return its result.

        Raises :class:`JobCancelled` if the user cancelled, or the job's
        :class:`~tktomo.ptycho_align.session.protocol.JobFailed` if it raised. Uses a
        nested event loop rather than a ``processEvents`` spin so the dialog repaints and
        no stray input is delivered to the window behind it.
        """
        state = self.session.job_state(handle.job_id)
        if state is not None and state.done:
            return self._outcome(handle)  # finished before we could even show a dialog

        dialog = QProgressDialog(label, "Cancel" if cancellable else "", 0, 100, parent)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumDuration(minimum_duration_ms)  # don't flash up for a fast job
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        if not cancellable:
            dialog.setCancelButton(None)

        loop = QEventLoop()
        cancelled = False

        def on_progress(fraction: float, message: str) -> None:
            dialog.setValue(int(max(0.0, min(1.0, fraction)) * 100))
            dialog.setLabelText(message or label)

        def on_done(job_id: str, *_args) -> None:
            if job_id == handle.job_id:
                loop.quit()

        def on_cancel() -> None:
            nonlocal cancelled
            cancelled = True
            self.session.cancel_job(handle.job_id)

        self.progress.connect(on_progress)
        self.job_finished.connect(on_done)
        self.job_failed.connect(on_done)
        dialog.canceled.connect(on_cancel)
        try:
            # Re-check after connecting: the job may have finished in the window between
            # the check above and the subscription, and then nothing would ever quit the
            # loop.
            state = self.session.job_state(handle.job_id)
            if state is None or not state.done:
                loop.exec()
        finally:
            self.progress.disconnect(on_progress)
            self.job_finished.disconnect(on_done)
            self.job_failed.disconnect(on_done)
            # Disconnect before closing: QProgressDialog treats a close as a cancel and
            # emits `canceled`, so leaving this connected would make every job that
            # finished normally look as though the user had aborted it.
            dialog.canceled.disconnect(on_cancel)
            dialog.close()

        if cancelled:
            raise JobCancelled(handle.kind)
        return self._outcome(handle)

    def _outcome(self, handle: JobHandle) -> Any:
        state = self.session.job_state(handle.job_id)
        if state is None:
            raise JobFailed(f"{handle.kind} left no result")
        if state.error is not None:
            raise state.error
        if state.cancelled:
            raise JobCancelled(handle.kind)
        return state.result
