from __future__ import annotations

import multiprocessing as mp
import traceback
from typing import Type

from . import _EcsGlobals, export_globals, import_globals
from .. import events
from .. import Update, SystemStop


class System:
    """ Responsible for manipulating Component data.

    A System can be run either locally on the main process
    or it can run with a SystemRunner which will run it in a process
    and handle ipc.

    Systems can implement behaviour within the update stub function for
    regular updating behaviour and can also implement eventhandler
    decorated functions to respond to events raised else where in the program.
    """

    def __init__(self, runner=None):
        """ Start the System and signs up to receive appropriate events.

        Parameters
        ----------
        runner : SystemRunner | None
            Only applicable if running in a process. System must have
            a reference to its runner for ipc.
        """
        events.register_marked(self)
        self._runner = runner
        self._running = True

    def update(self):
        """Stub for subclass defined behavior."""

    def stop(self):
        """ Makes sure the System stops handling events when no longer in use. """
        self._running = False
        events.unregister_marked(self)

    def raise_event(self, event, key=None):
        """ Simply posts the event if this System is running locally,
        otherwise passes the event to this systems SystemRunner so it
        can be piped back to the main process.

        Parameters
        ----------
        event : Event
        key : Hashable
        """
        if self._runner is None:
            # main process can just post events
            events.post(event, key)
        else:
            # systems running in a process should delegate to the runner.
            self._runner.outgoing_events.append((event, key))

    @events.eventhandler(Update)
    def _update_handler(self, _):
        self.update()
        self.raise_event(SystemUpdateComplete(type(self)))


class SystemRunner(mp.Process):
    """ Responsible for initializing a System in a new process and handling ipc. """

    def __init__(self, system):
        """ Initialize means of ipc. Must grab reference of global state
        to send to the new process since windows can't fork the process.

        Parameters
        ----------
        system : type[System]
            The System this runner is responsible for.
        """
        super().__init__(target=self.main, args=(export_globals,))
        self._child_conn, self.conn = mp.Pipe()
        self.outgoing_events = []
        self._running = False
        self._system = system

    def start(self):
        """Starts the process and signs up with the events module."""
        super().start()
        handler_types = events.find_eventhandlers(self._system).keys()
        events.service_connection(self.conn, *handler_types)

    def main(self, ecs_globals):
        """ Main function to run in the child process. Handles ipc.

        Parameters
        ----------
        ecs_globals : Any
            values from EcsGlobals. Must be passed over to work
            properly on windows spawned processes.
        """
        import_globals(ecs_globals)
        events.clear_handlers()

        inst = self._system(runner=self)
        self._running = True

        while self._running:
            try:
                self._poll()
            except Exception as e:
                msg_with_traceback = f"{e}\n\n{traceback.format_exc()}"
                self._child_conn.send(type(e)(msg_with_traceback))
                break
        inst.stop()

    def join(self, timeout=None):
        """ Sends the stop code to the child and stops communications before joining. """
        self.conn.send((SystemStop(), None))
        super().join(timeout)
        events.stop_connection_service(self.conn)

    def _poll(self):
        while self.outgoing_events:
            event_key_pair = self.outgoing_events.pop(0)
            self._child_conn.send(event_key_pair)

        if not self._child_conn.poll(0):
            return

        message = self._child_conn.recv()
        (event, key) = message
        events.post(event, key=key)

        if isinstance(event, SystemStop):
            self._running = False


class SystemUpdateComplete(events.Event):
    """
    Posted to the main process after a system finishes handling an Update event.
    """

    __slots__ = ["system"]

    system: Type[System]
