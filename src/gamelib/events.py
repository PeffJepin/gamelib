import threading
import time

from collections import defaultdict
from multiprocessing.connection import PipeConnection
from typing import Dict, Type, List, Callable


class Event:
    pass


class MessageBus:
    def __init__(self, initial_handlers: Dict[Type[Event], List[Callable]] = None):
        self.handlers = defaultdict(list)
        if initial_handlers:
            for k, list_ in initial_handlers.items():
                self.handlers[k].extend(list_)
        self._adapters = dict()

    def register(self, event_type: Type[Event], callback: Callable):
        self.handlers[event_type].append(callback)

    def unregister(self, event_type: Type[Event], callback: Callable):
        self.handlers[event_type].remove(callback)

    def handle(self, event: Event):
        for handler in self.handlers[type(event)]:
            handler(event)

    def service_connection(self, conn: PipeConnection, event_types: List[Type[Event]]):
        adapter = _ConnectionAdapter(self, conn, event_types)
        for type_ in event_types:
            self.register(type_, adapter)
        self._adapters[conn] = adapter

    def stop_connection_service(self, conn: PipeConnection):
        adapter = self._adapters.pop(conn)
        for type_ in adapter.event_types:
            self.unregister(type_, adapter)
        adapter.is_active = False


class _ConnectionAdapter:
    def __init__(self, mb: MessageBus, conn: PipeConnection, event_types: List[Type[Event]], recv_freq: int = 1):
        self.mb = mb
        self.freq = recv_freq
        self.conn = conn
        self.event_types = event_types
        self.is_active = True
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        # TODO:
        #   This will need some smarter Exception handling at some point. What happens if the connection is lost?
        #   Should the program just error out? Or is that okay and this thread can just exit peacefully?
        #   Once logging is set up it should at least be logged.
        try:
            while self.is_active:
                while self.conn.poll(0):
                    self.mb.handle(self.conn.recv())
                time.sleep(self.freq / 1_000)
        except (BrokenPipeError, EOFError):
            self.is_active = False

    def __call__(self, event):
        self.conn.send(event)
