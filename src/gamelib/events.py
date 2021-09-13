from collections import defaultdict
from typing import Dict, Type, List, Callable


class Event:
    pass


class MessageBus:
    handlers: Dict[Type[Event], List[Callable]] = defaultdict(list)

    def register(self, event_type: Type[Event], callback: Callable):
        self.handlers[event_type].append(callback)

    def unregister(self, event_type: Type[Event], callback: Callable):
        self.handlers[event_type].remove(callback)

    def handle(self, event: Event):
        for handler in self.handlers[type(event)]:
            handler(event)
