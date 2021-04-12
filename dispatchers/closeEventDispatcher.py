from dispatchers.eventDispatcher import EventDispatcher
from listeners.closeEventListener import CloseEventListener
from singleton.singletonMeta import SingletonMeta


class CloseEventDispatcher(EventDispatcher, metaclass=SingletonMeta):

    def __init__(self):
        self.instances = []

    def register(self, instance):
        if isinstance(instance, CloseEventListener):
            if not self.instances.__contains__(instance):
                self.instances.append(instance)

    def unregister(self, instance):
        if isinstance(instance, CloseEventListener):
            if self.instances.__contains__(instance):
                self.instances.remove(instance)

    def dispatch(self, data):
        for instance in self.instances:
            instance.call(data)