from dispatchers.eventDispatcher import EventDispatcher
from listeners.refreshUiEventListener import RefreshUiEventListener
from singleton.singletonMeta import SingletonMeta


class RefreshUiEventDispatcher(EventDispatcher, metaclass=SingletonMeta):

    def __init__(self):
        self.instances = []

    def register(self, instance):
        if isinstance(instance, RefreshUiEventListener):
            if not self.instances.__contains__(instance):
                self.instances.append(instance)

    def unregister(self, instance):
        if isinstance(instance, RefreshUiEventListener):
            if self.instances.__contains__(instance):
                self.instances.remove(instance)

    def dispatch(self, data):
        for instance in self.instances:
            instance.call(data)