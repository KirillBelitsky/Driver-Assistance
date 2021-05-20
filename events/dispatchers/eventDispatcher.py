from singleton.singletonMeta import SingletonMeta


class EventDispatcher(metaclass=SingletonMeta):

    def __init__(self):
        self.event_objects_map = {}

    def register(self, event, instance):
        instances = self.event_objects_map.get(event)
        if instances is None:
            instances = []
            self.event_objects_map.update({event: instances})

        if not instances.__contains__(instance):
            instances.append(instance)

    def unregister(self, event, instance):
        instances = self.event_objects_map.get(event)
        if instances is None:
            instances = []
            self.event_objects_map.update({event: instances})

        if instances.__contains__(instance):
            instances.remove(instance)

    def dispatch(self, event, data):
        if self.event_objects_map.get(event) is not None:
            for instance in self.event_objects_map.get(event):
                instance.call(data)