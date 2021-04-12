class EventDispatcher:
    """Function interface to dispatch different events."""

    def register(self, instance):
        """Register an listener to dispatch it later."""
        pass

    def unregister(self, instance):
        """Unregister an listener."""
        pass

    def dispatch(self, data):
        """Dispatch data all registered listeners."""
        pass
