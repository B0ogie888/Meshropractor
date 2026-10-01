"""Control display quality without changing model geometry."""
from PySide6.QtCore import QObject


class ViewportPerformance(QObject):
    def __init__(self, plotter):
        super().__init__(plotter)
        self.plotter = plotter
        self.saved = []
        self.interactive_edges = True
        self._quality = None
        self.interactor = plotter.iren.interactor
        self.observers = [self.interactor.AddObserver('StartInteractionEvent', self.begin),
                          self.interactor.AddObserver('EndInteractionEvent', self.end)]

    def configure(self, *, anti_aliasing='msaa', samples=4, interactive_edges=True):
        if anti_aliasing not in ('none', 'fxaa', 'msaa'):
            raise ValueError('Unsupported anti-aliasing mode')
        if samples not in (4, 8):
            raise ValueError('MSAA samples must be 4 or 8')
        quality = (anti_aliasing, samples)
        if quality != self._quality:
            # PyVista's MSAA setter alone does not disable a previous FXAA pass.
            self.plotter.disable_anti_aliasing()
            if anti_aliasing != 'none':
                self.plotter.enable_anti_aliasing(anti_aliasing, multi_samples=samples)
            self._quality = quality
        self.interactive_edges = bool(interactive_edges)
        if not self.interactive_edges:
            self.end()

    def begin(self, *_):
        if self.saved or not self.interactive_edges: return
        for actor in self.plotter.actors.values():
            if not hasattr(actor, 'GetMapper') or not hasattr(actor, 'GetProperty'): continue
            mapper = actor.GetMapper()
            data = mapper.GetInput() if mapper is not None and hasattr(mapper, 'GetInput') else None
            prop = actor.GetProperty()
            if (data is not None and hasattr(data, 'GetNumberOfCells') and data.GetNumberOfCells() >= 100_000
                    and hasattr(prop, 'GetEdgeVisibility') and prop.GetEdgeVisibility() and actor.GetVisibility()):
                self.saved.append(prop)
                prop.SetEdgeVisibility(False)

    def end(self, *_):
        changed = bool(self.saved)
        for prop in self.saved: prop.SetEdgeVisibility(True)
        self.saved.clear()
        if changed: self.plotter.render()

    def dispose(self):
        for observer in self.observers: self.interactor.RemoveObserver(observer)
        self.observers.clear()
        self.end()
