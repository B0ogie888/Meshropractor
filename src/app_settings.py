"""One settings store, preserving preferences used before the 0.3 UI merge."""
from PySide6.QtCore import QSettings


def load_settings(factory=QSettings):
    settings = factory('MeshropractorTeam', 'Meshropractor')
    marker = 'migration/unified_0_3'
    if not settings.contains(marker):
        previous = factory('MeshropractorTeam', 'MeshropractorNew')
        for key in previous.allKeys():
            if key.startswith(('migration/', 'new_ui/')):
                continue
            if settings.contains(key):
                settings.setValue('migration/before_0_3/' + key, settings.value(key))
            settings.setValue(key, previous.value(key))
        settings.setValue(marker, True)
        settings.sync()
    return settings
