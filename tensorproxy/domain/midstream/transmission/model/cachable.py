from dataclasses import dataclass


@dataclass
class Cachable:
    """Класс, инстансы которого содержат кэшированные значения."""

    def invalidate_cache(self):
        """Обновляет все кэшированные значения.

        Метод для переопределения в дочерних классах.
        """
        self._cache_is_valid_ = False
