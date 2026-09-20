"""Product database: product marker ID -> product name -> target shelf ID."""
from dataclasses import dataclass
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class Product:
    marker_id: int
    name: str
    shelf_id: str


class ProductManager:
    def __init__(self, products: Iterable[Product]):
        self._by_marker: Dict[int, Product] = {p.marker_id: p for p in products}

    @classmethod
    def from_dict(cls, data: dict) -> "ProductManager":
        return cls(Product(int(marker_id), entry["name"], str(entry["shelf"]))
                   for marker_id, entry in data.items())

    def get_by_marker(self, marker_id: int) -> Optional[Product]:
        return self._by_marker.get(marker_id)

    def resolve(self, query: str) -> Optional[Product]:
        """Look a product up by marker ID ("102") or by name ("product b")."""
        query = str(query).strip()
        if query.isdigit():
            return self._by_marker.get(int(query))
        for product in self._by_marker.values():
            if product.name.lower() == query.lower():
                return product
        return None

    def all(self):
        return sorted(self._by_marker.values(), key=lambda p: p.marker_id)
