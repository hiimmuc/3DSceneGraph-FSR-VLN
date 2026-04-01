"""Class to represent a view in a HMSG."""

from memory.hmsg.graph.persistence import EntityPersistence, PersistenceHandler


class View(PersistenceHandler):
    """Class to represent a View/image in a room.

    Args:
        view_id: Unique identifier for the view
        room_id: Identifier of the room this view belongs to
        img_id: Image index of the view in the dataset
        name: Name of the view (optional)
    """

    def __init__(self, view_id: str | int, room_id: str | int, img_id: int, name: str = None):
        """Initialize a View entity.

        Args:
            view_id: Unique identifier for the view
            room_id: Identifier of the room this view belongs to
            img_id: Image index of the view in the dataset
            name: Name of the view (optional)
        """
        self._view_id = view_id
        self._room_id = room_id
        self._img_id = img_id
        self._name = name
        self._object_ids: list = []
        self._img_path: str = None
        self._embedding = None
        self._text_descriptions: list = []

    # Properties for encapsulation (OCP: Open/Closed Principle)
    @property
    def view_id(self):
        return self._view_id

    @property
    def room_id(self):
        return self._room_id

    @property
    def img_id(self):
        return self._img_id

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def object_ids(self):
        return self._object_ids

    @property
    def img_path(self):
        return self._img_path

    @img_path.setter
    def img_path(self, value: str):
        self._img_path = value

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, value):
        self._embedding = value

    @property
    def text_descriptions(self):
        return self._text_descriptions

    def add_object_id(self, object_id: int | str) -> None:
        """Add an object ID to this view.

        Args:
            object_id: Object identifier to add
        """
        if object_id not in self._object_ids:
            self._object_ids.append(object_id)

    def serialize(self) -> dict:
        """Serialize the view to a dictionary (SRP: Persistence handling).

        Returns:
            Dictionary with all view data
        """
        return {
            "view_id": int(self._view_id) if isinstance(self._view_id, int) else self._view_id,
            "room_id": int(self._room_id) if isinstance(self._room_id, int) else self._room_id,
            "img_id": int(self._img_id) if isinstance(self._img_id, int) else self._img_id,
            "object_ids": [int(x) if isinstance(x, int) else x for x in self._object_ids],
            "img_path": self._img_path,
            "text_descriptions": [str(x) for x in self._text_descriptions],
            "name": self._name,
        }

    def deserialize(self, data: dict) -> None:
        """Deserialize the view from a dictionary.

        Args:
            data: Dictionary with view data
        """
        self._object_ids = data.get("object_ids", [])
        self._img_path = data.get("img_path")
        # Handle both old "text_discription" (typo) and new "text_descriptions"
        self._text_descriptions = data.get("text_descriptions", data.get("text_discription", []))
        self._name = data.get("name")

    def save(self, path: str) -> None:
        """Save view metadata to disk.

        Args:
            path: Directory path to save the view
        """
        EntityPersistence.save_entity_metadata_only(str(self._view_id), self.serialize(), path)

    def load(self, path: str) -> None:
        """Load view metadata from disk.

        Args:
            path: Directory path to load the view from
        """
        data = EntityPersistence.load_entity_metadata_only(str(self._view_id), path)
        self.deserialize(data)

    def __str__(self) -> str:
        """String representation of the view."""
        return (
            f"{self.__class__.__name__}("
            f"id={self._view_id}, room={self._room_id}, "
            f"img={self._img_id}, path={self._img_path}, "
            f"objects={len(self._object_ids)})"
        )

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return self.__str__()
