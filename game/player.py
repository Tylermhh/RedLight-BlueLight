class Player:
    def __init__(self, id: int, face_filter: list | None = None):
        self.id = id
        self.face_filter = face_filter
        # TODO: add more fields to interface with CV script as necessary

    def is_moving(self) -> bool:
        # TODO: interface with CV script
        # this is a stub
        return False

    def is_won(self) -> bool:
        # TODO: interface with CV script
        # this is a stub
        # how to do this???
        return False