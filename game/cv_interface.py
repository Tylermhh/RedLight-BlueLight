from player import *

def get_player_filters() -> dict[int, Player]:
    # TODO: interface with CV script to create player objects
    # this is a stub
    return {i: Player(i) for i in range(4)}

def check_player_movement(players_playing: dict[int, Player], players_lost: dict[int, Player]) -> None:
    # TODO: interface with CV script to check player movement
    # this is a stub

    players_caught_ids: list[int] = []
    for id in players_playing:
        if players_playing[id].is_moving():
            players_caught_ids.append(id)

    for id in players_caught_ids:
        players_lost[id] = players_playing.pop(id)