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

def check_player_winning(players_playing: dict[int, Player], players_won: dict[int, Player]) -> None:
    # TODO: interface with CV script to check if players have won the game
    # this is a stub
    # how to do this???
    players_won_ids: list[int] = []
    for id in players_playing:
        if players_playing[id].is_won():
            players_won_ids.append(id)

    for id in players_won_ids:
        players_won[id] = players_playing.pop(id)