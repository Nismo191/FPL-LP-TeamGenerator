from re import S
import pulp as p
import numpy as np
import pandas as pd
from pulp.apis.coin_api import PULP_CBC_CMD

pd.set_option('display.max_column', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)
pd.set_option('display.max_colwidth', 500)
pd.set_option('expand_frame_repr', True)

def generate_team(budget, owner_theshold, form_theshold, fixture_threshold, sub_multiplier, excluded_clubs, game_week, number_of_gw_lookahead):

    look_ahead_gw = game_week + number_of_gw_lookahead

    df = pd.read_csv("https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2021-22/players_raw.csv")
    df_fixtures = pd.read_csv("https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2021-22/fixtures.csv")

    df_fixtures = df_fixtures[df_fixtures['event'].notna()]
    

    for x in df_fixtures.index:
        if df_fixtures.loc[x, "event"] < game_week or df_fixtures.loc[x, "event"] > look_ahead_gw:
            df_fixtures.drop(x, inplace=True)

    df_fixtures.reset_index(drop=True, inplace=True)

    print(df_fixtures)

    for x in df.index:
        if df.loc[x, "chance_of_playing_this_round"] == "None":
            df.loc[x, "chance_of_playing_this_round"] = 100
    
    for x in df.index:
        if df.loc[x, "team_code"] in excluded_clubs:
            df.drop(x, inplace=True)

    df.reset_index(drop=True, inplace=True)

    df = pd.merge(df, df_fixtures, left_on="team", right_on="team_a", how="left")
    df = pd.merge(df, df_fixtures, left_on="team", right_on="team_h", how="left")

    df = df.assign(**{"team_h_y": df.team_h_y.fillna(df.team_h_x)})
    df = df.assign(**{"team_a_y": df.team_a_y.fillna(df.team_a_x)})
    df = df.assign(**{"team_h_difficulty_y": df.team_h_difficulty_y.fillna(df.team_h_difficulty_x)})
    df = df.assign(**{"team_a_difficulty_y": df.team_a_difficulty_y.fillna(df.team_a_difficulty_x)})


    fixture_difficulty = []
    for x in df.index:
        if df.loc[x, "team_h_y"] == float(df.loc[x, "team"]):
            fixture_difficulty.append(int(df.loc[x, "team_h_difficulty_y"]))
        else:
            fixture_difficulty.append(int(df.loc[x, "team_a_difficulty_y"]))

    df["fixture_difficulty"] = fixture_difficulty

    # print(df[["team", "team_h_y", "team_a_y", "team_h_difficulty_y", "team_a_difficulty_y", "fixture_difficulty"]])
    df.to_csv("debug.csv")

    expected_scores = df["total_points"]
    prices = df["now_cost"] / 10
    positions = df["element_type"]
    team_id = df["team"]
    clubs = df["team_code"]
    name = df["web_name"]
    selected_percentage = df["selected_by_percent"]
    form = df["form"]
    chance_of_playing = df["chance_of_playing_this_round"]
    fixture_difficulty = df["fixture_difficulty"]

    num_players = len(expected_scores)
    total_budget = budget
    sub_factor = sub_multiplier

    print("Excluded clubs: ")
    print(np.unique(clubs))

    model = p.LpProblem("Maximize Points", p.LpMaximize)

    decisions = [
        p.LpVariable("x{}".format(i), lowBound=0, upBound=1, cat="Integer")
        for i in range(num_players)
    ]

    sub_decisions = [
        p.LpVariable("z{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]

    captain_decisions = [
        p.LpVariable("y{}".format(i), lowBound=0, upBound=1, cat="Integer")
        for i in range(num_players)
    ]

    # objective function:
    model += sum((captain_decisions[i] + decisions[i] + sub_decisions[i]*sub_factor) * expected_scores[i]
                    for i in range(num_players)), "Objective"

    # cost constraint
    model += sum((decisions[i] + sub_decisions[i]) * prices[i] for i in range(num_players)) <= total_budget  # total cost

    # position constraints
    # 1 starting goalkeeper
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 1) == 1
    # 2 total goalkeepers
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 1) == 2

    # 3-5 starting defenders
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 2) >= 3
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 2) <= 5
    # 5 total defenders
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 2) == 5

    # 3-5 starting midfielders
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 3) >= 3
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 3) <= 5
    # 5 total midfielders
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 3) == 5

    # 1-3 starting attackers
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 4) >= 1
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 4) <= 3
    # 3 total attackers
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 4) == 3

    # Below x% ownership
    model += sum(decisions[i] for i in range(num_players) if selected_percentage[i] < owner_theshold) == 11

    # Form constraint
    model += sum(decisions[i] for i in range(num_players) if form[i] > form_theshold) == 11

    # Playing Chance constraint
    model += sum(decisions[i] for i in range(num_players) if int(chance_of_playing[i]) >= 100) == 11

    # Fixture constraint
    model += sum(decisions[i] for i in range(num_players) if fixture_difficulty[i] <= fixture_threshold) == 11

    # club constraint
    for club_id in np.unique(clubs):
        model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if clubs[i] == club_id) <= 3  # max 3 players


    model += sum(decisions) == 11  # total team size
    model += sum(captain_decisions) == 1  # 1 captain

    for i in range(num_players):  
        model += (decisions[i] - captain_decisions[i]) >= 0  # captain must also be on team
        model += (decisions[i] + sub_decisions[i]) <= 1  # subs must not be on team

    model.solve(PULP_CBC_CMD(msg=0))

    print("Starting:")
    price = 0
    points = 0
    position = 5

    for j in range(position):
        if j == 1:
            print("GK")
        elif j == 2:
            print("DEF")
        elif j == 3:
            print("MID")
        elif j == 4:
            print("ATT")
        for i in range(df.shape[0]):
            if decisions[i].value() != 0:
                if df["element_type"][i] == j:
                    print("**{}** Points = {}, Price = {}, Owned % = {}, Form = {}, Play Chance = {}, Fixture_difficulty = {}".format(df["web_name"][i], df["total_points"][i], df["now_cost"][i], df["selected_by_percent"][i], df["form"][i], df["chance_of_playing_next_round"][i], df["fixture_difficulty"][i]))
                    price += df["now_cost"][i]
                    points += df["total_points"][i]
    print("Total Price: {}".format(price/10))
    print("Total Points: {}".format(points))

    print("Substitutes:")
    for i in range(df.shape[0]):
        if sub_decisions[i].value() != 0:
            print("**{}** Points = {}, Price = {}, Owned % = {}".format(df["web_name"][i], df["total_points"][i], df["now_cost"][i], df["selected_by_percent"][i]))
            price += df["now_cost"][i]
            points += df["total_points"][i]
    print("Total Price: {}".format(price/10))
    print("Total Points: {}".format(points))

    print("Captain:")
    for i in range(df.shape[0]):
        if captain_decisions[i].value() == 1:
            print("**CAPTAIN: {}** Points = {}, Price = {}".format(df["web_name"][i], df["total_points"][i], df["now_cost"][i], df["selected_by_percent"][i]))



if __name__ == "__main__":
    #[ 1  2  3  4  6  7  8 11 13 14 20 21 31 36 39 43 45 57 90 94]
    #[ 2        4  6  7  8 11 13    20 21 31 36    43 45 57 90 94]
    # excluded_teams = [94, 36, 31, 11, 13, 1, 45, 20, 57, 21, 7, 90]
    budget = 100.6
    owner_theshold = 100
    form_threshold = 3
    fixture_threshold = 2
    sub_multiplier = 0.1
    excluded_teams = []
    game_week = 19.0
    number_of_gw_lookahead = 2
    generate_team(budget, owner_theshold, form_threshold, fixture_threshold, sub_multiplier, excluded_teams, game_week, number_of_gw_lookahead)
