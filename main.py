from re import S
import pulp as p
import numpy as np
import pandas as pd


def generate_team(budget, owner_theshold, form_theshold, sub_multiplier, excluded_clubs):

    df = pd.read_csv("https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2021-22/players_raw.csv")

    
    for x in df.index:
        if df.loc[x, "chance_of_playing_this_round"] == "None":
            df.loc[x, "chance_of_playing_this_round"] = 100
    
    for x in df.index:
        if df.loc[x, "team"] in excluded_clubs:
            df.drop(x, inplace=True)

    df.reset_index(drop=True, inplace=True)

    print(df)

    expected_scores = df["total_points"]
    prices = df["now_cost"] / 10
    positions = df["element_type"]
    clubs = df["team_code"]
    name = df["web_name"]
    selected_percentage = df["selected_by_percent"]
    form = df["form"]
    chance_of_playing = df["chance_of_playing_this_round"]

    

    num_players = len(expected_scores)
    total_budget = budget
    sub_factor = sub_multiplier

    print(chance_of_playing)

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

    # club constraint
    for club_id in np.unique(clubs):
        model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if clubs[i] == club_id) <= 3  # max 3 players

    model += sum(decisions) == 11  # total team size
    model += sum(captain_decisions) == 1  # 1 captain

    for i in range(num_players):  
        model += (decisions[i] - captain_decisions[i]) >= 0  # captain must also be on team
        model += (decisions[i] + sub_decisions[i]) <= 1  # subs must not be on team

    model.solve()

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
                    print("**{}** Points = {}, Price = {}, Owned % = {}, Form = {}, Play Chance = {}".format(df["web_name"][i], df["total_points"][i], df["now_cost"][i], df["selected_by_percent"][i], df["form"][i], df["chance_of_playing_next_round"][i]))
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
    generate_team(100.6, 100, 3, 0.1, [])


# GW15
# Starting:
# **Smith Rowe** Points = 69, Price = 61, Owned % = 28.3
# **Rüdiger** Points = 72, Price = 61, Owned % = 22.0
# **James** Points = 76, Price = 62, Owned % = 35.2
# **Mendy** Points = 66, Price = 63, Owned % = 19.8
# **Gallagher** Points = 72, Price = 61, Owned % = 24.6
# **van Dijk** Points = 78, Price = 66, Owned % = 18.0
# **Salah** Points = 152, Price = 131, Owned % = 73.1
# **Alexander-Arnold** Points = 99, Price = 81, Owned % = 40.2
# **Cancelo** Points = 84, Price = 68, Owned % = 39.2
# **Bernardo** Points = 84, Price = 76, Owned % = 24.0
# **Dennis** Points = 75, Price = 57, Owned % = 25.7
# Total Price: 78.7
# Substitutes:
# **Gray** Points = 65, Price = 55, Owned % = 9.8
# **Broja** Points = 33, Price = 50, Owned % = 2.4
# **King** Points = 54, Price = 58, Owned % = 6.5
# **Sá** Points = 67, Price = 50, Owned % = 3.3
# Total Price: 100.0
# Total Points: 1146

# GW16
# GK
# **Sá** Points = 71, Price = 51, Owned % = 3.6, Form = 6.2, Play Chance = None
# DEF
# **Rüdiger** Points = 81, Price = 61, Owned % = 23.0, Form = 5.6, Play Chance = 100
# **van Dijk** Points = 83, Price = 67, Owned % = 18.8, Form = 6.8, Play Chance = None
# **Alexander-Arnold** Points = 108, Price = 81, Owned % = 41.4, Form = 8.8, Play Chance = 100
# **Cancelo** Points = 91, Price = 68, Owned % = 36.8, Form = 4.8, Play Chance = 0
# MID
# **Mount** Points = 77, Price = 76, Owned % = 19.0, Form = 7.0, Play Chance = 100
# **Gallagher** Points = 87, Price = 61, Owned % = 25.4, Form = 5.0, Play Chance = 100
# **Salah** Points = 160, Price = 131, Owned % = 73.3, Form = 8.6, Play Chance = None
# **Bernardo** Points = 90, Price = 77, Owned % = 30.3, Form = 8.2, Play Chance = None
# **Bowen** Points = 74, Price = 65, Owned % = 7.3, Form = 3.8, Play Chance = None
# ATT
# **Dennis** Points = 84, Price = 58, Owned % = 33.1, Form = 8.8, Play Chance = 100
# Total Price: 79.6
# Substitutes:
# **Ramsdale** Points = 70, Price = 50, Owned % = 16.0
# **Broja** Points = 35, Price = 51, Owned % = 2.6
# **King** Points = 56, Price = 58, Owned % = 7.8
# **Coady** Points = 56, Price = 45, Owned % = 6.4
# Total Price: 100.0
# Total Points: 1223

# GW16 (Below 50% Ownership)
# GK
# **Alisson** Points = 72, Price = 60, Owned % = 8.0, Form = 5.8, Play Chance = 100
# DEF
# **Rüdiger** Points = 81, Price = 61, Owned % = 23.0, Form = 5.6, Play Chance = 100
# **van Dijk** Points = 83, Price = 67, Owned % = 18.8, Form = 6.8, Play Chance = None
# **Alexander-Arnold** Points = 108, Price = 81, Owned % = 41.4, Form = 8.8, Play Chance = 100
# **Cancelo** Points = 91, Price = 68, Owned % = 36.8, Form = 4.8, Play Chance = 0
# MID
# **Mount** Points = 77, Price = 76, Owned % = 19.0, Form = 7.0, Play Chance = 100
# **Gallagher** Points = 87, Price = 61, Owned % = 25.4, Form = 5.0, Play Chance = 100
# **Bernardo** Points = 90, Price = 77, Owned % = 30.3, Form = 8.2, Play Chance = None
# **Son** Points = 81, Price = 103, Owned % = 16.6, Form = 7.7, Play Chance = 100
# **Bowen** Points = 74, Price = 65, Owned % = 7.3, Form = 3.8, Play Chance = None
# ATT
# **Dennis** Points = 84, Price = 58, Owned % = 33.1, Form = 8.8, Play Chance = 100
# Total Price: 77.7
# Substitutes:
# **Gabriel** Points = 67, Price = 52, Owned % = 4.1
# **Pukki** Points = 58, Price = 59, Owned % = 6.1
# **King** Points = 56, Price = 58, Owned % = 7.8
# **Sá** Points = 71, Price = 51, Owned % = 3.6
# Total Price: 99.7
# Total Points: 1180