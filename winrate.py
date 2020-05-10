import os

try:
    with open('log.txt', 'r') as file:
        wins = 0
        losses = 0
        total = 0
        for line in file.readlines():
            if "Victory" in line:
                wins += 1
            elif "Defeat" in line:
                losses += 1

        total = wins + losses

        win_rate = round((wins / total) * 100)

        print("Win Rate: {}%".format(win_rate))
        print("Total Matches: {}".format(total))
        print("W:L - {}:{}".format(wins, losses))
except FileNotFoundError:
    print("File not found.\n\nGo start the bot to have some match history")