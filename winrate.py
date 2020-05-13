import os

try:
    with open('log.txt', 'r') as file:
        os.system('cls')

        r_wins = 0
        m_wins = 0
        r_losses = 0
        m_losses = 0
        r_total = 0
        m_total = 0

        for line in file.readlines():
            if "Victory" in line:
                if "Model" in line:
                    m_wins += 1
                else:
                    r_wins += 1
            elif "Defeat" in line:
                if "Model" in line:
                    m_losses += 1
                else:
                    r_losses += 1

        r_total = r_wins + r_losses
        m_total = m_wins + m_losses

        r_win_rate = round((r_wins / r_total) * 100)
        m_win_rate = round((m_wins / m_total) * 100)

        print("\n-------------- Random Actions ----------------")
        print("Win Rate: {}%".format(r_win_rate))
        print("Total Matches: {}".format(r_total))
        print("W:L - {}:{}\n".format(r_wins, r_losses))

        print("\n-------------- Model Actions -----------------")
        print("Win Rate: {}%".format(m_win_rate))
        print("Total Matches: {}".format(m_total))
        print("W:L - {}:{}\n".format(m_wins, m_losses))
except FileNotFoundError:
    print("File not found.\n\nGo start the bot to have some match history")
