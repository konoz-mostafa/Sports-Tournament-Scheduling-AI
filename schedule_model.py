import random
from datetime import datetime, timedelta
from data.teams_venues_times import teams, venues, match_times

start_date = datetime(2025, 5, 1)
end_date = datetime(2026, 1, 31)

# أيام البطولة
all_dates = []
current_date = start_date
while current_date <= end_date:
    all_dates.append(current_date)
    current_date += timedelta(days=1)

class Match:
    def __init__(self, team1, team2, date, time, venue):
        self.team1 = team1
        self.team2 = team2
        self.date = date
        self.time = time
        self.venue = venue

    def __repr__(self):
        return f"{self.team1} vs {self.team2} on {self.date.strftime('%Y-%m-%d')} at {self.time} in {self.venue}"

def generate_schedule(teams, venues, all_dates, match_times, min_rest_days=4):
    schedule = []
    last_played = {team: start_date - timedelta(days=min_rest_days) for team in teams}

    # تقسيم المباريات لدور أول ودور ثاني
    pairings = [(t1, t2) for i, t1 in enumerate(teams) for t2 in teams[i+1:]]
    pairings_double = pairings + [(t2, t1) for t1, t2 in pairings]  # ذهاب وإياب
    random.shuffle(pairings_double)

    # ترتيب الأيام أسبوعياً
    week_dates = all_dates[::7]

    for week_start in week_dates:
        used_teams = set()
        for home, away in pairings_double:
            if home in used_teams or away in used_teams:
                continue

            # أيام ممكنة في الأسبوع مع راحة 4 أيام
            possible_days = [d for d in all_dates if week_start <= d < week_start + timedelta(days=7)
                             and (d - last_played[home]).days >= min_rest_days
                             and (d - last_played[away]).days >= min_rest_days]
            if not possible_days:
                continue

            date = random.choice(possible_days)
            time = random.choice(match_times)
            venue = random.choice(venues)
            match = Match(home, away, date, time, venue)
            schedule.append(match)

            last_played[home] = date
            last_played[away] = date
            used_teams.update([home, away])
            pairings_double.remove((home, away))

    schedule.sort(key=lambda m: m.date)
    return schedule

def print_team_schedules(schedule):
    team_schedule = {team: [] for team in teams}
    for match in schedule:
        team_schedule[match.team1].append(match)
        team_schedule[match.team2].append(match)
    for team, matches in team_schedule.items():
        print(f"\n=== Schedule for {team} ===")
        matches.sort(key=lambda m: m.date)
        for m in matches:
            opponent = m.team2 if m.team1 == team else m.team1
            print(f"{m.date.strftime('%Y-%m-%d')} at {m.time} vs {opponent} in {m.venue}")

if __name__ == "__main__":
    schedule = generate_schedule(teams, venues, all_dates, match_times)
    print("=== First 20 Matches in Tournament ===")
    for m in schedule[:20]:
        print(m)
    print_team_schedules(schedule)
