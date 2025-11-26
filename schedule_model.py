import random
from datetime import datetime, timedelta
from data.teams_venues_times import teams, venues, match_times

# مدة البطولة
start_date = datetime(2025, 5, 1)
end_date = datetime(2026, 1, 31)

# إعداد أيام البطولة
all_dates = []
current_date = start_date
while current_date <= end_date:
    all_dates.append(current_date)
    current_date += timedelta(days=1)


# تعريف الـ Match
class Match:
    def __init__(self, team1, team2, date, time, venue):
        self.team1 = team1
        self.team2 = team2
        self.date = date
        self.time = time
        self.venue = venue

    def __repr__(self):
        return f"{self.team1} vs {self.team2} on {self.date.strftime('%Y-%m-%d')} at {self.time} in {self.venue}"


# توليد جدول دوري أسبوعي مع تكرار المباريات مرتين
def generate_weekly_schedule(teams, venues, all_dates, match_times, min_rest_days=4):
    schedule = []
    last_played = {team: start_date - timedelta(days=min_rest_days) for team in teams}

    # توليد كل المباريات الممكنة مرتين لكل فريق
    pairings = []
    for t1 in teams:
        for t2 in teams:
            if t1 != t2:
                pairings.append((t1, t2))
                pairings.append((t2, t1))

    random.shuffle(pairings)

    # تخزين عدد مرات المباراة بين كل فريقين
    matches_count = {}
    for t1 in teams:
        for t2 in teams:
            if t1 != t2:
                matches_count[(t1, t2)] = 0

    # ترتيب الأيام أسبوعياً
    week_dates = all_dates[::7]  # بداية كل أسبوع

    for week_start in week_dates:
        used_teams = set()
        for home, away in pairings:
            if matches_count[(home, away)] >= 1:
                continue
            if home in used_teams or away in used_teams:
                continue

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
            matches_count[(home, away)] += 1
            used_teams.update([home, away])

    # ترتيب الجدول حسب التاريخ
    schedule.sort(key=lambda m: m.date)
    return schedule


# طباعة جدول كل فريق
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


# تجربة الكود
if __name__ == "__main__":
    schedule = generate_weekly_schedule(teams, venues, all_dates, match_times)
    print("=== First 20 Matches in Tournament ===")
    for m in schedule[:20]:
        print(m)
    print_team_schedules(schedule)
