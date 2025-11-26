from flask import Flask, render_template
from schedule_model import generate_schedule, teams, venues, all_dates, match_times

app = Flask(__name__)

# توليد جدول الدوري
full_schedule = generate_schedule(teams, venues, all_dates, match_times)

# ترتيب الجدول حسب التاريخ والساعة
full_schedule.sort(key=lambda m: (m.date, m.time))

# إنشاء جداول لكل فريق
team_schedules = {team: [m for m in full_schedule if m.team1 == team or m.team2 == team] for team in teams}

@app.route('/')
def index():
    # إرسال كل جدول الدوري وجداول الفرق للـ template
    return render_template('index.html', full_schedule=full_schedule, team_schedules=team_schedules)

@app.route('/team/<team_name>')
def team_page(team_name):
    team_schedule = team_schedules.get(team_name, [])
    return render_template('team.html', team=team_name, schedule=team_schedule)

if __name__ == '__main__':
    app.run(debug=True)
