"""
Fitness Function for Sports Tournament Scheduling
Calculates fitness score for tournament schedules
"""


def compute_fitness(schedule, teams=None, min_rest_days=4, weights=None):
    """
    Calculate fitness score for a schedule (higher is better)
    Enhanced to penalize ALL validation errors
    
    Parameters:
    -----------
    schedule : List[Match]
        List of matches in the schedule
    teams : List[str], optional
        List of all teams (required for full validation)
    min_rest_days : int
        Minimum rest days required between matches for a team
    weights : dict, optional
        Weights for different penalty types
        
    Returns:
    --------
    float
        Fitness score (0-100, higher is better)
    """
    if weights is None:
        weights = {
            'venue_conflict': 1.0, 
            'rest_violation': 0.7, 
            'repeated_opponent': 0.0, 
            'time_balance': 0.3,
            'missing_matches': 5.0,  # Heavy penalty for missing matches
            'home_away_imbalance': 2.0,  # Penalty for unbalanced pairs
            'consecutive_home': 1.5,  # Penalty for 3+ consecutive home
            'wrong_match_count': 3.0  # Penalty for wrong matches per team
        }

    penalty = 0
    
    # If teams list provided, do full validation
    if teams is not None:
        expected_matches = len(teams) * (len(teams) - 1)  # 306
        expected_per_team = (len(teams) - 1) * 2  # 34
        
        # Check total match count
        if len(schedule) != expected_matches:
            missing = abs(len(schedule) - expected_matches)
            penalty += missing * weights['missing_matches']
    
    venue_day_time = {}
    last_played = {}
    matches_per_pair = {}
    team_times = {}
    team_match_count = {}
    team_schedule = {}
    
    if teams is not None:
        team_match_count = {team: 0 for team in teams}
        team_schedule = {team: [] for team in teams}

    for match in schedule:
        # Check venue conflicts (same venue, date, time)
        key = (match.date, match.time, match.venue)
        if key in venue_day_time:
            penalty += weights['venue_conflict']
        else:
            venue_day_time[key] = True

        # Check rest day violations
        for team in [match.team1, match.team2]:
            if team in last_played:
                delta_days = (match.date - last_played[team]).days
                if delta_days < min_rest_days:
                    penalty += weights['rest_violation']
            last_played[team] = match.date

        # Track pairs for home/away balance
        pair = tuple(sorted([match.team1, match.team2]))
        matches_per_pair[pair] = matches_per_pair.get(pair, 0) + 1
        
        # Track match count per team
        if teams is not None:
            team_match_count[match.team1] = team_match_count.get(match.team1, 0) + 1
            team_match_count[match.team2] = team_match_count.get(match.team2, 0) + 1
            
            # Track home/away for consecutive check
            team_schedule[match.team1].append(('home', match.date))
            team_schedule[match.team2].append(('away', match.date))

        # Track time distribution per team
        for team in [match.team1, match.team2]:
            if team not in team_times:
                team_times[team] = {}
            team_times[team][match.time] = team_times[team].get(match.time, 0) + 1

    # Penalty for wrong match count per team
    if teams is not None:
        for team, count in team_match_count.items():
            if count != expected_per_team:
                penalty += abs(count - expected_per_team) * weights['wrong_match_count']
    
    # Penalty for home/away imbalance
    for pair, count in matches_per_pair.items():
        if count != 2:
            penalty += abs(count - 2) * weights['home_away_imbalance']
        elif teams is not None:
            # Check if one is home and one is away
            home_count = sum(1 for m in schedule 
                           if tuple(sorted([m.team1, m.team2])) == pair and m.team1 == pair[0])
            if home_count != 1:
                penalty += weights['home_away_imbalance']
    
    # Penalty for consecutive home matches
    if teams is not None:
        for team, matches in team_schedule.items():
            matches.sort(key=lambda x: x[1])
            consecutive_home = 0
            for match_type, date in matches:
                if match_type == 'home':
                    consecutive_home += 1
                    if consecutive_home >= 3:
                        penalty += weights['consecutive_home']
                        break
                else:
                    consecutive_home = 0

    # Penalty for unbalanced time distribution
    for team, times in team_times.items():
        if len(times) > 1:
            diff = abs(times.get('17:00', 0) - times.get('20:00', 0))
            penalty += diff * weights['time_balance']

    fitness = max(0, 100 - penalty)
    return fitness


def compute_fitness_verbose(schedule, teams=None, min_rest_days=4, weights=None):
    """
    Calculate fitness score with detailed breakdown (for debugging)
    Enhanced to penalize ALL validation errors
    
    Returns:
    --------
    tuple
        (fitness_score, breakdown_dict)
    """
    if weights is None:
        weights = {
            'venue_conflict': 1.0, 
            'rest_violation': 0.7, 
            'repeated_opponent': 0.0, 
            'time_balance': 0.3,
            'missing_matches': 5.0,
            'home_away_imbalance': 2.0,
            'consecutive_home': 1.5,
            'wrong_match_count': 3.0
        }

    penalty = 0
    
    # If teams list provided, do full validation
    if teams is not None:
        expected_matches = len(teams) * (len(teams) - 1)  # 306
        expected_per_team = (len(teams) - 1) * 2  # 34
        
        # Check total match count
        missing_matches_penalty = 0
        if len(schedule) != expected_matches:
            missing = abs(len(schedule) - expected_matches)
            missing_matches_penalty = missing * weights['missing_matches']
            penalty += missing_matches_penalty
    else:
        missing_matches_penalty = 0
    
    venue_day_time = {}
    last_played = {}
    matches_per_pair = {}
    team_times = {}
    team_match_count = {}
    team_schedule = {}
    
    if teams is not None:
        team_match_count = {team: 0 for team in teams}
        team_schedule = {team: [] for team in teams}

    venue_conflict_count = 0
    rest_violation_count = 0
    repeated_opponents_count = 0
    time_balance_penalty = 0
    wrong_match_count_penalty = 0
    home_away_imbalance_penalty = 0
    consecutive_home_penalty = 0

    for match in schedule:
        # Check venue conflicts
        key = (match.date, match.time, match.venue)
        if key in venue_day_time:
            penalty += weights['venue_conflict']
            venue_conflict_count += 1
        else:
            venue_day_time[key] = True

        # Check rest day violations
        for team in [match.team1, match.team2]:
            if team in last_played:
                delta_days = (match.date - last_played[team]).days
                if delta_days < min_rest_days:
                    penalty += weights['rest_violation']
                    rest_violation_count += 1
            last_played[team] = match.date

        # Track pairs
        pair = tuple(sorted([match.team1, match.team2]))
        if pair in matches_per_pair:
            penalty += weights['repeated_opponent']
            repeated_opponents_count += 1
        matches_per_pair[pair] = matches_per_pair.get(pair, 0) + 1
        
        # Track match count per team
        if teams is not None:
            team_match_count[match.team1] = team_match_count.get(match.team1, 0) + 1
            team_match_count[match.team2] = team_match_count.get(match.team2, 0) + 1
            
            # Track home/away for consecutive check
            team_schedule[match.team1].append(('home', match.date))
            team_schedule[match.team2].append(('away', match.date))

        # Track time distribution
        for team in [match.team1, match.team2]:
            if team not in team_times:
                team_times[team] = {}
            team_times[team][match.time] = team_times[team].get(match.time, 0) + 1

    # Penalty for wrong match count per team
    if teams is not None:
        for team, count in team_match_count.items():
            if count != expected_per_team:
                team_penalty = abs(count - expected_per_team) * weights['wrong_match_count']
                penalty += team_penalty
                wrong_match_count_penalty += team_penalty
    
    # Penalty for home/away imbalance
    for pair, count in matches_per_pair.items():
        if count != 2:
            pair_penalty = abs(count - 2) * weights['home_away_imbalance']
            penalty += pair_penalty
            home_away_imbalance_penalty += pair_penalty
        elif teams is not None:
            # Check if one is home and one is away
            home_count = sum(1 for m in schedule 
                           if tuple(sorted([m.team1, m.team2])) == pair and m.team1 == pair[0])
            if home_count != 1:
                penalty += weights['home_away_imbalance']
                home_away_imbalance_penalty += weights['home_away_imbalance']
    
    # Penalty for consecutive home matches
    if teams is not None:
        for team, matches in team_schedule.items():
            matches.sort(key=lambda x: x[1])
            consecutive_home = 0
            for match_type, date in matches:
                if match_type == 'home':
                    consecutive_home += 1
                    if consecutive_home >= 3:
                        penalty += weights['consecutive_home']
                        consecutive_home_penalty += weights['consecutive_home']
                        break
                else:
                    consecutive_home = 0

    # Penalty for unbalanced time distribution
    for team, times in team_times.items():
        if len(times) > 1:
            diff = abs(times.get('17:00', 0) - times.get('20:00', 0))
            team_time_penalty = diff * weights['time_balance']
            penalty += team_time_penalty
            time_balance_penalty += diff

    fitness = max(0, 100 - penalty)

    breakdown = {
        'fitness_score': fitness,
        'venue_conflicts': venue_conflict_count,
        'rest_violations': rest_violation_count,
        'repeated_opponents': repeated_opponents_count,
        'time_balance_penalty': time_balance_penalty,
        'missing_matches_penalty': missing_matches_penalty,
        'wrong_match_count_penalty': wrong_match_count_penalty,
        'home_away_imbalance_penalty': home_away_imbalance_penalty,
        'consecutive_home_penalty': consecutive_home_penalty,
        'total_penalty': penalty
    }

    return fitness, breakdown

