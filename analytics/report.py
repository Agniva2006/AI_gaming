def generate_match_report(match, stats):
    """
    Phase E: Generates a post-match natural language analytical report.
    """
    pct = stats.get_possession_pct()
    
    report = [
        "\n" + "="*40,
        "          FULL TIME MATCH REPORT",
        "="*40,
        f"Score: BLUE (4-3-3)  {match.score[0]} - {match.score[1]}  RED (4-4-2)",
        "-"*40,
        f"Possession:         {pct[0]:.1f}%  |  {pct[1]:.1f}%",
        f"Passes Attempted:   {stats.passes_attempted[0]:<5}  |  {stats.passes_attempted[1]}",
        f"Shots:              {stats.shots[0]:<5}  |  {stats.shots[1]}",
        f"Tackles/Recoveries: {stats.tackles[0]:<5}  |  {stats.tackles[1]}",
        "-"*40,
        "TACTICAL SUMMARY:"
    ]
    
    # Analyze Possession
    if pct[0] > 55:
        report.append("> BLUE dominated midfield possession. The 4-3-3 shape allowed their")
        report.append("> central midfielders to dictate tempo while wingers stayed wide.")
    elif pct[1] > 55:
        report.append("> RED controlled the game. Their compact 4-4-2 banks effectively")
        report.append("> choked the center, forcing BLUE into wide, low-percentage areas.")
    else:
        report.append("> A tightly contested match. Both tactical systems neutralized each")
        report.append("> other, resulting in a fierce battle for the middle third.")
        
    # Analyze Attacking Output
    if stats.shots[0] > stats.shots[1] + 3:
        report.append("> BLUE's high press from the ST and Wingers successfully disrupted")
        report.append("> RED's buildup, leading to numerous scoring opportunities.")
    elif stats.shots[1] > stats.shots[0] + 3:
        report.append("> RED's counter-attacks were lethal. The two strikers constantly")
        report.append("> exploited the spaces left behind BLUE's advancing fullbacks.")
        
    # Analyze Work Rate
    if stats.tackles[0] + stats.tackles[1] > 60:
        report.append("> The match was highly physical with constant turnovers and pressing.")
        
    report.append("="*40 + "\n")
    return "\n".join(report)
