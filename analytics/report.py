def generate_match_report(match, stats):
    """
    Generates a comprehensive post-match analytical report with Expected Goals (xG),
    possession, pass accuracy, and tactical system breakdown.
    """
    pct = stats.get_possession_pct()
    xg_a = match.cumulative_xg.get(0, 0.0)
    xg_b = match.cumulative_xg.get(1, 0.0)

    report = [
        "\n" + "=" * 50,
        "          [MATCH REPORT] FULL TIME TACTICAL ANALYSIS",
        "=" * 50,
        f"Score:  YOU ({match.team_a.formation_type})  {match.score[0]}  -  {match.score[1]}  RL AI ({match.team_b.formation_type})",
        f"xG:     YOU {xg_a:.2f} xG            |  {xg_b:.2f} xG RL AI",
        "-" * 50,
        f"Possession:         {pct[0]:.1f}%          |  {pct[1]:.1f}%",
        f"Passes Attempted:   {stats.passes_attempted[0]:<14} |  {stats.passes_attempted[1]}",
        f"Shots on Goal:      {stats.shots[0]:<14} |  {stats.shots[1]}",
        f"Tackles Won:        {stats.tackles[0]:<14} |  {stats.tackles[1]}",
        "-" * 50,
        "[ANALYTICS] TACTICAL & ANALYTICAL SUMMARY:"
    ]

    # Analyze Shot Conversion vs xG (Clinical vs Wasteful)
    if match.score[0] > xg_a + 0.5:
        report.append("> [CLINICAL] YOU outperformed your Expected Goals, demonstrating exceptional finishing.")
    elif match.score[0] < xg_a - 0.7:
        report.append("> [WASTEFUL] YOU generated high-quality chances but failed to convert against the AI goalkeeper.")

    if match.score[1] > xg_b + 0.5:
        report.append("> [LETHAL] RL AI seized half-chances with clinical precision to punish defensive lapses.")

    # Possession Analysis
    if pct[0] > 56.0:
        report.append("> [DOMINANCE] YOU controlled midfield tempo with disciplined short passing and compactness.")
    elif pct[1] > 56.0:
        report.append("> [AI CONTROL] The RL AI dictated match pace, compressing passing lanes and forcing long balls.")
    else:
        report.append("> [CONTESTED] Midfield battle was evenly contested with end-to-end transition play.")

    # Flank and Tendency Analysis from Opponent Tendency Profiler
    from ai.tendency_profiler import tendency_profiler
    tendency = tendency_profiler.get_profile_summary()
    counter = tendency_profiler.get_counter_strategy()

    report.append("-" * 50)
    report.append("[AI DEBRIEF] RL AI OPPONENT PROFILING & ADAPTATION:")
    flanks = tendency.get("flank_percentages", {})
    report.append(f"> Attacking Territory: Left {flanks.get('left', 0)}% | Center {flanks.get('center', 0)}% | Right {flanks.get('right', 0)}%")
    report.append(f"> Preferred Pass Style: {tendency.get('pass_style', 'Balanced')} ({tendency.get('through_ball_pct', 0)}% Through-Balls)")
    report.append(f"> AI Counter Strategy: [{counter.get('strategy_name', 'Balanced')}]")
    report.append(f"> AI Tactical Debrief: {counter.get('tactical_debrief', '')}")

    report.append("=" * 50 + "\n")
    return "\n".join(report)
