"""Illustrative algebra from published aggregate counts, no project data imports."""
from math import ceil,log,sqrt
from statistics import NormalDist
import json
from pathlib import Path
z=NormalDist().inv_cdf(.975)+NormalDist().inv_cdf(.8)
values=[]
for discordance in (.06,.10):
    for effect in (.02,.03,.05):
        independent=ceil(z*z*(discordance-effect*effect)/(effect*effect))
        values.append({'paired_discordance_q':discordance,'target_difference_delta':effect,
          'approx_independent_pairs_for_80pct_power_two_sided_alpha05':independent,
          'illustrative_pairs_with_design_effect_1_5':ceil(1.5*independent),
          'illustrative_pairs_with_design_effect_2':2*independent})
result={'method':'Planning illustration only: D=paired correct difference in {-1,0,1}; E[D]=delta, Var[D]=q-delta^2; n≈(z.975+z.8)^2 Var[D]/delta^2. Not a patient-cluster power calculation or a replacement analysis rule.',
 'tables':values,'independent_zero_harm_trials_needed_for_one_sided_95pct_upper_rate_below_0_005':ceil(log(.05)/log(.995)),
 'V6_retained_source_bound_candidate_oracle_ceiling':233,'V6_same_fixed_auxiliary_fatal_outcomes_ceiling':232,
 'V7_retained_candidate_oracle_ceiling_ignoring_auxiliary_failure_and_selection_cost':234,
 'V7_same_fixed_auxiliary_fatal_outcomes_ceiling':233,
 'V7_minimum_for_current_historical_development_gate':240,'V7_additional_oracle_opportunities_needed_if_A233':6,
 'V7_known_B_budget_stops_fraction_of_entered':81/268,
 'V7_recorded_mean_team_tokens_ratio_to_single':114984.19/72180.06,
 'warning':'The aggregate counts are development observations; oracle ceilings are exact only for these retained candidates/phase outcomes, not population/architecture ceilings. Reported token means omit unknown failed usage; ratio is not billing/FLOPs ratio.'}
Path(__file__).with_name('planning_calculations.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
