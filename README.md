# PitchArsenalModelling
Application to create/edit pitcher arsenals and calculate the expected results


TODO:

Data:
- Include other metrics besides whiff and csw
- include other pitch metrics in indivudal pitch models (context) - zone%?   chase%?
    - COMPLETED -> Not completely flattening out all other pitch data, but creating some "context" features. ie. relation to primary fastball, usage, etc. 
- filter to min pitch threshold
- rolling avgs over season?
- Pitch type noise? ie. SL and CT with similar shapes? do we want to deal with that? clustering?

Modelling:
- train per pitch models to predict whiff/csw/others
- from whiff/csw compute approx fip/era (with confidence range) - will need BB, HR, etc eventually
- incorporate stuff+?

App:
- Create pitch plot
- allow user to create new arsenal or add to current
- specify pitch specs (velo, break, angles, etc)
- ingest into model, predict whiff, csw, fip, etc.


