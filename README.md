# PitchArsenalModelling
Application to create/edit pitcher arsenals and calculate the expected results


TODO:

Data:
- Include other metrics besides whiff and csw
- include other pitch metrics in indivudal pitch models (context) - zone%?   chase%?
    - COMPLETED -> Not completely flattening out all other pitch data, but creating some "context" features. ie. relation to primary fastball, usage, etc. 
- filter to min pitch threshold
    - COMPLETED -> initial filter of 30 pitches pre processing, still need to filter after
    - Stuff+ stabilizes at 60-80 pitches. 
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



## TODO as of Feb 1

Data:
- Seem to have settled on "optimal" feature space
    - include location somewhat, but also simple enough user can specify/change
- pct vs RHB feature?

Models:
- Seems to be optimizied
- Include more model details in script? feat importances, shap, etc?
- shap plots show gray dots when feature is null?

App:
- remove unknown pitchers
- FUTURE: add possible deviation param for IVB/HB
- connect database back to model pipeline - preprocess, predict, show prediction in app. 
- aesthetics
- in ivb/hb plot do scatter of entire season rather than aggregated
- Specify handedness, flip axis on HB plot






