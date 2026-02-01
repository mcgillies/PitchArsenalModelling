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



## TODO as of Jan 31

Data:
- Tried incorporating fully flattened pitches - minimal difference in model results
    - "Best" model on val loss overfits (basically 0 train loss)
- Figure out how to leverage command in the model using plate location

Models:
- Tried all tree-based models w/ hyperparam opt. not much difference
- basic neural nets? doubtful but maybe

App:
- remove unknown pitchers
- fix bubble size in pitch plot
- FUTURE: add possible deviation param for IVB/HB
- include all relevant features in data table - depends on what gets passed to model
- connect database back to model pipeline - preprocess, predict, show prediction in app. 





