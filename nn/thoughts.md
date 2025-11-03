# initial ideas from project brief
from the project brief: Predicting Joint Angles Using Neural Networks
additionally it specifies inputs are set of vectors and I assume the output will also be a set of vectors.
so does this mean we need to input like half a gait, e.g. 150 frames of 300 and then we predict the other 150 from the nn?
I dont think we can use chromosome encoding as the set of inputs as I beleive the output of the NN actually needs to be the gait. 

actually at a closer look, it seems that we are not required to make a gait, maybe that is completely wrong interpretation.
in which case maybe the NN needs to take a 1x24 joints and then produce a valid version of that?
I think this is further implied by deliverables point 3. example of output poses (joint angles or rendered spider poses) vs in part one what is animations or visual indicators.

maybe this could look like, set of frames from part 1, then NN predicts rest and we can compare against actual sinewave produced by target sol.

oh actually set of frames 300x24, and half are given to the NN, the closer it gets to actual ones the better.

could optimize using less set of frames since A and B are the same
