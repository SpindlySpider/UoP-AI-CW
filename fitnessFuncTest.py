'''
Example Fitness Ideas
Sample “good” poses:
Keeping the spider balanced (e.g., center of mass near origin).
Symmetric leg angles: difference between left-right leg pairs should be small.
Reach a height: make the spider lift its body to a target height.
Minimize joint strain: penalize extreme angles.
Example formula:
'''
# Fitness = - (abs(sum(L1 - R1)) + abs(sum(L2 - R2)) + abs(sum(L3 - R3)) + abs(sum(L4 - R4))) 
# Symmetry term

#SAMPLE FITNESS FUNCTION CODE
import numpy

def fitness_function(angles):
    angles = numpy.array(angles)
    
    # Stability: penalize uneven heights / extreme angles
    stability_score = 1 - numpy.mean(numpy.abs(angles) / numpy.pi)
    
    # Symmetry: left vs right legs
    symmetry_diff = 0
    for i in range(4):
        L = angles[i*3:(i+1)*3]
        R = angles[24 - (i+1)*3:24 - i*3]
        symmetry_diff += numpy.mean(numpy.abs(L + R)) # assuming symmetric legs have opposite angles
    symmetry_score = 1 - symmetry_diff / (4 * numpy.pi)
    
    # Goal: keep body height around target
    z_target = 0.1
    height_error = abs(numpy.mean(angles[::3]) - z_target)  # proxy
    goal_score = numpy.exp(-height_error)
    
    # Strain penalty: discourage extreme angles
    strain_penalty = numpy.mean((angles / numpy.pi) ** 2)
    
    fitness = 0.4*stability_score + 0.3*symmetry_score + 0.2*goal_score + 0.1*(1 - strain_penalty) # Combine scores
    return fitness
'''
Scaling and Normalization

Because each sub-score (symmetry, stability, etc.) may have different numeric ranges, you should normalize them between 0 and 1.
This ensures that one term doesn't dominate just because it's numerically larger.

Example:
'''
# stability_score = (stability_score - min_val) / (max_val - min_val)
