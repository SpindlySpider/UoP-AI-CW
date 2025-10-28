type Period = float 
type Magnitude = float
type H_offset = float
type Negative = bool
type Chromosome = tuple[Magnitude, Period,H_offset,Negative]
type Individual = list[Chromosome]
type Population = list[Individual]
type Gait = list[list[float]]
