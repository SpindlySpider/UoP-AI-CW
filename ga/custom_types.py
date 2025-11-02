'''
Define chromosome, individual and population types for type hinting
The chromosome has five genes:
- Amplitude (float): defines the maximum rotation of the joints
- Period (float): define the speed of the gait
- H_offset (float): defines the horizontal offset of the gait
- Negative (bool): defines if the gait is inverted
- V_offset (float): defines the angle of the femur joints
'''


type Period = float 
type Amplitude = float
type H_offset = float
type V_offset = float
type Negative = bool
type Chromosome = tuple[Amplitude, Period,H_offset,Negative,V_offset]
type Individual = list[Chromosome]
type Population = list[Individual]
type Gait = list[list[float]]
