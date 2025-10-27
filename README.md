# AI Coursework

[project brief](https://ports-main-mdl-euwest2.s3.eu-west-2.amazonaws.com/62/d4/62d4e24dd98beac63eb499fd81b6c30d4bc3075a?response-content-disposition=inline%3B%20filename%3D%22M33174_item_1_cwnp_25.pdf%22&response-content-type=application%2Fpdf&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAWRN6GJFLZ774IX4M%2F20251021%2Feu-west-2%2Fs3%2Faws4_request&X-Amz-Date=20251021T164451Z&X-Amz-SignedHeaders=host&X-Amz-Expires=21549&X-Amz-Signature=c515ac668bdaddad2eda6eb2e4f85ee50c2734529eeaac2097361dcea44302cf)

## Genetic algorithm

### chromosome
Current chromosome encoding is a array of size 24 with floats as elements.
The floats represent radian angles of joints.
Every 3 elements correspond to a limb (coxa,femur,tibia).

An individual is a full gait, so a array of chromosomes.

### fitness function
#### coxa
Current idea is, each coxa joint in an individual is given a fitness value over time as a float. This value is then summed to get the over all fitness of the individiuals coxa.
Current implementation:
- an individual is a 2D array of [[angle1,angle2,...,angle24],...,<end number of gait>]
- a each coxa movements is extracted from the individual list:
    - first each coxa is extracted so this would be at indexs = [0,3,6,9,12,15,18,21] in the 24 array
    - these coxa will be stored in individual lists e.g. coxa1,coxa2,...,coxa8
    - for each frame in an individual the rotational value of the coxa is appended to its list.
    - this means that each coxas list will be the length of the gait
    - we then use a function to get the target rotation of this coxa, and calculate the error for the predicted value
    - we sum each coxas fitness to get  the overall fitness of each leg and then add these values together.
- The target value would be a rotation we want the spiders leg to move towards ideally the next leg in sequential order moves in the opposite direction:
    - for example the coxa has a range of motion of 30-70 degrees, so over time (each entry in the individual) we want the leg rotation to be moving between 30 and 70 degrees. 
    - we can achieve this using a [periodic function](https://www.mathsisfun.com/algebra/amplitude-period-frequency-phase-shift.html) like sine.
    - to get repeated values at certain "time" intervals
    - so for the target_rotation = `A sin(B(x)) + D`
    - where `A` is half of the range of min and max angle (so when sin = -1, it will go down to 30 and when sin=1 it goes up to 70) in this case 20 (40/2)
    - where `B` is the period of movement, basically controls the speed of movement.
    - where `D` is the mid point between 30 and 70 so 50
    - where `x` is the current frame in the gait (e.g. individual[frame][angle1])

    - additionally we can count the current number of the coxa join (e.g. coxa1,cox2) and then inverse the target to ensure that the rotation of the leg behind the current is going in the opposite direction, like in this [video](https://youtu.be/GtHzpX0FCFY)
    - for example at frame 22 (e.g. individual[22]) we look at the target rotation value, calculated using `coxa1_t=20*sin(0.5*22)+50` which equals `30.00019586898593` for coxa1, however for coxa2, as it is a even number we instead set the x to negative `coxa2_t=20*sin(0.5*-22)+50` which equals `69.99980413101407`, rounded up coxa 1 is aiming for min rotation (30) where as coxa 2 is aiming for maximum rotation (70)
  ##### picture of coxa 1 and 2 over time
<img width="623" height="637" alt="image" src="https://github.com/user-attachments/assets/6eadc1d5-36e3-4ae7-8fe2-84f085be5fd0" />
<img width="680" height="613" alt="image" src="https://github.com/user-attachments/assets/c49797e0-d623-4122-ac96-b954492b6af4" />

- note: according to [this paper](https://ieeexplore.ieee.org/document/4650677) the x-y (which this models rotation is limited to) can move 35 degree,  meaning the fitness function tracks the range of min and max `-17.5,17.5`


### selection algorithm
Ended up using a 3 way tournament

### crossover
Used uniform crossover as single point cross over lead to decreasing fitness

### mutation
Genes are mutated by adding or subtracting 0.5 from the current value.

### testing / results
To test an individual I added this to matlab
```
v = readmatrix('ga/results.txt')
A = deg2rad(v)

for idx = 1:300
    plot_spider_pose(A(idx,:))
    pause(0.001)
end
```
The CWD must include the results.txt which is produced by `main.py`, ideally this should be run from this repos root.
after running `main.py` and including the above snippet into matlab you should see a result like this:

## Neural network
