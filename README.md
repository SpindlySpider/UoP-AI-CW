# AI Coursework

[project brief](https://ports-main-mdl-euwest2.s3.eu-west-2.amazonaws.com/62/d4/62d4e24dd98beac63eb499fd81b6c30d4bc3075a?response-content-disposition=inline%3B%20filename%3D%22M33174_item_1_cwnp_25.pdf%22&response-content-type=application%2Fpdf&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAWRN6GJFLZ774IX4M%2F20251021%2Feu-west-2%2Fs3%2Faws4_request&X-Amz-Date=20251021T164451Z&X-Amz-SignedHeaders=host&X-Amz-Expires=21549&X-Amz-Signature=c515ac668bdaddad2eda6eb2e4f85ee50c2734529eeaac2097361dcea44302cf)

## Genetic algorithm

### chromosome
Current chromosome encoding is a array of size 24 with floats as elements.
The floats represent radian angles of joints.
Every 3 elements correspond to a limb (coxa,femur,tibia).

An individual is a full gait, so a array of chromosomes.

### fitness function
Current implementation uses a sine as a target for coxa rotation over multiple frames. hopefully this will lead to a semi realistic movement.

### selection algorithm

### crossover

### mutation

## Neural network
