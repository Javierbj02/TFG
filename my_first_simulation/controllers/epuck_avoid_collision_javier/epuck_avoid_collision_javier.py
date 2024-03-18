"""epuck_avoid_collision_javier controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, DistanceSensor, Motor, Supervisor

TIME_STEP = 16
MAX_SPEED = 6.28

# create the Robot instance.
robot = Supervisor()

epuck = robot.getFromDef("EPUCK")


# Initilize the devices, which are the Distance sensors.
distance_sensors = []

sensor_names = [
    'ps0', 'ps1', 'ps2', 'ps3',
    'ps4', 'ps5', 'ps6', 'ps7'
]

for i in range(8):
    distance_sensors.append(robot.getDevice(sensor_names[i]))
    distance_sensors[i].enable(TIME_STEP)
    

# Initialize the motors
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')

leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

leftMotor.setVelocity(0.0)
leftMotor.setVelocity(0.0)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(TIME_STEP) != -1:
    # Read sensors outputs    
    sensor_values = [sensor.getValue() for sensor in distance_sensors]

    
    #print("PS Values: ", psValues) 
    
    # Detect Obstacles
    # right_obstacle = psValues[0] > 80.0 or psValues[1] > 80.0 or psValues[2] > 80.0
    # left_obstacle = psValues[5] > 80.0 or psValues[6] > 80.0 or psValues[7] > 80.0
    right_obstacle = any(value > 80 for value in sensor_values[:3])
    left_obstacle = any(value > 80 for value in sensor_values[5:])

    # Initialize motor speeds

    leftSpeed = 0.5 * MAX_SPEED
    rightSpeed = 0.5 * MAX_SPEED

    if left_obstacle:
        # turn right
        leftSpeed = 0.5 * MAX_SPEED
        rightSpeed = -0.5 * MAX_SPEED
    elif right_obstacle:
        # turn left
        leftSpeed = -0.5 * MAX_SPEED
        rightSpeed = 0.5 * MAX_SPEED
        
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
    
    
    # Position X and Y of the robot in the world from translate field
    #translation = epuck.getField("translation")
    print("------------------------------------")
    print("Position x: ", epuck.getPosition()[0]*1000)
    print("Position y: ", epuck.getPosition()[1]*1000)
    #print("Position z: ", epuck.getPosition()[2]*1000)
    print("------------------------------------")
    
# Enter here exit cleanup code.
