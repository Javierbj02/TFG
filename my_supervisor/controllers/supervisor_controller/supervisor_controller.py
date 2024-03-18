"""supervisor_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Supervisor

TIME_STEP = 64

# create the Robot instance.
robot = Supervisor()

i = 0

bb8_node = robot.getFromDef('BB-8')

translation_field = bb8_node.getField('translation')


# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(TIME_STEP) != -1:
    print("Position of BB-8: ", translation_field.getSFVec3f())
    if i == 0:
        new_value = [2.5, 0.0, 0.0]
        translation_field.setSFVec3f(new_value) 
           
    print("position: ", bb8_node.getPosition())
        
    i = i + 1

# Enter here exit cleanup code.
