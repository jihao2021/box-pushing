import math
def local_to_global_transform(current_x, current_y, local_x, local_y, heading):
    updated_current_x = local_x * math.cos(math.radians(heading)) - local_y * math.sin(math.radians(heading)) + current_x
    updated_current_y = local_x * math.sin(math.radians(heading)) + local_y * math.cos(math.radians(heading)) + current_y
    return updated_current_x, updated_current_y

def global_to_local_transform(current_x, current_y, global_x, global_y, heading):
    updated_local_x = (global_x - current_x) * math.cos(math.radians(heading)) + (global_y - current_y) * math.sin(math.radians(heading))
    updated_local_y = - (global_x - current_x) * math.sin(math.radians(heading)) + (global_y - current_y) * math.cos(math.radians(heading))
    return updated_local_x, updated_local_y