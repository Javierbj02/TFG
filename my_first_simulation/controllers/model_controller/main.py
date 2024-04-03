# def round_numbers(real_position):
#     # Redondeamos la coordenada X a la más cercana que termine en 2 o en 7
#     # Redondeamos la coordenada X al más cercano que termine en 0 o en 5
#     if real_position[0] >= 0:
#         real_position[0] = round(real_position[0] / 10) * 10 + 2 if real_position[0] % 10 < 5 else (round(real_position[0] / 10) * 10 + 7 if real_position[0] % 10 == 5 else round(real_position[0] / 10) * 10 - 3)
#     else:
#         real_position[0] = round(real_position[0] / 10) * 10 + 3 if real_position[0] % 10 < 5 else (round(real_position[0] / 10) * 10 - 7 if real_position[0] % 10 == 5 else round(real_position[0] / 10) * 10 - 2)

#     real_position[1] = round(real_position[1] / 5) * 5
    
#     if (real_position[1] > 2779):
#         real_position[1] = real_position[1] - 1
    
#     # Check limits of position X
    
#     if (real_position[0] < -1437):
#         real_position[0] = -1437
#     elif (real_position[0] > 1357):
#         real_position[0] = 1357
#     elif (real_position[0] > -187 and real_position[0] < 107):
#         if (real_position[0] < -40):
#             real_position[0] = -187
#         else:
#             real_position[0] = 107
            
#     # Check limits of position Y
    
#     if (real_position[1] < 1155):
#         real_position[1] = 1155
#     elif (real_position[1] > 4029):
#         real_position[1] = 4029
#     elif (real_position[1] > 2405 and real_position[1] < 2779):
#         if (real_position[1] < 2592):
#             real_position[1] = 2405
#         else:
#             real_position[1] = 2779
    
#     return real_position

def round_numbers(real_position):
    # Redondear la coordenada X al más cercano que termine en 2 o en 7
    if real_position[0] >= 0:
        real_position[0] = round(real_position[0] / 10) * 10 + (2 if real_position[0] % 10 < 5 else (7 if real_position[0] % 10 == 5 else -3))
    else:
        real_position[0] = round(real_position[0] / 10) * 10 + (3 if real_position[0] % 10 < 5 else (-7 if real_position[0] % 10 == 5 else -2))

    # Redondear la coordenada Y al más cercano que termine en 0 o en 5
    real_position[1] = round(real_position[1] / 5) * 5
    
    # Si la coordenada Y es mayor a 2779, restar 1
    if real_position[1] > 2779:
        real_position[1] -= 1

    # Aplicar límites de posición X
    real_position[0] = max(-1437, min(1357, real_position[0]))
    if -187 < real_position[0] < 107:
        real_position[0] = -187 if real_position[0] < -40 else 107

    # Aplicar límites de posición Y
    real_position[1] = max(1155, min(4029, real_position[1]))
    if 2405 < real_position[1] < 2779:
        real_position[1] = 2405 if real_position[1] < 2592 else 2779

    return real_position


position = round_numbers([-758.0246871437575, 1798.9217383115995])
print(position) 