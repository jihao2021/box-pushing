from . import constant
from .patches import patches


def set_up_background_new(background_width,background_height):
    patch_list1 = []
    patch_list2 = []
    patch_list3 = []
    patch_list4 = []
    patch_list5 = []
    patch_list6 = []
    patch_list7 = []
    patch_list = []
    patch_color = constant.blue
    for x in range(0,background_width):
        for y in range(0, background_height):
            if x >= 0 and x <= background_width / 2 and y == background_height / 20:
                patch_list1.append(patches(x,y,color=patch_color))

            elif x >= background_width / 2 and x < 3 / 4 * background_width and y == (1/2 * background_height / background_width * x - 1/ 5 * background_height)  :
                patch_list2.append(patches(x,y, color=patch_color))

            elif  x >= 3 / 4 * background_width and x <= background_width and y == background_height * 7 / 40:
                patch_list3.append(patches(x,y, color=patch_color))

            elif x >= 0 and x <= background_width / 2  and y == 19/20 * background_height:
                patch_list4.append(patches(x,y, color=patch_color))

            elif x >= background_width / 2 and x < 3/ 4 * background_width and y == (-1/2 * background_height / background_width * x + (24/20 ) * background_height):
                patch_list5.append(patches(x,y, color=patch_color))

            elif  x >= 3 / 4 * background_width and x <= background_width and y == background_height * 33 / 40 :
                patch_list6.append(patches(x,y,color=patch_color))

            elif x == background_width / 30:
                patch_list7.append(patches(x,y,color=patch_color))

            else:
                pass
        for patch in patch_list1:
            print(patch.x,patch.y)
        for patch in patch_list2:
            print(patch.x,patch.y)
        for patch in patch_list3:
            print(patch.x,patch.y)
        for patch in patch_list4:
            print(patch.x,patch.y)
        for patch in patch_list5:
            print(patch.x,patch.y)
        patch_list += patch_list1
        patch_list += patch_list2
        patch_list += patch_list3
        patch_list += patch_list4
        patch_list += patch_list5
        patch_list += patch_list6
        patch_list += patch_list7
        patch_list_new = patch_list[::10]
    return patch_list_new

def set_up_background_new_new(background_width,background_height):
    patch_list1 = []
    patch_list2 = []
    patch_list3 = []
    patch_list4 = []
    patch_list5 = []
    patch_list6 = []
    patch_list7 = []
    patch_list = []
    patch_color = constant.white
    for x in range(0,background_width+1):
        for y in range(0,background_height+1):
            if x >= background_width / 30 and x <= background_width / 2 and y == background_height / 20:
                patch_list1.append(patches(x,y,color=patch_color))
            elif x >= background_width / 30 and x <= background_width / 2 and y == background_height / 20 * 19:
                patch_list2.append(patches(x, y, color=patch_color))
            elif x == background_width / 30 and y >= background_height / 20 and y <= background_height / 20 * 19:
                patch_list3.append(patches(x, y, color=patch_color))
            elif x >= background_width / 2 and x <= 3 / 4 * background_width and y == int(16 / 25 * x - 168):
                patch_list4.append(patches(x, y, color=patch_color))
            elif x >= background_width / 2 and x <= 3 / 4 * background_width and y == int(- 16 / 25 * x + 648):
                patch_list5.append(patches(x, y, color=patch_color))
            elif x >= 3 / 4 * background_width and x <= background_width and y == background_height * 1 / 4:
                patch_list6.append(patches(x, y, color=patch_color))
            elif x >= 3 / 4 * background_width and x <= background_width and y == background_height * 3 / 4:
                patch_list7.append(patches(x, y, color=patch_color))
            else:
                pass

    # for patch in patch_list1:
    #     print(patch.x,patch.y)
    # for patch in patch_list2:
    #     print(patch.x,patch.y)
    # for patch in patch_list3:
    #     print(patch.x,patch.y)
    # for patch in patch_list4:
    #     print(patch.x,patch.y)
    # for patch in patch_list5:
    #     print(patch.x,patch.y)
    # for patch in patch_list6:
    #     print(patch.x,patch.y)
    # for patch in patch_list7:
    #     print(patch.x,patch.y)

    patch_list += patch_list1
    patch_list += patch_list2
    patch_list += patch_list3
    patch_list += patch_list4
    patch_list += patch_list5
    patch_list += patch_list6
    patch_list += patch_list7

    patch_list_new = patch_list[::10]

    return patch_list_new

def set_up_background_new_new_white(background_width,background_height):
    patch_list1 = []
    patch_list2 = []
    patch_list3 = []
    patch_list4 = []
    patch_list5 = []
    patch_list6 = []
    patch_list7 = []
    patch_list = []
    patch_color = constant.black
    for x in range(0,background_width+1):
        for y in range(0,background_height+1):
            if x >= background_width / 30 and x <= background_width / 2 and y == background_height / 20:
                patch_list1.append(patches(x,y,color=patch_color))
            elif x >= background_width / 30 and x <= background_width / 2 and y == background_height / 20 * 19:
                patch_list2.append(patches(x, y, color=patch_color))
            elif x == background_width / 30 and y >= background_height / 20 and y <= background_height / 20 * 19:
                patch_list3.append(patches(x, y, color=patch_color))
            elif x >= background_width / 2 and x <= 3 / 4 * background_width and y == int(16 / 25 * x - 168):
                patch_list4.append(patches(x, y, color=patch_color))
            elif x >= background_width / 2 and x <= 3 / 4 * background_width and y == int(- 16 / 25 * x + 648):
                patch_list5.append(patches(x, y, color=patch_color))
            elif x >= 3 / 4 * background_width and x <= background_width and y == background_height * 1 / 4:
                patch_list6.append(patches(x, y, color=patch_color))
            elif x >= 3 / 4 * background_width and x <= background_width and y == background_height * 3 / 4:
                patch_list7.append(patches(x, y, color=patch_color))
            else:
                pass

    # for patch in patch_list1:
    #     print(patch.x,patch.y)
    # for patch in patch_list2:
    #     print(patch.x,patch.y)
    # for patch in patch_list3:
    #     print(patch.x,patch.y)
    # for patch in patch_list4:
    #     print(patch.x,patch.y)
    # for patch in patch_list5:
    #     print(patch.x,patch.y)
    # for patch in patch_list6:
    #     print(patch.x,patch.y)
    # for patch in patch_list7:
    #     print(patch.x,patch.y)

    patch_list += patch_list1
    patch_list += patch_list2
    patch_list += patch_list3
    patch_list += patch_list4
    patch_list += patch_list5
    patch_list += patch_list6
    patch_list += patch_list7

    patch_list_new = patch_list[::10]

    return patch_list_new