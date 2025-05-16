name = """
Tabletop-Open-Microwave-v1,0.00%,0,10
Tabletop-Close-Cabinet-v1,0.00%,0,10
Tabletop-Balance-Pivot-WithBalls-v1,0.00%,0,10
Tabletop-Move-Balls-WithDustpan-v1,0.00%,0,10
Tabletop-Pull-Pivot-v1,0.00%,0,10
Tabletop-Rotate-Cube-v1,0.00%,0,10
Tabletop-Move-Cube-DynamicFriction-v1,0.00%,0,10
Tabletop-Move-Balls-WithPivot-v1,0.00%,0,10
Tabletop-Close-Door-WithObstacle-v1,0.00%,0,10
Tabletop-Put-Ball-IntoContainer-v1,0.00%,0,10
Tabletop-Pick-Bottle-v1,0.00%,0,10
Tabletop-Close-Drawer-WithLongObstacle-v1,0.00%,0,10
Tabletop-Merge-USB-v1,0.00%,0,10
Tabletop-Pick-Apple-v1,0.00%,0,10
Tabletop-Keep-Pivot-Balance-v1,0.00%,0,10
Tabletop-Pick-Cube-ToHolder-v1,0.00%,0,10
Tabletop-Pick-Book-FromShelf-v1,0.00%,0,10
Tabletop-Find-Book-FromShelf-v1,0.00%,0,10
Tabletop-Stack-Cubes-v1,0.00%,0,10
Tabletop-Find-Book-Black-v1,0.00%,0,10
Tabletop-Open-Trigger-v1,0.00%,0,10
Tabletop-Insert-Objects-WithShape-v1,0.00%,0,10
Tabletop-Open-Cabinet-WithSwitch-v1,0.00%,0,10
Tabletop-Pick-Pen-v1,0.00%,0,10
Tabletop-Close-Drawer-WithObstacle-v1,0.00%,0,10
Tabletop-Put-Fork-OnPlate-v1,0.00%,0,10
Tabletop-Insert-WithOrientation-v1,0.00%,0,10
Tabletop-Close-Cabinet-WithObstacle-v1,0.00%,0,10
Tabletop-Move-Cube-WithHolder-v1,0.00%,0,10
Tabletop-Open-Door-WithCabinet-v1,0.00%,0,10
Tabletop-Find-Dice-v1,0.00%,0,10
Tabletop-Lift-Book-v1,0.00%,0,10
Tabletop-Find-Cube-RedDown-v1,0.00%,0,10
Tabletop-Clean-For-Dinner-v1,0.00%,0,10
Tabletop-Finish-Hanobi-v1,0.00%,0,10
Tabletop-Close-Microwave-v1,0.00%,0,10
Tabletop-Open-Door-v1,0.00%,0,10
Tabletop-Move-Cross-WithStick-v1,0.00%,0,10
Tabletop-Insert-Conical-v1,0.00%,0,10
Tabletop-Open-Drawer-v1,0.00%,0,10
Tabletop-Open-Cabinet-v1,0.00%,0,10
Tabletop-Move-Line-WithStick-v1,0.00%,0,10
Tabletop-Open-Cabinet-WithDoor-v1,0.00%,0,10
Tabletop-Merge-Box-v1,0.00%,0,10
Tabletop-Rotate-Holder-v1,0.00%,0,10
Tabletop-Find-Cube-WithPivot-v1,10.00%,1,10
Tabletop-Move-Cube-WithPivot-v1,0.00%,0,10
Tabletop-Open-Cabinet-WithObstacle-v1,0.00%,0,10
Tabletop-Close-Drawer-v1,0.00%,0,10
Tabletop-Close-Door-v1,100.00%,10,10
Tabletop-Rotate-USB-v1,0.00%,0,10
"""

task_names = []
for line in name.strip().splitlines():
    if line.strip():  # skip empty lines
        task = line.split(",")[0].strip()
        task_names.append(task)

# Print result
for task in task_names:
    print(task)


import pickle as pkl

with open("skip_task.pkl", "wb") as f:
    pkl.dump(task_names, f)
